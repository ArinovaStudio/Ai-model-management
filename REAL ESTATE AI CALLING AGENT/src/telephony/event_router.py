"""Full-Duplex Telephony WebSocket Event Router.

Processes binary audio frames over WebSocket:
Incoming Caller PCM -> Jitter Buffer -> VAD -> STT -> Dialogue Manager -> TTS -> Outgoing Audio Stream.
Catches unexpected drops and triggers mid-call disconnect recovery.
"""

import asyncio
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from src.core.logging import logger
from src.audio.vad import VoiceActivityDetector
from src.audio.stt import stt_engine
from src.audio.tts import tts_engine
from src.agent.dialogue_manager import dialogue_manager
from src.telephony.stream_handler import AudioStreamHandler, AudioFormatConverter
from src.telephony.session_manager import session_manager
from src.core.config import settings


async def handle_telephony_websocket(
    websocket: WebSocket,
    session_id: str,
    caller_phone: Optional[str] = None,
) -> None:
    """Manages full-duplex conversational voice turn-taking over WebSocket."""
    await websocket.accept()
    logger.info(f"WebSocket audio session established: {session_id}")

    # Ensure session exists
    session = session_manager.get_session(session_id)
    if not session:
        session = session_manager.create_session(session_id=session_id, caller_phone=caller_phone)

    stream_handler = AudioStreamHandler()
    vad = VoiceActivityDetector(
        sample_rate=settings.AUDIO_SAMPLE_RATE,
        silence_timeout_ms=settings.MAX_SILENCE_DURATION_MS,
        vad_aggressiveness=3,
    )

    playback_task: Optional[asyncio.Task] = None
    playback_interrupt_event = asyncio.Event()
    playback_active = False
    playback_is_greeting = False
    playback_started_at: Optional[float] = None
    playback_bytes_sent = 0
    is_processing_turn = False

    turn_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=3)

    async def _stream_audio_chunks(text: str, is_greeting: bool = False) -> None:
        """Stream TTS audio frames asynchronously to caller, aborting immediately upon barge-in."""
        import time
        from src.core.metrics import metrics_registry

        nonlocal playback_active, playback_started_at, playback_bytes_sent, playback_is_greeting

        playback_interrupt_event.clear()
        playback_active = True
        playback_is_greeting = is_greeting
        playback_started_at = None
        playback_bytes_sent = 0

        t_tts = time.perf_counter()
        tts_sr = tts_engine.sample_rate
        telephony_sr = settings.AUDIO_SAMPLE_RATE
        frame_size = int(telephony_sr * 0.02 * 2)

        try:
            try:
                await websocket.send_json({"type": "agent_speech_start"})
            except Exception:
                pass

            for audio_chunk in tts_engine.synthesize_stream(text):
                if playback_interrupt_event.is_set():
                    logger.info(f"[{session_id}] Audio playback cancelled due to barge-in.")
                    return

                if tts_sr != telephony_sr:
                    audio_chunk = AudioFormatConverter.resample_pcm16(
                        pcm_bytes=audio_chunk,
                        in_rate=tts_sr,
                        out_rate=telephony_sr,
                    )

                for offset in range(0, len(audio_chunk), frame_size):
                    if playback_interrupt_event.is_set():
                        logger.info(f"[{session_id}] Audio playback cancelled due to barge-in.")
                        return

                    chunk_slice = audio_chunk[offset : offset + frame_size]
                    await websocket.send_bytes(chunk_slice)
                    playback_bytes_sent += len(chunk_slice)
                    if playback_started_at is None:
                        playback_started_at = time.perf_counter()

                    await asyncio.sleep(0.001)

                if playback_interrupt_event.is_set():
                    return

            metrics_registry.record_stage_latency("tts", time.perf_counter() - t_tts)

            if not playback_interrupt_event.is_set():
                if is_greeting:
                    try:
                        await websocket.send_json({
                            "type": "greeting_metadata",
                            "transcript": "",
                            "agent_response": text,
                            "confidence": 1.0,
                        })
                        await websocket.send_json({"type": "greeting_end"})
                    except Exception:
                        pass
                else:
                    try:
                        await websocket.send_json({"type": "turn_end"})
                    except Exception:
                        pass
        except asyncio.CancelledError:
            logger.info(f"[{session_id}] Audio playback coroutine cancelled.")
            raise
        except Exception as e:
            logger.warning(f"[{session_id}] Audio playback streaming error: {e}")
        finally:
            playback_active = False

    async def turn_worker() -> None:
        nonlocal playback_task, playback_active, is_processing_turn
        import time
        from src.core.metrics import metrics_registry

        while True:
            turn_audio = await turn_queue.get()
            is_processing_turn = True
            try:
                # If agent was actively playing audio and received a turn boundary, only interrupt if NOT initial greeting
                if playback_task and not playback_task.done() and not playback_is_greeting:
                    playback_interrupt_event.set()
                    playback_task.cancel()
                    playback_task = None
                    playback_active = False
                    try:
                        await websocket.send_json({"type": "playback_interrupted"})
                    except Exception:
                        pass

                logger.info(
                    f"[{session_id}] VAD turn boundary detected ({len(turn_audio)} bytes). Transcribing..."
                )
                metrics_registry.record_vad_boundary()

                # 1. Faster-Whisper STT with entity safeguard (async to prevent event loop freeze)
                t_stt = time.perf_counter()
                detailed_stt = await stt_engine.transcribe_turn_detailed_async(turn_audio)
                metrics_registry.record_stage_latency("stt", time.perf_counter() - t_stt)

                transcript = detailed_stt["text"]
                confidence = detailed_stt["confidence"]
                requires_confirmation = detailed_stt.get("requires_confirmation", False)

                if not transcript or not transcript.strip():
                    logger.info(f"[{session_id}] Empty transcript. Skipping turn.")
                    continue

                logger.info(
                    f"[{session_id}] Caller: '{transcript}' (Confidence: {confidence * 100:.1f}%, RequiresConfirm: {requires_confirmation})"
                )
                session_manager.mark_turn_activity(session_id)

                # Check for natural call conclusion phrases
                u_text_lower = transcript.lower().strip()
                if any(ph in u_text_lower for ph in ["that's it", "thats it", "that's all", "thats all", "goodbye", "bye bye", "have a good day"]):
                    if session:
                        session["status"] = "completed"

                # Send real-time transcription immediately to frontend for instantaneous UI feedback
                try:
                    await websocket.send_json({
                        "type": "turn_transcription",
                        "transcript": transcript,
                    })
                except Exception:
                    pass

                # 2. Dialogue Manager & Tool Dispatch
                t_dm = time.perf_counter()
                turn_res = await dialogue_manager.process_turn(
                    user_input=transcript,
                    session_id=session_id,
                    lead_phone=session.get("caller_phone"),
                    confidence=confidence,
                    requires_confirmation=requires_confirmation,
                )
                metrics_registry.record_stage_latency("dialogue", time.perf_counter() - t_dm)

                agent_reply = turn_res["voice_response"]
                logger.info(f"[{session_id}] Agent: '{agent_reply}'")

                # Send text transcript metadata packet
                try:
                    await websocket.send_json({
                        "type": "turn_metadata",
                        "transcript": transcript,
                        "agent_response": agent_reply,
                        "tool_executed": turn_res.get("tool_executed"),
                        "tool_result": turn_res.get("tool_result"),
                        "session_state": turn_res.get("session_state"),
                        "confidence": confidence,
                    })
                except Exception:
                    pass

                # Reset VAD and clear jitter buffer right before speaking so queued mic frames don't trigger barge-in
                vad.reset_turn()
                stream_handler.reset()

                # 3. Stream Synthetic Audio Back to Caller with Barge-In capability
                logger.info(f"[{session_id}] Synthesizing streaming response...")
                playback_task = asyncio.create_task(_stream_audio_chunks(agent_reply, is_greeting=False))
                try:
                    await playback_task
                except asyncio.CancelledError:
                    pass

                # If playback completed naturally (NOT cancelled by a caller barge-in),
                # purge trailing ambient noise captured during normal agent speech.
                # If the caller barged in, PRESERVE the turn_queue so the caller's speech is processed!
                if not playback_interrupt_event.is_set():
                    while not turn_queue.empty():
                        try:
                            discarded = turn_queue.get_nowait()
                            turn_queue.task_done()
                            logger.info(f"[{session_id}] Purged ambient noise turn chunk ({len(discarded)} bytes) captured during playback.")
                        except (asyncio.QueueEmpty, ValueError):
                            break
                    vad.reset_turn()
                    stream_handler.reset()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{session_id}] Error in turn worker: {e}", exc_info=True)
            finally:
                is_processing_turn = False
                turn_queue.task_done()

    turn_worker_task: Optional[asyncio.Task] = None
    try:
        turn_worker_task = asyncio.create_task(turn_worker())

        # Send initial greeting to caller upon connection if call just started
        if session.get("turn_count", 0) == 0:
            # Check if there is an ingested lead context for this caller phone
            lead_greeting = None
            if caller_phone:
                try:
                    from src.agent.lead_service import lead_service
                    lead_ctx = await lead_service.get_lead_context_for_caller(caller_phone)
                    if lead_ctx and lead_ctx.get("name"):
                        lead_name = lead_ctx["name"]
                        niche = (lead_ctx.get("niche_preference") or "property").replace("_", " ")
                        loc = lead_ctx.get("preferred_location") or "Bengaluru"
                        lead_greeting = f"Hello {lead_name}! This is Aria from Prime Realty. I see you recently inquired about a {niche} in {loc}. How can I assist you today?"
                except Exception as e:
                    logger.debug(f"Lead lookup for greeting skipped: {e}")

            greeting_text = lead_greeting or "Hi, I'm Aria from Prime Realty. How can I help you find your ideal property today?"
            logger.info(f"[{session_id}] Streaming initial agent greeting: '{greeting_text}'")
            vad.reset_turn()
            stream_handler.reset()
            playback_task = asyncio.create_task(_stream_audio_chunks(greeting_text, is_greeting=True))

        while True:
            # Receive audio frame (binary bytes or text control frame)
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"]:
                data = msg["bytes"]
            elif "text" in msg:
                # Support text messages from browser testing console while preserving telephony audio compatibility
                import json
                try:
                    payload = json.loads(msg["text"])
                    if isinstance(payload, dict) and payload.get("type") in ("user_interrupt", "barge_in"):
                        if playback_task and not playback_task.done():
                            logger.info(f"[{session_id}] Client-initiated interruption. Halting active agent speech.")
                            playback_interrupt_event.set()
                            playback_task.cancel()
                            playback_task = None
                            playback_active = False
                        continue
                    if isinstance(payload, dict) and payload.get("type") in ("user_text", "text_message") and payload.get("text"):
                        user_text = payload["text"].strip()
                        if user_text:
                            if playback_task and not playback_task.done():
                                playback_interrupt_event.set()
                                playback_task.cancel()
                                playback_task = None
                                playback_active = False

                            session_manager.mark_turn_activity(session_id)
                            turn_res = await dialogue_manager.process_turn(
                                user_input=user_text,
                                session_id=session_id,
                                lead_phone=session.get("caller_phone"),
                                confidence=1.0,
                                requires_confirmation=False,
                            )
                            agent_reply = turn_res["voice_response"]
                            try:
                                await websocket.send_json({
                                    "type": "turn_metadata",
                                    "transcript": user_text,
                                    "agent_response": agent_reply,
                                    "tool_executed": turn_res.get("tool_executed"),
                                    "tool_result": turn_res.get("tool_result"),
                                    "session_state": turn_res.get("session_state"),
                                    "confidence": 1.0,
                                    "requires_confirmation": False,
                                })
                            except Exception:
                                pass
                            vad.reset_turn()
                            stream_handler.reset()
                            playback_task = asyncio.create_task(_stream_audio_chunks(agent_reply, is_greeting=False))
                except Exception:
                    pass
                continue
            else:
                continue

            # Ingest through jitter buffer for 20ms alignment
            stream_handler.process_incoming_packet(data)

            # Process all ready 20ms frames through VAD
            while stream_handler.jitter_buffer.has_frame():
                frame = stream_handler.pop_aligned_frame()
                if not frame:
                    break

                turn_completed, turn_audio = vad.process_pcm_chunk(frame)

                # Real-time barge-in detection:
                # In browser/web testing without hardware echo canceller, server-side VAD sees speaker bleed
                # as caller speech. To prevent cutting off agent sentences, barge-in is initiated explicitly
                # from client button or deliberate caller speech (100+ speech frames / 2.0s sustained speech).
                import time
                if (
                    playback_active
                    and not playback_is_greeting
                    and playback_bytes_sent >= 80000
                    and playback_started_at is not None
                    and (time.perf_counter() - playback_started_at) > 3.0
                    and vad.is_speech_active
                    and vad.active_speech_frames >= 100
                ):
                    logger.info(
                        f"[{session_id}] Real-time barge-in detected ({vad.active_speech_frames} speech frames)! Halting active agent speech."
                    )
                    playback_interrupt_event.set()
                    if playback_task and not playback_task.done():
                        playback_task.cancel()
                        playback_task = None
                    playback_active = False
                    try:
                        await websocket.send_json({"type": "playback_interrupted"})
                    except Exception:
                        pass

                if turn_completed and turn_audio:
                    if len(turn_audio) < 3200:
                        logger.info(
                            f"[{session_id}] Discarding short audio segment ({len(turn_audio)} bytes)."
                        )
                        continue

                    if is_processing_turn:
                        # During turn processing, reject ambient noise and brief hesitation segments under 2.0 seconds (32000 bytes)
                        if len(turn_audio) < 32000:
                            logger.info(
                                f"[{session_id}] Discarding short ambient noise segment ({len(turn_audio)} bytes) during turn processing."
                            )
                            continue
                        logger.info(
                            f"[{session_id}] Caller continued speaking during turn processing; queueing follow-up speech ({len(turn_audio)} bytes)."
                        )

                    # Put into queue, dropping oldest if queue is full
                    if turn_queue.full():
                        try:
                            stale_drop = turn_queue.get_nowait()
                            turn_queue.task_done()
                            logger.info(f"[{session_id}] Dropped oldest queue item ({len(stale_drop)} bytes) to make room for newer speech.")
                        except (asyncio.QueueEmpty, ValueError):
                            pass

                    await turn_queue.put(turn_audio)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for session {session_id}")
    except Exception as e:
        if "disconnect" in str(e).lower() or isinstance(e, RuntimeError):
            logger.info(f"WebSocket client disconnected for session {session_id}")
        else:
            logger.error(f"Error in telephony WebSocket loop for session {session_id}: {e}", exc_info=True)
    finally:
        if turn_worker_task and not turn_worker_task.done():
            turn_worker_task.cancel()
            try:
                await turn_worker_task
            except (asyncio.CancelledError, Exception):
                pass

        if playback_task and not playback_task.done():
            playback_interrupt_event.set()
            playback_task.cancel()
            playback_task = None
            playback_active = False

        current_session = session_manager.get_session(session_id)
        if current_session and current_session.get("status") == "active":
            # Session was abruptly terminated while active: trigger mid-call disconnect recovery
            logger.warning(f"Active session {session_id} closed abruptly! Triggering recovery handler...")
            recovery_task = asyncio.create_task(
                session_manager.handle_mid_call_disconnect(
                    session_id=session_id,
                    disconnect_reason="client_websocket_disconnect",
                )
            )
            while not recovery_task.done():
                try:
                    await asyncio.shield(recovery_task)
                except asyncio.CancelledError:
                    pass
            if not recovery_task.cancelled() and recovery_task.exception():
                logger.error(
                    f"Recovery task failed with exception: {recovery_task.exception()}",
                    exc_info=recovery_task.exception(),
                )

