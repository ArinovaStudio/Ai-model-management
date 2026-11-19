"""End-to-End Telephony & Dialogue Latency Profiler.

Profiles the complete turnaround lifecycle:
Caller Speech Ended -> VAD Trigger -> STT Transcription -> Dialogue Reasoning (LLM/Tools) -> TTS Time-to-First-Audio (TTFA) -> Full Audio Buffer.

Computes latency breakdown, percentiles (min, mean, p95), and real-time factor (RTF).
"""

import asyncio
import io
import time
import numpy as np
import soundfile as sf
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import logger
from src.core.config import settings
from src.audio.vad import VoiceActivityDetector
from src.audio.stt import stt_engine
from src.audio.tts import tts_engine
from src.agent.dialogue_manager import dialogue_manager


def generate_test_speech_pcm(phrase: str) -> tuple[bytes, float]:
    """Generates 16kHz 16-bit mono PCM test speech and returns (pcm_bytes, duration_sec)."""
    wav_bytes = tts_engine.synthesize_to_wav(phrase)
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16")
    if sr != 16000:
        num_samples = int(len(data) * 16000 / sr)
        indices = np.linspace(0, len(data) - 1, num_samples)
        data = np.interp(indices, np.arange(len(data)), data).astype(np.int16)
    duration = len(data) / 16000.0
    return data.tobytes(), duration


async def profile_single_turn(phrase: str, iteration: int) -> dict[str, Any]:
    """Profiles a complete turn across all audio and AI reasoning stages."""
    logger.info(f"\n--- Profiling Iteration #{iteration}: '{phrase}' ---")
    pcm_bytes, audio_duration = generate_test_speech_pcm(phrase)
    chunk_size = 640  # 20ms at 16kHz 16-bit mono

    vad = VoiceActivityDetector(sample_rate=16000, silence_timeout_ms=settings.MAX_SILENCE_DURATION_MS)

    # 1. VAD Ingestion & Silence Boundary Detection
    t0 = time.perf_counter()
    turn_audio = None
    turn_completed = False

    # Stream speech frames
    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i : i + chunk_size]
        if len(chunk) == chunk_size:
            done, audio = vad.process_pcm_chunk(chunk)
            if done:
                turn_completed = True
                turn_audio = audio

    # Feed silence frames until VAD emits end-of-turn
    silence_frame = b"\x00" * chunk_size
    silence_frames_fed = 0
    t_speech_end = time.perf_counter()

    while not turn_completed and silence_frames_fed < 50:
        done, audio = vad.process_pcm_chunk(silence_frame)
        silence_frames_fed += 1
        if done:
            turn_completed = True
            turn_audio = audio
            break

    t_vad_triggered = time.perf_counter()
    vad_latency_ms = (t_vad_triggered - t_speech_end) * 1000.0

    if not turn_audio:
        turn_audio = pcm_bytes

    # 2. Faster-Whisper STT Transcription
    t_stt_start = time.perf_counter()
    transcript, confidence = stt_engine.transcribe_turn(turn_audio)
    t_stt_end = time.perf_counter()
    stt_latency_ms = (t_stt_end - t_stt_start) * 1000.0

    # 3. Dialogue Reasoning & Tool Dispatch (LLM / RAG / Concurrency)
    session_id = f"profile_turn_{iteration}_{int(time.time())}"
    t_dm_start = time.perf_counter()
    dm_res = await dialogue_manager.process_turn(
        user_input=transcript or phrase,
        session_id=session_id,
        lead_phone="+919876543210",
        confidence=confidence,
    )
    t_dm_end = time.perf_counter()
    dm_latency_ms = (t_dm_end - t_dm_start) * 1000.0
    agent_response = dm_res["voice_response"]

    # 4. Piper TTS Streaming & Time-to-First-Audio (TTFA)
    t_tts_start = time.perf_counter()
    first_chunk_received = False
    ttfa_ms = 0.0
    total_audio_bytes = 0

    for chunk in tts_engine.synthesize_stream(agent_response):
        if not first_chunk_received:
            ttfa_ms = (time.perf_counter() - t_tts_start) * 1000.0
            first_chunk_received = True
        total_audio_bytes += len(chunk)

    t_tts_end = time.perf_counter()
    tts_total_ms = (t_tts_end - t_tts_start) * 1000.0

    # Total Perceived Turnaround Latency: Speech End -> First Spoken Agent Frame Ready
    perceived_e2e_turnaround_ms = vad_latency_ms + stt_latency_ms + dm_latency_ms + ttfa_ms

    logger.info(f"  VAD End-of-Turn Lag : {vad_latency_ms:.1f} ms ({silence_frames_fed * 20} ms silence window)")
    logger.info(f"  STT Transcription   : {stt_latency_ms:.1f} ms ('{transcript[:50]}...')")
    logger.info(f"  Dialogue & Tools    : {dm_latency_ms:.1f} ms (Tool: {dm_res.get('tool_executed')})")
    logger.info(f"  TTS Time-to-First   : {ttfa_ms:.1f} ms (TTFA)")
    logger.info(f"  TTS Total Synthesis : {tts_total_ms:.1f} ms ({total_audio_bytes} bytes streamed)")
    logger.info(f"  * Perceived Turnaround: {perceived_e2e_turnaround_ms:.1f} ms")

    # Clean up benchmark conversation turns to avoid polluting production DB
    try:
        from src.db.session import AsyncSessionLocal
        from src.db.models import ConversationTurn
        from sqlalchemy import delete
        async with AsyncSessionLocal() as sess:
            await sess.execute(delete(ConversationTurn).where(ConversationTurn.session_id == session_id))
            await sess.commit()
    except Exception:
        pass
    dialogue_manager.close_session(session_id)

    return {
        "vad_ms": vad_latency_ms,
        "stt_ms": stt_latency_ms,
        "dm_ms": dm_latency_ms,
        "ttfa_ms": ttfa_ms,
        "tts_total_ms": tts_total_ms,
        "perceived_turnaround_ms": perceived_e2e_turnaround_ms,
        "audio_bytes": total_audio_bytes,
    }


async def run_latency_profile():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 78)
    print("      REAL ESTATE AI VOICE AGENT - LATENCY & PERFORMANCE PROFILER      ")
    print("=" * 78)

    test_phrases = [
        "Hi, I am interested in Green Valley residential plots around North Hills.",
        "Can I check available visit slots for this Saturday?",
        "Yes, please confirm the booking for John Doe at 9876543210.",
    ]

    all_metrics: list[dict[str, Any]] = []

    for idx, phrase in enumerate(test_phrases, start=1):
        metrics = await profile_single_turn(phrase, idx)
        all_metrics.append(metrics)

    # Compute Statistics
    def stats(key: str) -> tuple[float, float, float]:
        vals = [m[key] for m in all_metrics]
        return min(vals), float(np.mean(vals)), float(np.percentile(vals, 95))

    v_min, v_mean, v_p95 = stats("vad_ms")
    s_min, s_mean, s_p95 = stats("stt_ms")
    d_min, d_mean, d_p95 = stats("dm_ms")
    t_min, t_mean, t_p95 = stats("ttfa_ms")
    p_min, p_mean, p_p95 = stats("perceived_turnaround_ms")

    print("\n" + "=" * 78)
    print("                   COMPONENT LATENCY BREAKDOWN (ms)                   ")
    print("=" * 78)
    print(f"{'Pipeline Stage':<34} | {'Min (ms)':<10} | {'Mean (ms)':<10} | {'p95 (ms)':<10}")
    print("-" * 78)
    print(f"{'1. VAD Speech Cutoff Lag':<34} | {v_min:>10.1f} | {v_mean:>10.1f} | {v_p95:>10.1f}")
    print(f"{'2. Faster-Whisper int8 STT':<34} | {s_min:>10.1f} | {s_mean:>10.1f} | {s_p95:>10.1f}")
    print(f"{'3. Dialogue Manager & RAG/Tools':<34} | {d_min:>10.1f} | {d_mean:>10.1f} | {d_p95:>10.1f}")
    print(f"{'4. Piper TTS First Audio (TTFA)':<34} | {t_min:>10.1f} | {t_mean:>10.1f} | {t_p95:>10.1f}")
    print("-" * 78)
    print(f"{'* PERCEIVED TURNAROUND (Speech End -> Voice)':<34} | {p_min:>10.1f} | {p_mean:>10.1f} | {p_p95:>10.1f}")
    print("=" * 78)
    print("Profiling report completed successfully.\n")


if __name__ == "__main__":
    asyncio.run(run_latency_profile())
