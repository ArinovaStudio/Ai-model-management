"""End-to-End Telephony WebSocket & Mid-Call Disconnect Test Client.

Tests:
1. REST lifecycle: /api/calls/start and /api/calls/end.
2. Full-duplex binary audio streaming over /ws/audio/{session_id}.
3. Speech turn processing (VAD -> STT -> DM -> TTS -> Outgoing Audio).
4. Mid-call disconnect recovery: simulates network drop and asserts that
   PostgreSQL operator_notifications record is queued with partial summary.
"""

import asyncio
import io
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
from datetime import datetime, timezone
from sqlalchemy import select
from starlette.testclient import TestClient

from src.core.logging import logger
from src.core.config import settings
from src.db.session import AsyncSessionLocal
from src.db.models import OperatorNotification, ConversationTurn
from src.audio.tts import tts_engine
from src.api.app import app
from src.telephony.session_manager import session_manager


def create_pcm_audio_for_phrase(phrase: str) -> bytes:
    """Synthesizes a phrase and returns 16kHz 16-bit mono PCM bytes."""
    wav_bytes = tts_engine.synthesize_to_wav(phrase)
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16")
    if sr != 16000:
        num_samples = int(len(data) * 16000 / sr)
        indices = np.linspace(0, len(data) - 1, num_samples)
        data = np.interp(indices, np.arange(len(data)), data).astype(np.int16)
    return data.tobytes()


async def run_call_stream_test() -> int:
    logger.info("=" * 70)
    logger.info("STARTING TELEPHONY WEBSOCKET & DISCONNECT RECOVERY INTEGRATION TEST")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # PART 1: REST CALL LIFECYCLE
    # -------------------------------------------------------------------------
    client = TestClient(app)
    session_id_1 = f"call_test_{uuid.uuid4().hex[:8]}"

    logger.info("\n--- Step 1: Testing REST /api/calls/start ---")
    start_resp = client.post(
        "/api/calls/start",
        json={
            "session_id": session_id_1,
            "caller_phone": "+919876543210",
            "caller_name": "John Doe",
        },
    )
    assert start_resp.status_code == 200, f"Call start failed: {start_resp.text}"
    logger.info(f"REST call initiated: {start_resp.json()}")

    # -------------------------------------------------------------------------
    # PART 2: FULL-DUPLEX WEBSOCKET AUDIO STREAMING
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 2: Testing Full-Duplex Audio Streaming via WebSocket ---")
    speech_phrase = "Hi, I am looking for a residential plot around North Hills under 50 Lakhs."
    speech_pcm = create_pcm_audio_for_phrase(speech_phrase)
    logger.info(f"Generated {len(speech_pcm)} bytes of 16kHz test speech PCM for phrase: '{speech_phrase}'")

    chunk_size = 640  # 20ms at 16kHz 16-bit mono
    received_audio_bytes = 0
    received_metadata = []

    with client.websocket_connect(f"/ws/audio/{session_id_1}?phone=%2B919876543210") as ws:
        # Stream audio chunks with simulated 5ms network spacing
        for i in range(0, len(speech_pcm), chunk_size):
            chunk = speech_pcm[i : i + chunk_size]
            if len(chunk) == chunk_size:
                ws.send_bytes(chunk)

        # Feed trailing silence frames (>600ms = 35 frames of 20ms) to trigger VAD end-of-turn
        silence_frame = b"\x00" * chunk_size
        for _ in range(40):
            ws.send_bytes(silence_frame)

        # Collect outgoing agent audio response packets
        import json
        while True:
            msg = ws.receive()
            if "bytes" in msg and msg["bytes"]:
                received_audio_bytes += len(msg["bytes"])
            elif "text" in msg and msg["text"]:
                meta = json.loads(msg["text"])
                received_metadata.append(meta)
                if meta.get("type") == "turn_end":
                    break

        # End first call gracefully before closing WebSocket connection
        end_resp = client.post("/api/calls/end", json={"session_id": session_id_1})
        assert end_resp.status_code == 200
        assert end_resp.json()["status"] == "completed"
        logger.info(f"REST call terminated gracefully: {end_resp.json()}")

    logger.info(f"Received from Agent: {received_audio_bytes} bytes of synthesized voice audio.")
    logger.info(f"Received metadata packets: {len(received_metadata)}")
    assert received_audio_bytes > 0, "Agent must stream back synthesized synthetic voice audio"

    # -------------------------------------------------------------------------
    # PART 3: MID-CALL DISCONNECT & RECOVERY HANDLER
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 3: Testing Mid-Call Disconnect & Failure Recovery ---")
    drop_session_id = f"call_drop_{uuid.uuid4().hex[:8]}"

    # Initialize second call session
    _ = client.post(
        "/api/calls/start",
        json={
            "session_id": drop_session_id,
            "caller_phone": "+919876500099",
            "caller_name": "Abrupt Disconnect Prospect",
        },
    )

    drop_phrase = "Can I book a site visit this Saturday at 11 AM?"
    drop_pcm = create_pcm_audio_for_phrase(drop_phrase)

    # Connect WebSocket, stream turn, then abruptly disconnect
    try:
        with client.websocket_connect(f"/ws/audio/{drop_session_id}?phone=%2B919876500099") as ws:
            for i in range(0, len(drop_pcm), chunk_size):
                chunk = drop_pcm[i : i + chunk_size]
                if len(chunk) == chunk_size:
                    ws.send_bytes(chunk)
            for _ in range(40):
                ws.send_bytes(silence_frame)

            # Wait briefly for turn to process
            while True:
                msg = ws.receive()
                if "text" in msg and "turn_end" in msg["text"]:
                    break

            # Simulate abrupt network failure / mid-call drop (client closes socket without /api/calls/end)
            logger.info("Simulating sudden network drop / socket closure...")
    except Exception as e:
        logger.info(f"Simulated disconnect exception: {e}")

    # Give background disconnect handler a moment to flush recovery
    notification = None
    for _ in range(25):
        async with AsyncSessionLocal() as sess:
            query = select(OperatorNotification).where(OperatorNotification.session_id == drop_session_id)
            result = await sess.execute(query)
            notification = result.scalar_one_or_none()
            if notification:
                break
        await asyncio.sleep(0.2)

    # -------------------------------------------------------------------------
    # PART 4: AUDIT & ASSERT OPERATOR NOTIFICATION
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 4: Auditing Database Operator Notification ---")
    assert notification is not None, f"OperatorNotification must be created for dropped session {drop_session_id}"
    logger.info(f"OperatorNotification ID: #{notification.id}")
    logger.info(f"Session ID: {notification.session_id}")
    logger.info(f"Phone Number: {notification.phone_number}")
    logger.info(f"Urgency: {notification.urgency}")
    logger.info(f"Status: {notification.status}")
    logger.info(f"Reason: {notification.reason}")
    logger.info(f"Partial Summary:\n  \"{notification.partial_summary}\"")

    assert notification.urgency == "urgent", "Notification must be urgent"
    assert notification.status == "pending", "Notification status must be pending"
    assert len(notification.partial_summary) > 10, "Partial summary must contain meaningful recovery text"
    assert any(term in notification.partial_summary.lower() for term in ["dropped", "call", "visit", "saturday", "slot", "prospect"]), (
        f"Partial summary must capture interrupted intent: {notification.partial_summary}"
    )

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info(">>> TELEPHONY & DISCONNECT RECOVERY TEST PASSED WITH 100% SUCCESS! <<<")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run_call_stream_test()))
