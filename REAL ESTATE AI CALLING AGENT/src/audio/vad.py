"""Voice Activity Detection (VAD) & Audio Buffering Subsystem.

Processes continuous 16kHz 16-bit mono PCM audio streams, detects speech
initiation and completion boundaries, and emits clean speech buffers when
caller silence (>1200ms) signals the completion of a conversational turn.
"""

from collections import deque
from typing import Tuple, Optional
import webrtcvad
import numpy as np
from src.core.config import settings
from src.core.logging import logger


class VoiceActivityDetector:
    """Detects voice activity and buffers conversational turns using WebRTC VAD."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        silence_timeout_ms: int = 1200,
        vad_aggressiveness: int = 3,
        pre_speech_padding_ms: int = 300,
        max_speech_duration_ms: int = 20000,
    ):
        """Initializes the VAD engine.

        Args:
            sample_rate: Sample rate in Hz (must be 8000, 16000, 32000, or 48000). Default: 16000.
            frame_duration_ms: Frame length in ms (10, 20, or 30). Default: 20ms.
            silence_timeout_ms: Milliseconds of continuous silence before triggering turn completion.
            vad_aggressiveness: WebRTC VAD mode (0=least aggressive, 3=most aggressive filtering).
            pre_speech_padding_ms: Milliseconds of pre-speech audio preserved in ring buffer.
            max_speech_duration_ms: Maximum continuous speech duration before forcing turn completion.
        """
        assert sample_rate in (8000, 16000, 32000, 48000), f"Unsupported sample rate: {sample_rate}"
        assert frame_duration_ms in (10, 20, 30), f"Frame duration must be 10, 20, or 30 ms."

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.silence_timeout_ms = silence_timeout_ms
        self.max_speech_duration_ms = max_speech_duration_ms
        self.bytes_per_sample = 2  # 16-bit PCM = 2 bytes
        self.frame_size_bytes = int(sample_rate * (frame_duration_ms / 1000.0) * self.bytes_per_sample)

        self.vad = webrtcvad.Vad(vad_aggressiveness)

        # Pre-speech ring buffer to avoid clipping first consonant/syllable
        padding_frames_count = max(1, pre_speech_padding_ms // frame_duration_ms)
        self.ring_buffer: deque[bytes] = deque(maxlen=padding_frames_count)

        # Silence frame threshold calculation
        self.required_silence_frames = max(1, silence_timeout_ms // frame_duration_ms)
        self.max_speech_frames = max(1, max_speech_duration_ms // frame_duration_ms)

        # State tracking
        self.is_speech_active = False
        self.silence_frames_count = 0
        self.active_speech_frames = 0
        self.consecutive_voiced_frames = 0
        self.speech_frames: list[bytes] = []
        self.incoming_byte_buffer = bytearray()

    def reset_turn(self) -> None:
        """Resets the conversational turn state without discarding incoming buffered bytes."""
        self.is_speech_active = False
        self.silence_frames_count = 0
        self.active_speech_frames = 0
        self.consecutive_voiced_frames = 0
        self.speech_frames.clear()
        self.ring_buffer.clear()

    def reset(self) -> None:
        """Resets the detector state completely for a new turn or call."""
        self.reset_turn()
        self.incoming_byte_buffer.clear()

    def process_pcm_chunk(self, chunk: bytes) -> Tuple[bool, Optional[bytes]]:
        """Processes incoming raw PCM bytes of arbitrary length.

        Returns:
            Tuple of (is_turn_complete, completed_speech_turn_bytes).
            If is_turn_complete is True, completed_speech_turn_bytes contains
            the complete turn audio, and internal turn state is automatically reset.
        """
        self.incoming_byte_buffer.extend(chunk)

        # Process all complete frames currently available
        while len(self.incoming_byte_buffer) >= self.frame_size_bytes:
            frame = bytes(self.incoming_byte_buffer[: self.frame_size_bytes])
            del self.incoming_byte_buffer[: self.frame_size_bytes]

            # Fast 16-bit PCM RMS energy calculation to reject ambient static / hum
            samples = np.frombuffer(frame, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if len(samples) > 0 else 0.0

            try:
                # Require frame energy to be above typical ambient silence noise floor (RMS >= 120)
                is_speech = (rms >= 120.0) and self.vad.is_speech(frame, self.sample_rate)
            except Exception as e:
                logger.warning(f"VAD classification error: {e}")
                is_speech = False

            if not self.is_speech_active:
                # Accumulate pre-speech ring buffer while waiting for voice
                self.ring_buffer.append(frame)
                if is_speech:
                    self.consecutive_voiced_frames += 1
                    if self.consecutive_voiced_frames >= 3:  # 60ms sustained speech
                        self.is_speech_active = True
                        self.silence_frames_count = 0
                        self.active_speech_frames = self.consecutive_voiced_frames
                        # Prepend pre-speech padding
                        self.speech_frames.extend(self.ring_buffer)
                else:
                    self.consecutive_voiced_frames = 0
            else:
                # In active speech
                self.speech_frames.append(frame)

                # Ceiling safeguard: force turn completion if continuous speech exceeds max duration
                if len(self.speech_frames) >= self.max_speech_frames:
                    total_turn_frames = len(self.speech_frames)
                    voiced_ratio = self.active_speech_frames / max(1, total_turn_frames)
                    if self.active_speech_frames < 8 or (total_turn_frames > 35 and voiced_ratio < 0.10):
                        self.reset_turn()
                        return False, None
                    logger.info(
                        f"VAD speech duration ceiling reached ({self.max_speech_duration_ms}ms). "
                        "Forcing conversational turn completion."
                    )
                    turn_audio = b"".join(self.speech_frames)
                    self.reset_turn()
                    return True, turn_audio

                if is_speech:
                    self.active_speech_frames += 1
                    self.silence_frames_count = 0
                else:
                    self.silence_frames_count += 1
                    if self.silence_frames_count >= self.required_silence_frames:
                        total_turn_frames = len(self.speech_frames)
                        voiced_ratio = self.active_speech_frames / max(1, total_turn_frames)
                        # Discard turns with fewer than 9 voiced frames (<180ms)
                        # or long turns (>30 frames = 600ms) with less than 14% voiced content (ambient drift / breathing)
                        if self.active_speech_frames < 9 or (total_turn_frames > 30 and voiced_ratio < 0.14):
                            self.reset_turn()
                            return False, None
                        turn_audio = b"".join(self.speech_frames)
                        self.reset_turn()
                        return True, turn_audio

        return False, None


# Global VAD instance configured with project settings
vad_detector = VoiceActivityDetector(
    sample_rate=settings.AUDIO_SAMPLE_RATE,
    silence_timeout_ms=settings.MAX_SILENCE_DURATION_MS,
)
