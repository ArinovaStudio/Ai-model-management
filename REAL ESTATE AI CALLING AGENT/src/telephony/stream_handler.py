"""Telephony Jitter Buffer & Audio Stream Handler.

Maintains sub-20ms frame alignment for incoming WebRTC/SIP RTP packets and
provides format conversions between WebRTC Linear PCM16, G.711 u-law, and A-law.
"""

from collections import deque
from typing import Optional
import numpy as np
from src.core.config import settings
from src.core.logging import logger

import warnings

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import audioop
except ImportError:
    try:
        import audioop_lts as audioop
    except ImportError:
        audioop = None


class JitterBuffer:
    """Jitter buffer ensuring smooth 20ms frame alignment and jitter compensation."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        target_delay_ms: int = 20,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.bytes_per_sample = 2  # 16-bit PCM = 2 bytes
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * self.bytes_per_sample)

        # Buffer queue
        self.buffer = bytearray()
        self.target_delay_ms = target_delay_ms
        self.min_buffer_size = int(sample_rate * (target_delay_ms / 1000.0) * self.bytes_per_sample)

        # Statistics
        self.packets_received = 0
        self.frames_popped = 0
        self.underruns = 0

    def push(self, packet: bytes) -> None:
        """Pushes raw incoming audio payload into the jitter buffer."""
        self.buffer.extend(packet)
        self.packets_received += 1

    def pop_frame(self) -> Optional[bytes]:
        """Pops a strictly aligned 20ms frame (640 bytes at 16kHz) if available."""
        if len(self.buffer) < self.frame_size:
            self.underruns += 1
            return None

        frame = bytes(self.buffer[: self.frame_size])
        del self.buffer[: self.frame_size]
        self.frames_popped += 1
        return frame

    def has_frame(self) -> bool:
        """Returns True if at least one complete 20ms frame is buffered."""
        return len(self.buffer) >= self.frame_size

    def flush(self) -> bytes:
        """Flushes and returns all currently buffered bytes."""
        remaining = bytes(self.buffer)
        self.buffer.clear()
        return remaining

    def reset(self) -> None:
        """Resets the jitter buffer state."""
        self.buffer.clear()
        self.packets_received = 0
        self.frames_popped = 0
        self.underruns = 0

    def clear(self) -> None:
        """Alias for reset() to support buffer clearing."""
        self.reset()


class AudioFormatConverter:
    """Converts between VoIP codecs: Linear PCM16, G.711 u-law (PCMU), and A-law (PCMA)."""

    @staticmethod
    def ulaw_to_pcm16(ulaw_bytes: bytes) -> bytes:
        """Converts G.711 u-law (8kHz) to 16-bit Linear PCM."""
        if audioop:
            return audioop.ulaw2lin(ulaw_bytes, 2)
        # NumPy lookup fallback
        u = np.frombuffer(ulaw_bytes, dtype=np.uint8)
        u = ~u
        t = ((u & 0x0F) << 3) + 0x84
        t <<= (u & 0x70) >> 4
        pcm = np.where(u & 0x80, 0x84 - t, t - 0x84).astype(np.int16)
        return pcm.tobytes()

    @staticmethod
    def pcm16_to_ulaw(pcm_bytes: bytes) -> bytes:
        """Converts 16-bit Linear PCM to G.711 u-law."""
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]
        if not pcm_bytes:
            return b""
        if audioop:
            return audioop.lin2ulaw(pcm_bytes, 2)
        # NumPy fallback approximation
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        sign = np.where(samples < 0, 0x7F, 0xFF)
        magnitude = np.abs(samples)
        magnitude = np.clip(magnitude + 0x84, 0, 0x7FFF)
        # Logarithmic compression
        exponent = np.floor(np.log2(magnitude) - 7).astype(np.uint8)
        exponent = np.clip(exponent, 0, 7)
        mantissa = (magnitude >> (exponent + 3)) & 0x0F
        ulaw = ~(sign ^ ((exponent << 4) | mantissa))
        return ulaw.astype(np.uint8).tobytes()

    @staticmethod
    def resample_pcm16(pcm_bytes: bytes, in_rate: int, out_rate: int) -> bytes:
        """Resamples 16-bit mono PCM between sample rates (e.g. 8kHz -> 16kHz)."""
        if in_rate == out_rate or not pcm_bytes:
            return pcm_bytes

        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]
        if not pcm_bytes:
            return b""

        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        num_target_samples = int(len(audio_int16) * out_rate / in_rate)
        if num_target_samples <= 0:
            return b""

        indices = np.linspace(0, len(audio_int16) - 1, num_target_samples)
        resampled = np.interp(indices, np.arange(len(audio_int16)), audio_int16).astype(np.int16)
        return resampled.tobytes()


class AudioStreamHandler:
    """Manages full-duplex audio stream with jitter buffering and format normalization."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.jitter_buffer = JitterBuffer(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
            target_delay_ms=settings.JITTER_BUFFER_MS,
        )

    def process_incoming_packet(
        self,
        packet: bytes,
        format_type: str = "pcm16",
        in_sample_rate: int = 16000,
    ) -> None:
        """Decodes and ingests an audio packet into the jitter buffer.

        Supported format_type: 'pcm16', 'pcmu', 'g711u'.
        """
        if format_type.lower() in ("pcmu", "g711u", "ulaw"):
            pcm_data = AudioFormatConverter.ulaw_to_pcm16(packet)
            if in_sample_rate != self.sample_rate:
                pcm_data = AudioFormatConverter.resample_pcm16(pcm_data, in_sample_rate, self.sample_rate)
        else:
            pcm_data = packet
            if in_sample_rate != self.sample_rate:
                pcm_data = AudioFormatConverter.resample_pcm16(pcm_data, in_sample_rate, self.sample_rate)

        self.jitter_buffer.push(pcm_data)

    def pop_aligned_frame(self) -> Optional[bytes]:
        """Pops the next aligned 20ms frame from the jitter buffer."""
        return self.jitter_buffer.pop_frame()

    def reset(self) -> None:
        self.jitter_buffer.reset()

    def clear(self) -> None:
        self.jitter_buffer.reset()
