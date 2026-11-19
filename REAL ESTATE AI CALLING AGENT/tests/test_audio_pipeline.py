"""Tests for Audio Pipeline: VAD, Faster-Whisper STT, and Piper TTS."""

import pytest
import numpy as np
from src.audio.vad import VoiceActivityDetector
from src.audio.stt import SpeechToTextEngine
from src.audio.tts import TextToSpeechEngine


def test_vad_silence_cutoff_and_buffering():
    """Verifies VAD silence detection and speech turn boundary emission."""
    vad = VoiceActivityDetector(sample_rate=16000, silence_timeout_ms=600)
    frame_size = 640  # 20ms at 16kHz 16-bit mono

    # Synthetic tone resembling speech formant (400Hz sine wave)
    t = np.linspace(0, 0.02, 320, endpoint=False)
    speech_frame = (np.sin(2 * np.pi * 400 * t) * 20000).astype(np.int16).tobytes()
    silence_frame = b"\x00" * frame_size

    turn_completed = False
    emitted_audio = None

    # Feed 20 frames of speech (400ms)
    for _ in range(20):
        done, audio = vad.process_pcm_chunk(speech_frame)
        if done:
            turn_completed = True
            emitted_audio = audio

    # Feed silence frames until cutoff is reached (up to 45 frames = 900ms)
    for _ in range(45):
        done, audio = vad.process_pcm_chunk(silence_frame)
        if done:
            turn_completed = True
            emitted_audio = audio
            break

    assert turn_completed is True
    assert emitted_audio is not None
    assert len(emitted_audio) > 0


def test_tts_streaming_and_wav_generation():
    """Verifies that Piper TTS streams audio chunks and generates valid WAV files."""
    tts = TextToSpeechEngine()
    test_text = "Your site visit is scheduled for Saturday at 11 AM."

    chunks = list(tts.synthesize_stream(test_text))
    assert len(chunks) > 0, "TTS streaming must yield at least one audio chunk"
    assert all(isinstance(c, bytes) and len(c) > 0 for c in chunks)

    wav_bytes = tts.synthesize_to_wav(test_text)
    assert wav_bytes.startswith(b"RIFF"), "Generated audio must be a valid RIFF/WAV file"
    assert len(wav_bytes) > 1000


def test_stt_transcription_and_entity_confidence():
    """Verifies Faster-Whisper transcription and entity confidence safeguard."""
    stt = SpeechToTextEngine()
    tts = TextToSpeechEngine()

    sample_phrase = "Can I book a site visit for tomorrow at 2 PM?"
    wav_bytes = tts.synthesize_to_wav(sample_phrase)

    # Standard transcription
    text, conf = stt.transcribe_turn(wav_bytes)
    assert len(text) > 0
    assert conf > 0.5
    clean_text = text.lower()
    assert any(term in clean_text for term in ["book", "visit", "tomorrow", "2"])

    # Detailed analysis
    detailed = stt.transcribe_turn_detailed(wav_bytes)
    assert "critical_entities" in detailed
    assert len(detailed["critical_entities"]) > 0

    # Test uncertainty safeguard flag triggering on strict threshold
    strict_stt = SpeechToTextEngine(confidence_threshold=0.999)
    strict_res = strict_stt.transcribe_turn_detailed(wav_bytes)
    assert strict_res["requires_confirmation"] is True
