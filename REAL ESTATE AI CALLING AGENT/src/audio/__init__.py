"""Local speech pipelines: STT (Faster-Whisper), VAD, and TTS (Piper ONNX)."""

from src.audio.vad import VoiceActivityDetector, vad_detector
from src.audio.stt import SpeechToTextEngine, stt_engine
from src.audio.tts import TextToSpeechEngine, tts_engine

__all__ = [
    "VoiceActivityDetector",
    "vad_detector",
    "SpeechToTextEngine",
    "stt_engine",
    "TextToSpeechEngine",
    "tts_engine",
]
