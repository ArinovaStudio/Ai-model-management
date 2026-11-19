"""TTS Engine supporting Kokoro-82M ONNX and Piper ONNX for Low-Latency Voice Synthesis.

Defaults to Kokoro-82M ONNX for hyper-realistic, human-quality voice articulation
while automatically converting/resampling to 16,000 Hz 16-bit PCM for seamless compatibility
with telephony (Twilio / WebRTC / Browser WebSocket audio).

Preserves fallback to Piper ONNX if configured or if Kokoro model files are missing.
Streams audio chunks sentence-by-sentence to minimize Time-To-First-Audio (TTFA).
"""

import io
import re
import wave
from pathlib import Path
from typing import Generator, Iterable, Optional
import numpy as np

from src.core.config import settings
from src.core.logging import logger

# Optional import of kokoro_onnx, piper, torch, and zero-shot voice cloners
try:
    import kokoro_onnx
except ImportError:
    kokoro_onnx = None

try:
    import piper
except ImportError:
    piper = None

try:
    import torch
except ImportError:
    torch = None

try:
    from f5_tts.api import F5TTS
except ImportError:
    F5TTS = None


def resample_audio(audio_float32: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resamples float32 audio to target sample rate using high-speed interpolation."""
    orig_len = len(audio_float32)
    if orig_len == 0 or orig_sr == target_sr:
        return audio_float32
    target_len = int(round(orig_len * (float(target_sr) / float(orig_sr))))
    orig_indices = np.linspace(0, orig_len - 1, target_len)
    return np.interp(orig_indices, np.arange(orig_len), audio_float32)


def resample_24k_to_16k(audio_float32: np.ndarray) -> np.ndarray:
    """Downsamples 24,000 Hz float32 audio to 16,000 Hz using high-speed interpolation."""
    return resample_audio(audio_float32, 24000, 16000)


def float32_to_int16_bytes(audio_float32: np.ndarray) -> bytes:
    """Converts normalized float32 audio (-1.0 to 1.0) to 16-bit PCM signed integer bytes."""
    if len(audio_float32) == 0:
        return b""
    clipped = np.clip(audio_float32, -1.0, 1.0)
    int16_samples = (clipped * 32767.0).astype(np.int16)
    return int16_samples.tobytes()


class TextToSpeechEngine:
    """Streams natural synthetic voice audio from text streams using Kokoro ONNX, Piper ONNX, or Zero-Shot Voice Cloning."""

    def __init__(
        self,
        engine: str | None = None,
        voice: str | None = None,
    ):
        self.engine_type = engine or settings.TTS_ENGINE
        self.voice_name = voice or settings.KOKORO_DEFAULT_VOICE
        self._kokoro_model: Optional[object] = None
        self._piper_voice: Optional[object] = None
        self._zero_shot_model: Optional[object] = None
        self._zero_shot_device: Optional[str] = None

        # Sentence boundary delimiters for incremental chunking (preserves decimal numbers like 1.35 Cr and titles)
        self.sentence_delimiter = re.compile(r"(?<!\d)(?<!Mr)(?<!Ms)(?<!Dr)(?<!Rs)\s*([.!?;:\n]+)(?!\d)\s*")

    def _get_voice(self):
        """Backward-compatible pre-warming hook for the active TTS engine."""
        if self.engine_type == "zero_shot_clone":
            return self._get_zero_shot()
        elif self.engine_type == "kokoro":
            return self._get_kokoro()
        return self._get_piper()

    def _get_zero_shot(self):
        """Lazy-loads Zero-Shot Voice Cloning model (F5-TTS/CosyVoice) with GPU auto-acceleration."""
        if self._zero_shot_model is None:
            if F5TTS is None:
                logger.warning(
                    "f5-tts is not installed. Zero-shot cloning will fall back to Kokoro ONNX. "
                    "In production with GPU, run: pip install f5-tts"
                )
                return None

            device = settings.ZERO_SHOT_DEVICE
            if device == "auto":
                device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
            self._zero_shot_device = device

            logger.info(f"Initializing Zero-Shot Voice Cloning engine on device='{device}'...")
            try:
                self._zero_shot_model = F5TTS(device=device)
                logger.info("Zero-Shot Voice Cloning engine loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize Zero-Shot model on {device}: {e}. Falling back to Kokoro.")
                self._zero_shot_model = None
        return self._zero_shot_model

    def _get_kokoro(self):
        """Lazy-loads Kokoro-82M ONNX model into memory."""
        if self._kokoro_model is None:
            if kokoro_onnx is None:
                raise ImportError("kokoro-onnx is not installed. Please run `pip install kokoro-onnx`.")
            model_path = Path(settings.KOKORO_MODEL_PATH)
            voices_path = Path(settings.KOKORO_VOICES_PATH)
            if not model_path.exists() or not voices_path.exists():
                raise FileNotFoundError(
                    f"Kokoro model files not found at {model_path} or {voices_path}. "
                    "Run `python -m scripts.download_models` to fetch weights."
                )
            logger.info(f"Loading Kokoro-82M ONNX model from {model_path}...")
            self._kokoro_model = kokoro_onnx.Kokoro(str(model_path), str(voices_path))
            logger.info("Kokoro-82M ONNX TTS engine loaded successfully!")
        return self._kokoro_model

    def _get_piper(self):
        """Lazy-loads Piper ONNX voice model into memory as fallback or configured engine."""
        if self._piper_voice is None:
            if piper is None:
                raise ImportError("piper-tts is not installed. Please run `pip install piper-tts`.")
            model_path = Path(settings.PIPER_MODEL_PATH)
            config_path = Path(settings.PIPER_CONFIG_PATH)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Piper voice model not found at {model_path}. "
                    "Please run `python -m scripts.download_models` first."
                )
            logger.info(f"Loading Piper TTS voice model from {model_path}...")
            self._piper_voice = piper.PiperVoice.load(
                str(model_path),
                config_path=str(config_path) if config_path.exists() else None,
            )
            logger.info("Piper TTS voice loaded successfully.")
        return self._piper_voice

    @property
    def sample_rate(self) -> int:
        """Returns target audio sample rate expected by telephony pipeline (16,000 Hz)."""
        return settings.AUDIO_SAMPLE_RATE

    @property
    def channels(self) -> int:
        """Returns audio channel count (mono = 1)."""
        return 1

    @property
    def sample_width(self) -> int:
        """Returns sample width in bytes (16-bit PCM = 2 bytes)."""
        return 2

    def _clean_text_for_speech(self, text: str) -> str:
        """Cleans markdown, raw formatting, and verbalizes numbers/units for natural speech."""
        clean = text.strip()
        if not clean:
            return ""

        # 1. Clean markdown symbols
        clean = re.sub(r"[*#_`~>•]", " ", clean)
        clean = re.sub(r"^\s*-\s+", "", clean)

        # 2. Verbalize plot sizes & square footage into natural words
        clean = re.sub(r"\b1[,.]?200\s*(?:sqft|sq\.?\s*ft|square\s+feet)?\b", "twelve hundred square feet", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b1[,.]?500\s*(?:sqft|sq\.?\s*ft|square\s+feet)?\b", "fifteen hundred square feet", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b2[,.]?400\s*(?:sqft|sq\.?\s*ft|square\s+feet)?\b", "twenty-four hundred square feet", clean, flags=re.IGNORECASE)
        clean = re.sub(r"(\d+)\s*(?:sqft|sq\.?\s*ft)\b", r"\1 square feet", clean, flags=re.IGNORECASE)
        clean = re.sub(r"per\s+(?:sqft|sq\.?\s*ft)\b", "per square foot", clean, flags=re.IGNORECASE)

        # 3. Verbalize Indian currency
        clean = re.sub(r"(?:INR|Rs\.?)\s*48[,.]00[,.]000\b", "48 lakhs", clean, flags=re.IGNORECASE)
        clean = re.sub(r"(?:INR|Rs\.?)\s*85[,.]00[,.]000\b", "85 lakhs", clean, flags=re.IGNORECASE)
        clean = re.sub(r"(?:INR|Rs\.?)\s*62[,.]50[,.]000\b", "62.5 lakhs", clean, flags=re.IGNORECASE)
        clean = re.sub(r"(?:INR|Rs\.?)\s*4[,.]000\s*(?:per\s+square\s+foot|per\s+sqft)?\b", "4 thousand rupees per square foot", clean, flags=re.IGNORECASE)
        clean = re.sub(r"(?:INR|Rs\.?)\s*(\d+),00,000\b", r"\1 lakhs", clean, flags=re.IGNORECASE)
        clean = re.sub(r"(?:INR|Rs\.?)\s*(\d+),(\d+),000\b", r"\1 point \2 lakhs", clean, flags=re.IGNORECASE)
        clean = re.sub(r"(?:INR|Rs\.?)\s*(\d+),000\b", r"\1 thousand rupees", clean, flags=re.IGNORECASE)
        clean = re.sub(r"(?:INR|Rs\.?)\s*(\d+(?:\.\d+)?)\s*(?:Crores?|Cr)\b", r"\1 crore", clean, flags=re.IGNORECASE)
        clean = re.sub(r"(?:INR|Rs\.?)\s*(\d+(?:\.\d+)?)\s*(?:Lakhs?|Lacs?|L)\b", r"\1 lakhs", clean, flags=re.IGNORECASE)

        # 4. Dimensions (e.g. 30 ft x 40 ft)
        clean = re.sub(r"(\d+)\s*(?:ft|feet)?\s*[xX*]\s*(\d+)\s*(?:ft|feet)\b", r"\1 feet by \2 feet", clean, flags=re.IGNORECASE)

        # 5. Clean RERA serial codes & acronyms
        clean = re.sub(r"PRM/KA/RERA/[A-Za-z0-9/]+", "R-E-R-A approved registration", clean)
        clean = re.sub(r"\b3BHK\b", "three B-H-K", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b4BHK\b", "four B-H-K", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b5BHK\b", "five B-H-K", clean, flags=re.IGNORECASE)

        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def synthesize_sentence(self, sentence: str) -> bytes:
        """Synthesizes a single sentence into 16kHz 16-bit PCM audio bytes."""
        clean_text = self._clean_text_for_speech(sentence)
        if not clean_text:
            return b""

        # Attempt Zero-Shot Voice Cloning if configured and available
        if self.engine_type == "zero_shot_clone":
            zero_shot = self._get_zero_shot()
            if zero_shot is not None:
                try:
                    ref_audio = settings.ZERO_SHOT_REF_AUDIO_PATH
                    ref_text = settings.ZERO_SHOT_REF_TRANSCRIPT
                    wav, sr, _ = zero_shot.infer(
                        ref_file=ref_audio,
                        ref_text=ref_text,
                        gen_text=clean_text,
                    )
                    if hasattr(wav, "numpy"):
                        wav = wav.numpy()
                    if isinstance(wav, np.ndarray):
                        if wav.ndim > 1:
                            wav = wav.squeeze()
                        wav = wav.astype(np.float32)
                        if sr != self.sample_rate:
                            wav = resample_audio(wav, sr, self.sample_rate)
                        return float32_to_int16_bytes(wav)
                except Exception as e:
                    logger.warning(f"Zero-shot voice cloning error ({e}); falling back to Kokoro.")

        # Attempt Kokoro synthesis if configured or fallback
        if self.engine_type in ("kokoro", "zero_shot_clone"):
            try:
                kokoro = self._get_kokoro()
                # Kokoro outputs float32 at 24000 Hz
                samples_float32, orig_sr = kokoro.create(
                    clean_text,
                    voice=self.voice_name,
                    speed=settings.KOKORO_SPEED,
                    lang="en-us",
                )
                if orig_sr == 24000 and self.sample_rate == 16000:
                    samples_float32 = resample_24k_to_16k(samples_float32)

                return float32_to_int16_bytes(samples_float32)
            except Exception as e:
                logger.warning(f"Kokoro TTS synthesis encountered error ({e}); falling back to Piper.")

        # Piper ONNX fallback or explicit configuration
        voice = self._get_piper()
        pcm_chunks = []
        for chunk in voice.synthesize(clean_text):
            if chunk.audio_int16_bytes:
                pcm_chunks.append(chunk.audio_int16_bytes)

        return b"".join(pcm_chunks)

    def synthesize_stream(
        self,
        text_stream: Iterable[str] | str,
    ) -> Generator[bytes, None, None]:
        """Incremental sentence-by-sentence audio streaming generator.

        Accepts either a single string or an iterable/generator of tokens/text chunks.
        Splits text at natural sentence boundaries and yields raw 16-bit PCM audio chunks
        immediately as each sentence is synthesized, minimizing TTFA.
        """
        if isinstance(text_stream, str):
            text_stream = [text_stream]

        buffer = ""
        for chunk in text_stream:
            buffer += chunk
            # Split buffer into complete sentences and trailing remainder
            parts = self.sentence_delimiter.split(buffer)
            # If length >= 3, we have at least one complete sentence + delimiter
            while len(parts) >= 3:
                sentence = (parts[0] + parts[1]).strip()
                parts = parts[2:]
                buffer = "".join(parts)
                if sentence:
                    pcm_audio = self.synthesize_sentence(sentence)
                    if pcm_audio:
                        yield pcm_audio

        # Flush any remaining text in buffer
        remaining = buffer.strip()
        if remaining:
            pcm_audio = self.synthesize_sentence(remaining)
            if pcm_audio:
                yield pcm_audio

    def synthesize_to_bytes(self, text: str) -> bytes:
        """Synthesizes full text and returns complete concatenated raw PCM bytes."""
        return b"".join(self.synthesize_stream(text))

    def synthesize_to_wav(self, text: str, output_path: str | Path | None = None) -> bytes:
        """Synthesizes text into a standard RIFF/WAV audio container (16kHz, 16-bit mono)."""
        pcm_bytes = self.synthesize_to_bytes(text)
        sr = self.sample_rate

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(sr)
            wav_file.writeframes(pcm_bytes)

        wav_bytes = wav_buffer.getvalue()

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(wav_bytes)
            logger.info(f"Saved synthesized WAV audio ({len(wav_bytes)} bytes) to {out_file}")

        return wav_bytes

    def synthesize_to_file(self, text: str, output_path: str | Path) -> bytes:
        """Convenience alias to synthesize speech directly to an on-disk WAV file."""
        return self.synthesize_to_wav(text, output_path=output_path)


# Global TTS engine singleton
tts_engine = TextToSpeechEngine()
