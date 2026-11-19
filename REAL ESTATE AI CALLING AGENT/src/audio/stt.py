"""Faster-Whisper Speech-to-Text (STT) Subsystem with Entity Confidence Safeguard.

Runs locally on CPU using int8 quantization for ultra-low latency transcription.
Scans transcription for critical transactional entities (budget figures, phone numbers,
dates, property niches) and enforces explicit verbal confirmation if confidence drops below 0.70.
"""

import io
import math
import re
from typing import Any, Tuple, Optional
import numpy as np
from faster_whisper import WhisperModel
from src.core.config import settings
from src.core.logging import logger


class SpeechToTextEngine:
    """CPU-optimized Faster-Whisper STT engine with entity confidence verification."""

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        confidence_threshold: float | None = None,
        download_root: str | None = None,
    ):
        self.model_size = model_size or settings.WHISPER_MODEL_SIZE
        self.device = device or settings.WHISPER_DEVICE
        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.WHISPER_CONFIDENCE_THRESHOLD
        )
        self.download_root = download_root or settings.WHISPER_DOWNLOAD_ROOT

        self._model: Optional[WhisperModel] = None

        # Regex patterns for critical business entities requiring high confidence
        self.critical_entity_patterns = [
            # Phone numbers and digit sequences
            re.compile(r"^\d{4,12}$"),
            # Monetary & budget indicators
            re.compile(r"\b(lakh|lakhs|crore|crores|thousand|million|k|inr|rs|rupees?)\b", re.IGNORECASE),
            # Calendar and time constraints
            re.compile(r"\b(saturday|sunday|monday|tuesday|wednesday|thursday|friday|tomorrow|today|morning|evening|afternoon|o'clock|\d{1,2}\s*(?:am|pm)|a\.m\.|p\.m\.)\b", re.IGNORECASE),
        ]

    def _get_model(self) -> WhisperModel:
        """Lazy-loads the Faster-Whisper model into memory."""
        if self._model is None:
            logger.info(
                f"Loading Faster-Whisper STT (size='{self.model_size}', "
                f"device='{self.device}', compute_type='{self.compute_type}')..."
            )
            self._model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
                cpu_threads=4,
            )
            logger.info("Faster-Whisper model loaded successfully.")
        return self._model

    def _convert_audio_to_numpy(self, audio_data: bytes | np.ndarray) -> np.ndarray:
        """Converts raw 16kHz 16-bit mono PCM bytes or WAV container to float32 numpy array."""
        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype != np.float32:
                return audio_data.astype(np.float32)
            return audio_data

        # Check if WAV container header is present (RIFF)
        if audio_data.startswith(b"RIFF") and len(audio_data) > 44:
            import soundfile as sf
            with io.BytesIO(audio_data) as f:
                data, sr = sf.read(f, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                return data

        # Raw 16-bit mono PCM
        if len(audio_data) % 2 != 0:
            audio_data = audio_data[:-1]
        if not audio_data:
            return np.array([], dtype=np.float32)
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    def transcribe_turn(self, audio_bytes: bytes) -> Tuple[str, float]:
        """Transcribes audio turn and returns text with average confidence score.

        Args:
            audio_bytes: Raw 16kHz 16-bit mono PCM bytes or WAV audio.

        Returns:
            Tuple of (transcribed_text, average_confidence_score).
        """
        result = self.transcribe_turn_detailed(audio_bytes)
        return result["text"], result["confidence"]

    def transcribe_turn_detailed(self, audio_data: bytes | np.ndarray) -> dict[str, Any]:
        """Full transcription analysis with word-level confidence and entity safety verification."""
        audio_np = self._convert_audio_to_numpy(audio_data)
        if audio_np is None or len(audio_np) == 0:
            return {
                "text": "",
                "confidence": 0.0,
                "requires_confirmation": False,
                "critical_entities": [],
                "lowest_critical_score": 0.0,
                "duration_seconds": 0.0,
            }

        model = self._get_model()

        segments, info = model.transcribe(
            audio_np,
            beam_size=1,  # Greedy decoding for minimal latency on CPU
            word_timestamps=True,
            language="en",
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )

        full_text_parts = []
        all_word_scores: list[float] = []
        critical_entity_scores: list[float] = []
        critical_entities_found: list[str] = []

        for segment in segments:
            text_clean = segment.text.strip()
            if text_clean:
                full_text_parts.append(text_clean)

            # Evaluate word-level probabilities
            if segment.words:
                for word_info in segment.words:
                    word = word_info.word.strip()
                    clean_word = re.sub(r"[^\w]", "", word).lower()
                    # Faster-whisper word.probability is probability in [0.0, 1.0]
                    prob = float(word_info.probability)
                    all_word_scores.append(prob)

                    # Check if this word belongs to a critical transactional entity
                    is_critical = any(p.search(clean_word) for p in self.critical_entity_patterns)
                    if is_critical and clean_word:
                        critical_entity_scores.append(prob)
                        critical_entities_found.append(f"{clean_word}:{prob:.2f}")

        full_text = " ".join(full_text_parts).strip()
        overall_confidence = float(np.mean(all_word_scores)) if all_word_scores else 0.90

        # Repetition loop hallucination suppression
        words = full_text.split()
        if len(words) > 12:
            trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
            if trigrams and max(trigrams.count(t) for t in trigrams) >= 4:
                logger.warning(f"Whisper repetition loop hallucination detected. Suppressing duplicate phrases.")
                seen = set()
                deduped = []
                for w in words:
                    if w not in seen:
                        seen.add(w)
                        deduped.append(w)
                full_text = " ".join(deduped)

        # Filter out common Whisper silence hallucinations and low-confidence ambient fragments
        norm_text = re.sub(r"[^\w\s]", "", full_text.lower()).strip()
        common_hallucinations = {"okay", "okay okay", "okay okay okay", "thank you", "thanks", "you", "bye", "bye bye", "uh", "um"}
        if norm_text in common_hallucinations and overall_confidence < 0.65:
            logger.info(f"Filtering out ambient silence hallucination token: '{full_text}' (conf: {overall_confidence:.1%})")
            full_text = ""
        elif overall_confidence < 0.35 and len(words) <= 4 and not critical_entities_found:
            logger.info(f"Filtering out low-confidence background noise token: '{full_text}' (conf: {overall_confidence:.1%})")
            full_text = ""
        elif overall_confidence < 0.20 and len(words) <= 3:
            logger.info(f"Filtering out low-confidence background noise token: '{full_text}' (conf: {overall_confidence:.1%})")
            full_text = ""

        # Entity confidence safeguard check
        requires_confirmation = False
        lowest_critical_score = 1.0
        if critical_entity_scores:
            lowest_critical_score = min(critical_entity_scores)
            avg_critical_score = float(np.mean(critical_entity_scores))
            if avg_critical_score < self.confidence_threshold or lowest_critical_score < 0.50:
                requires_confirmation = True
                logger.warning(
                    f"Speech uncertainty safeguard triggered! Critical entities: {critical_entities_found}. "
                    f"Lowest score: {lowest_critical_score:.2f} < threshold ({self.confidence_threshold})"
                )

        return {
            "text": full_text,
            "confidence": overall_confidence,
            "requires_confirmation": requires_confirmation,
            "critical_entities": critical_entities_found,
            "lowest_critical_score": lowest_critical_score,
            "duration_seconds": info.duration if hasattr(info, "duration") else 0.0,
        }

    async def transcribe_turn_detailed_async(self, audio_data: bytes | np.ndarray) -> dict[str, Any]:
        """Asynchronously transcribes audio turn in a worker thread to prevent event loop blocking."""
        import asyncio
        return await asyncio.to_thread(self.transcribe_turn_detailed, audio_data)


# Global STT engine singleton
stt_engine = SpeechToTextEngine()
