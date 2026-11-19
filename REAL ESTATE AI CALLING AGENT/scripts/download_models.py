"""Audio Model Downloader & Cache Warmup Script.

Downloads and validates:
1. Faster-Whisper base.en model (int8 quantized for local CPU execution).
2. Piper TTS en_US-lessac-medium ONNX voice model and configuration JSON.
"""

import sys
from pathlib import Path
import httpx
from src.core.config import settings
from src.core.logging import logger


PIPER_ONNX_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
PIPER_JSON_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

KOKORO_ONNX_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"


def download_file(url: str, dest_path: Path) -> None:
    """Downloads a file over HTTP with progress logging."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"Model file already exists: {dest_path} ({dest_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return

    logger.info(f"Downloading from {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            total_bytes = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(temp_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes > 0 and downloaded % (1024 * 1024 * 5) < 65536:
                        percent = (downloaded / total_bytes) * 100
                        logger.info(f"  Downloaded {downloaded / 1024 / 1024:.1f}/{total_bytes / 1024 / 1024:.1f} MB ({percent:.1f}%)")

    temp_path.rename(dest_path)
    logger.info(f"Successfully downloaded: {dest_path} ({dest_path.stat().st_size / 1024 / 1024:.2f} MB)")


def download_piper_models() -> tuple[Path, Path]:
    """Downloads Piper TTS ONNX model and JSON config."""
    onnx_path = Path(settings.PIPER_MODEL_PATH)
    json_path = Path(settings.PIPER_CONFIG_PATH)

    logger.info("--- Downloading Piper TTS Voice Model ---")
    download_file(PIPER_ONNX_URL, onnx_path)
    download_file(PIPER_JSON_URL, json_path)

    # Verify loading with Piper
    import piper
    logger.info(f"Validating Piper model loading from {onnx_path}...")
    voice = piper.PiperVoice.load(str(onnx_path), config_path=str(json_path))
    logger.info("Piper voice model successfully validated!")
    return onnx_path, json_path


def download_whisper_models() -> None:
    """Downloads and caches the Faster-Whisper base.en int8 model."""
    logger.info("--- Downloading Faster-Whisper Model ---")
    from faster_whisper import WhisperModel

    download_root = Path(settings.WHISPER_DOWNLOAD_ROOT)
    download_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Initializing WhisperModel(size='{settings.WHISPER_MODEL_SIZE}', "
        f"device='{settings.WHISPER_DEVICE}', compute_type='{settings.WHISPER_COMPUTE_TYPE}')..."
    )
    model = WhisperModel(
        model_size_or_path=settings.WHISPER_MODEL_SIZE,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        download_root=str(download_root),
    )
    logger.info("Faster-Whisper model initialized and verified successfully!")


def download_kokoro_models() -> tuple[Path, Path]:
    """Downloads Kokoro-82M ONNX model and voices.bin."""
    onnx_path = Path(settings.KOKORO_MODEL_PATH)
    voices_path = Path(settings.KOKORO_VOICES_PATH)

    logger.info("--- Downloading Kokoro-82M TTS Model ---")
    download_file(KOKORO_ONNX_URL, onnx_path)
    download_file(KOKORO_VOICES_URL, voices_path)

    # Verify loading with kokoro-onnx
    import kokoro_onnx
    logger.info(f"Validating Kokoro model loading from {onnx_path}...")
    kokoro = kokoro_onnx.Kokoro(str(onnx_path), str(voices_path))
    voices = kokoro.get_voices()
    logger.info(f"Kokoro model successfully validated! Available voices: {len(voices)}")
    return onnx_path, voices_path


def main() -> int:
    settings.ensure_directories()
    try:
        download_kokoro_models()
        download_piper_models()
        download_whisper_models()
        logger.info("All speech models (Whisper STT, Kokoro TTS & Piper TTS) downloaded and verified successfully!")
        return 0
    except Exception as e:
        logger.error(f"Failed to download audio models: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
