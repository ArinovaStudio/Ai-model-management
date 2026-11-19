"""Cross-platform Production Model Pre-Pull & Cache Initialization.

Pre-downloads and caches:
1. Ollama LLM (Llama 3 8B / 13B) into internal GPU storage.
2. Piper ONNX TTS voice model and configuration.
3. Faster-Whisper base.en int8 model weights.
4. FastEmbed BAAI/bge-small-en-v1.5 embedding weights.
5. Runs Alembic database schema migrations.
"""

import sys
import os
import urllib.request
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from src.core.config import settings
from src.core.logging import logger


def pull_ollama_model():
    """Requests Ollama to pull and warm the configured model."""
    ollama_url = f"{settings.LLM_BASE_URL.rstrip('/')}/api/pull"
    model_name = settings.LLM_MODEL_NAME
    logger.info(f"[1/4] Pulling Ollama model '{model_name}' via {ollama_url}...")
    try:
        with httpx.Client(timeout=600.0) as client:
            resp = client.post(ollama_url, json={"name": model_name, "stream": False})
            if resp.status_code == 200:
                logger.info(f"Ollama model '{model_name}' successfully pulled and ready.")
            else:
                logger.warning(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
    except Exception as exc:
        logger.warning(f"Ollama pull via HTTP failed ({exc}). Ensure Ollama container is running.")


def download_piper_models():
    """Downloads Piper TTS ONNX weights and json config if not locally cached."""
    piper_dir = Path("./models/piper")
    piper_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_file = piper_dir / "en_US-lessac-medium.onnx"
    json_file = piper_dir / "en_US-lessac-medium.onnx.json"
    
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"
    
    logger.info("[2/4] Verifying Piper ONNX TTS models...")
    if not onnx_file.exists():
        logger.info(f"Downloading {onnx_file.name}...")
        try:
            urllib.request.urlretrieve(f"{base_url}/en_US-lessac-medium.onnx", onnx_file)
            logger.info("Piper ONNX weights downloaded.")
        except Exception as exc:
            logger.warning(f"Could not download Piper weights: {exc}")

    if not json_file.exists():
        logger.info(f"Downloading {json_file.name}...")
        try:
            urllib.request.urlretrieve(f"{base_url}/en_US-lessac-medium.onnx.json", json_file)
            logger.info("Piper config downloaded.")
        except Exception as exc:
            logger.warning(f"Could not download Piper config: {exc}")


def cache_speech_and_embeddings():
    """Warms the Faster-Whisper and FastEmbed caches."""
    logger.info("[3/4] Warming Faster-Whisper and FastEmbed local caches...")
    try:
        from faster_whisper import WhisperModel
        logger.info(f"Caching Faster-Whisper {settings.WHISPER_MODEL_SIZE} ({settings.WHISPER_COMPUTE_TYPE})...")
        WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            download_root=settings.WHISPER_DOWNLOAD_ROOT,
        )
        logger.info("Faster-Whisper weights loaded.")
    except Exception as exc:
        logger.warning(f"Faster-Whisper cache warning: {exc}")

    try:
        from fastembed import TextEmbedding
        logger.info("Caching FastEmbed BAAI/bge-small-en-v1.5...")
        TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        logger.info("FastEmbed weights cached.")
    except Exception as exc:
        logger.warning(f"FastEmbed cache warning: {exc}")


def run_migrations():
    """Applies Alembic migrations to the PostgreSQL database."""
    logger.info("[4/4] Applying Alembic database migrations...")
    import subprocess
    cmd = [sys.executable, "-m", "alembic", "upgrade", "head"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("Alembic migrations applied successfully.")
    else:
        logger.warning(f"Alembic migration output: {result.stderr or result.stdout}")


def main():
    logger.info("================================================================")
    logger.info(" Starting Sovereign Production Model Bootstrapper")
    logger.info("================================================================")
    pull_ollama_model()
    download_piper_models()
    cache_speech_and_embeddings()
    run_migrations()
    logger.info("================================================================")
    logger.info(" Model & Schema Bootstrap Complete!")
    logger.info("================================================================")


if __name__ == "__main__":
    main()
