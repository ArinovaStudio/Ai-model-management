#!/usr/bin/env bash
# ==============================================================================
# Production Model Pre-Pull & Cache Initialization Script
# Ensures 100% self-hosted, sovereign execution with zero runtime internet calls.
# ==============================================================================

set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
LLM_MODEL="${LLM_MODEL:-llama3:latest}"
PIPER_DIR="./models/piper"

echo "=================================================================="
echo " Starting Sovereign Production Model Bootstrapper"
echo "=================================================================="

# 1. Pre-pull Ollama LLM Model into internal GPU storage
echo "[1/4] Pulling ${LLM_MODEL} into Ollama..."
if command -v ollama &> /dev/null; then
    ollama pull "${LLM_MODEL}"
else
    echo "Using Docker container to pull model..."
    docker exec -it realestate_ollama ollama pull "${LLM_MODEL}" || true
fi

# 2. Pre-download Piper ONNX TTS Voice Model
echo "[2/4] Initializing Piper ONNX TTS Voice models..."
mkdir -p "${PIPER_DIR}"
if [ ! -f "${PIPER_DIR}/en_US-lessac-medium.onnx" ]; then
    echo "Downloading Piper lessac voice weights..."
    curl -L -o "${PIPER_DIR}/en_US-lessac-medium.onnx" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    curl -L -o "${PIPER_DIR}/en_US-lessac-medium.onnx.json" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
fi

# 3. Cache Faster-Whisper & FastEmbed Weights
echo "[3/4] Warming local STT and Embedding caches..."
python -c "
from faster_whisper import WhisperModel
print('Caching Faster-Whisper base.en int8...')
WhisperModel('base.en', device='cpu', compute_type='int8')

from fastembed import TextEmbedding
print('Caching FastEmbed BAAI/bge-small-en-v1.5...')
TextEmbedding(model_name='BAAI/bge-small-en-v1.5')
print('Speech and Embedding models verified and cached locally.')
"

# 4. Run Alembic Migrations
echo "[4/4] Applying PostgreSQL database migrations..."
alembic upgrade head

echo "=================================================================="
echo " Sovereign Model Pre-Pull & Bootstrap Complete!"
echo " All systems ready for air-gapped production operation."
echo "=================================================================="
