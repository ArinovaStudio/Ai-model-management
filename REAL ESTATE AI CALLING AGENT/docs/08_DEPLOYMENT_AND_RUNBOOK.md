# 08. Deployment, GPU Scaling & Operational Runbook

## 1. Local Development Environment Runbook

### Prerequisites:
* Python 3.14 (or Python 3.10–3.12).
* PostgreSQL 16 with `pgvector` extension enabled on port `5432` or `5435`.
* Pre-warmed model weights downloaded in `models/`.

### Environment Configuration (`.env`):
```ini
# Application Core
ENVIRONMENT=development
LOG_LEVEL=INFO
APP_HOST=0.0.0.0
APP_PORT=8000

# Database URLs
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5435/realestate_ai
DATABASE_SYNC_URL=postgresql://postgres:postgres@localhost:5435/realestate_ai

# LLM Configuration
# ARCHITECTURAL STANDARD: Local Ollama (http://localhost:11434)
# CURRENT RUNTIME: Cloud Groq (due to host machine CPU/RAM constraints)
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL_NAME=llama-3.3-70b-versatile
LLM_API_KEY=your_groq_api_key_here
LLM_TEMPERATURE=0.1

# Speech Recognition (Faster-Whisper)
WHISPER_MODEL_SIZE=base.en
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_DOWNLOAD_ROOT=./models/whisper

# Speech Synthesis (TTS)
# Options: 'kokoro' (Active Primary), 'piper', 'zero_shot_clone'
TTS_ENGINE=kokoro
KOKORO_MODEL_PATH=./models/kokoro/kokoro-v0_19.onnx
KOKORO_VOICES_PATH=./models/kokoro/voices.bin
KOKORO_DEFAULT_VOICE=af_sarah
KOKORO_SPEED=1.0

# Zero-Shot Voice Cloning Reference Audio
ZERO_SHOT_REF_AUDIO_PATH=./storage/cartesia_2026-11-03T21_59_47Z_Simi_sonic-3.6_en-IN.wav
```

### Starting the Server Locally:
Make sure no broken virtualenvs are active in your shell:
```powershell
# Deactivate any lingering virtualenv
deactivate

# Launch the FastAPI Uvicorn ASGI server
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

## 2. Production Docker & GPU Deployment Runbook

For production environments, the system deploys via Docker Compose with dedicated GPU hardware acceleration.

### Production Stack Overview (`docker-compose.prod.yml`):
```yaml
services:
  # 1. PostgreSQL with pgvector
  postgres:
    image: pgvector/pgvector:pg16
    container_name: realestate-postgres
    restart: always
    environment:
      POSTGRES_DB: realestate_ai
      POSTGRES_USER: realestate_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # 2. Local Ollama LLM Service (when running 100% offline)
  ollama:
    image: ollama/ollama:latest
    container_name: realestate-ollama
    restart: always
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ollama_models:/root/.ollama
    ports:
      - "11434:11434"

  # 3. RealEstateAI Voice Agent API Server
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: realestate-agent
    restart: always
    depends_on:
      - postgres
    environment:
      - DATABASE_URL=postgresql+asyncpg://realestate_user:${POSTGRES_PASSWORD}@postgres:5432/realestate_ai
      - DATABASE_SYNC_URL=postgresql://realestate_user:${POSTGRES_PASSWORD}@postgres:5432/realestate_ai
      - TTS_ENGINE=zero_shot_clone
      - ZERO_SHOT_DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "8000:8000"
```

---

## 3. Production Virtual GPU Scaling Strategy (F5-TTS / Zero-Shot)

When running on an NVIDIA GPU (e.g. NVIDIA T4, A10G, or RTX 4090):

1. **Activate CUDA Hardware**:
   - Set `TTS_ENGINE=zero_shot_clone`.
   - Set `ZERO_SHOT_DEVICE=cuda`.
   - Install `f5-tts` (`pip install f5-tts`).
2. **Speed Profile on GPU**:
   - Flow-matching diffusion synthesizes each spoken clause in **~180ms on CUDA**.
   - The reference audio (`Simi_sonic-3.6_en-IN.wav`) produces an authentic Indian English voice.
3. **Graceful Fallback**:
   - If CUDA memory runs low or an out-of-memory (OOM) condition occurs, the engine automatically catches the exception and falls back to Kokoro-82M ONNX without dropping the call.

---

## 4. Diagnostics, Health Checks & Observability

The platform includes diagnostic CLI scripts under `scripts/`:

### 1. Diagnostic Health Check (`scripts/agent_doctor.py`):
```powershell
python -m scripts.agent_doctor
```
Validates:
- Database connectivity and `pgvector` extension status.
- Faster-Whisper, Kokoro, and embedding model weights.
- Audio sample rate conversion and VAD framing.
- LLM endpoint availability and latency.

### 2. Audio Benchmark (`scripts/benchmark_audio.py`):
```powershell
python -m scripts.benchmark_audio
```
Profiles STT transcription latency, TTS synthesis speed, and resampling overhead on current hardware.

### 3. Verify Lead Pipeline (`scripts/verify_leads.py`):
```powershell
python -m scripts.verify_leads
```
Tests end-to-end lead submission, telephony session instantiation, and appointment persistence.
