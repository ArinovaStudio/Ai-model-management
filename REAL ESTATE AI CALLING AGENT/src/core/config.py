"""Application configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for RealEstateAI."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General Environment
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: str = "INFO"

    # PostgreSQL Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5435
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "realestate_ai"
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5435/realestate_ai",
        description="Async connection string for FastAPI / asyncpg",
    )
    DATABASE_SYNC_URL: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5435/realestate_ai",
        description="Sync connection string for migrations and synchronous tools",
    )

    # CalDAV (Radicale)
    CALDAV_URL: str = "http://localhost:5232"
    CALDAV_USERNAME: str = "agent"
    CALDAV_PASSWORD: str = "ProdCalDAVSecurePass2026!"
    CALDAV_CALENDAR_NAME: str = "SiteVisits"

    # Redis Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6380
    REDIS_URL: str = "redis://localhost:6380/0"

    # LLM Settings (Remote Cloud GPU in dev, Local Ollama in prod)
    LLM_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama / vLLM / OpenAI-compatible API endpoint",
    )
    LLM_MODEL_NAME: str = "llama3:latest"
    LLM_API_KEY: str | None = None
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT_SECONDS: float = 120.0
    LLM_ENABLE_VOICE_FEEDBACK_PASS: bool = True

    # STT Settings (Faster-Whisper)
    WHISPER_MODEL_SIZE: str = "base.en"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_CONFIDENCE_THRESHOLD: float = 0.70
    WHISPER_DOWNLOAD_ROOT: str = "./models/whisper"

    # TTS Settings (Kokoro ONNX, Piper ONNX & Zero-Shot Cloning)
    TTS_ENGINE: Literal["kokoro", "piper", "zero_shot_clone"] = "kokoro"
    KOKORO_MODEL_PATH: str = "./models/kokoro/kokoro-v0_19.onnx"
    KOKORO_VOICES_PATH: str = "./models/kokoro/voices.bin"
    KOKORO_DEFAULT_VOICE: str = "af_sarah"
    KOKORO_SPEED: float = 1.0

    # Zero-Shot Voice Cloning Settings (Simi Indian English reference clip)
    ZERO_SHOT_REF_AUDIO_PATH: str = "./storage/cartesia_2026-11-03T21_59_47Z_Simi_sonic-3.6_en-IN.wav"
    ZERO_SHOT_REF_TRANSCRIPT: str = (
        "main types of real estate, residential, homes where people live, like houses and apartments, "
        "commercial, buildings used for businesses, offices and shops, land, undeveloped plots, vacant "
        "areas or farms, special purpose, public properties like schools, parks and places of worship. "
        "If you want to explore further, tell me, are you looking to buy a home to live in or invest for income? "
        "What is your preferred location or budget range? If you are serious about locking in solid returns "
        "before the end of this quarter, we need to look closely at Plot 2011, that 3500 square foot corner lot "
        "in Greenview Estates, because corner plots in this corridor appreciate nearly 18% faster than standard lots. "
        "And with a flexible 20% down payment plan and clear titles ready for immediate registration, you could "
        "literally have your architect breaking ground by next month while the rest of the market is still trying to catch up."
    )
    ZERO_SHOT_DEVICE: str = "auto"  # 'auto' selects cuda if available, else cpu

    PIPER_VOICE_MODEL: str = "en_US-lessac-medium"
    PIPER_MODEL_PATH: str = "./models/piper/en_US-lessac-medium.onnx"
    PIPER_CONFIG_PATH: str = "./models/piper/en_US-lessac-medium.onnx.json"
    PIPER_DEVICE: str = "cpu"
    PIPER_CACHE_DIR: str = "./storage/audio_cache"

    # Telephony & Audio Streaming Parameters
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    AUDIO_SAMPLE_WIDTH: int = 2  # 16-bit PCM = 2 bytes
    JITTER_BUFFER_MS: int = 20
    MAX_SILENCE_DURATION_MS: int = 1800

    # Storage Paths
    STORAGE_DIR: str = "./storage"
    DOCUMENTS_DIR: str = "./storage/documents"
    MODELS_DIR: str = "./models"

    def ensure_directories(self) -> None:
        """Ensures that local storage, models, and cache directories exist."""
        Path(self.STORAGE_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.PIPER_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.MODELS_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.WHISPER_DOWNLOAD_ROOT).mkdir(parents=True, exist_ok=True)
        Path("./models/piper").mkdir(parents=True, exist_ok=True)
        Path("./models/kokoro").mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()

