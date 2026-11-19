"""Main FastAPI Application Entrypoint."""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import settings
from src.core.logging import logger
from src.api.routes import router
from src.workers.stale_hold_cleanup import start_stale_hold_cleanup_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("Initializing RealEstateAI Voice Agent API Server...")
    settings.ensure_directories()

    # Pre-warming speech and embedding models so first call is instant
    try:
        logger.info("Pre-warming Piper TTS and Faster-Whisper speech models...")
        from src.audio.tts import tts_engine
        from src.audio.stt import stt_engine
        from src.rag.embeddings import get_embedding_model

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, tts_engine._get_voice)
        await loop.run_in_executor(None, stt_engine._get_model)
        await loop.run_in_executor(None, get_embedding_model)
        logger.info("Speech and embedding models pre-warmed successfully.")
    except Exception as e:
        logger.warning(f"Model pre-warming encountered an error: {e}")

    # Start background cleanup worker for expired 5-minute reservation holds
    cleanup_task = asyncio.create_task(start_stale_hold_cleanup_loop(interval_seconds=60))
    yield
    logger.info("Shutting down RealEstateAI Voice Agent API Server...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


def create_app() -> FastAPI:
    """Builds and configures the FastAPI instance."""
    app = FastAPI(
        title="RealEstateAI Local-First Voice Agent",
        description="Local-First VoIP / WebRTC AI Voice Agent for real estate site visits, lead qualification, and knowledge retrieval.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    from pathlib import Path
    from fastapi.staticfiles import StaticFiles

    frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


app = create_app()
