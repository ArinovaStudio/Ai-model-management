"""Database session and engine management for Async and Sync connections."""

from collections.abc import AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import AsyncAdaptedQueuePool, NullPool
from sqlalchemy.orm import sessionmaker, Session
from src.core.config import settings

# Async Engine (for FastAPI, asyncio workers, concurrency tests)
# Use NullPool in non-production to avoid cross-event-loop asyncpg connection issues
if settings.ENVIRONMENT == "production":
    async_engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        poolclass=AsyncAdaptedQueuePool,
        pool_size=10,
        max_overflow=10,
        pool_pre_ping=True,
    )
else:
    async_engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        poolclass=NullPool,
    )

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Sync Engine (for Alembic migrations and CLI scripts)
sync_engine = create_engine(
    settings.DATABASE_SYNC_URL,
    echo=False,
    pool_pre_ping=True,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    class_=Session,
    autocommit=False,
    autoflush=False,
)


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for obtaining an asynchronous database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
