"""Pytest configuration and fixtures for async database testing."""

import sys
import asyncio
import pytest
from src.db.session import async_engine

# On Windows, use WindowsSelectorEventLoopPolicy to prevent Proactor loop closure issues with asyncpg
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture(autouse=True)
async def dispose_engine_after_test():
    """Ensures connections in the pool are cleanly closed before the test event loop closes."""
    yield
    await async_engine.dispose()
