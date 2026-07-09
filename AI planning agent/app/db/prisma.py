import logging
import time
from typing import Dict, Tuple, Any

from prisma import Prisma

logger = logging.getLogger(__name__)

db = Prisma(auto_register=True)

# In-memory cache for loaded projects during the execution of a pipeline
_project_cache: Dict[str, Tuple[float, Any]] = {}

def get_cached_project(project_id: str) -> Any:
    if project_id in _project_cache:
        t, project = _project_cache[project_id]
        if time.time() - t < 60:  # 60 seconds TTL
            return project
    return None

def set_cached_project(project_id: str, project: Any) -> None:
    _project_cache[project_id] = (time.time(), project)

def clear_project_cache(project_id: str) -> None:
    if project_id in _project_cache:
        _project_cache.pop(project_id, None)

async def connect_db() -> None:
    if not db.is_connected():
        await db.connect()
        logger.info("Connected to database")


async def disconnect_db() -> None:
    if db.is_connected():
        await db.disconnect()
        logger.info("Disconnected from database")
