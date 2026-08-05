import os
import json
import logging
from typing import Any, Optional
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, TimeoutError
from fastapi.encoders import jsonable_encoder

logger = logging.getLogger(__name__)

redis_client: Optional[Redis] = None

def get_redis_client() -> Redis:
    global redis_client
    if redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_client = Redis.from_url(redis_url, socket_timeout=2.0)
    return redis_client

async def close_redis() -> None:
    global redis_client
    if redis_client is not None:
        await redis_client.close()
        redis_client = None
        logger.info("Closed Redis connection pool")

async def is_redis_available() -> bool:
    try:
        client = get_redis_client()
        await client.ping()
        return True
    except Exception:
        return False

async def safe_redis_call(func, *args, **kwargs) -> Any:
    try:
        client = get_redis_client()
        return await func(client, *args, **kwargs)
    except (ConnectionError, TimeoutError) as e:
        logger.warning("Redis connection failed, gracefully falling back: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected Redis error: %s", e)
        return None

# Namespace prefix helper
def _get_key(project_id: str, suffix: str) -> str:
    return f"wbs:project:{project_id}:{suffix}"

# Specific Caching Helpers
async def get_cached_wbs_tasks(project_id: str) -> Optional[list]:
    async def _call(client):
        val = await client.get(_get_key(project_id, "tasks"))
        if val:
            return json.loads(val)
        return None
    return await safe_redis_call(_call)

async def set_cached_wbs_tasks(project_id: str, tasks: list) -> None:
    async def _call(client):
        serialized = json.dumps(jsonable_encoder(tasks))
        await client.set(_get_key(project_id, "tasks"), serialized, ex=900)
    await safe_redis_call(_call)

async def get_cached_wbs_estimates(project_id: str) -> Optional[dict]:
    async def _call(client):
        val = await client.get(_get_key(project_id, "estimates"))
        if val:
            return json.loads(val)
        return None
    return await safe_redis_call(_call)

async def set_cached_wbs_estimates(project_id: str, estimates: dict) -> None:
    async def _call(client):
        serialized = json.dumps(jsonable_encoder(estimates))
        await client.set(_get_key(project_id, "estimates"), serialized, ex=900)
    await safe_redis_call(_call)

async def invalidate_project_cache(project_id: str) -> None:
    async def _call(client):
        await client.delete(
            _get_key(project_id, "tasks"),
            _get_key(project_id, "estimates")
        )
        logger.info("Invalidated Redis cache for project %s", project_id)
    await safe_redis_call(_call)
