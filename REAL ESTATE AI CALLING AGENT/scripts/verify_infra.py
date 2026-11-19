"""Infrastructure verification script.

Validates connectivity and configuration for:
1. PostgreSQL + pgvector extension
2. Radicale CalDAV calendar service
3. Redis caching service
4. LLM endpoint (Ollama / vLLM / Cloud GPU)
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import socket
import urllib.request
import urllib.error
import json
from urllib.parse import urlparse
from src.core.config import settings
from src.core.logging import logger


def check_tcp_port(host: str, port: int, service_name: str) -> bool:
    """Checks if a TCP port is open and accepting connections."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            logger.info(f"[PASS] {service_name} TCP port is open on {host}:{port}")
            return True
        else:
            logger.error(f"[FAIL] {service_name} TCP port {port} on {host} is unreachable (error code: {result})")
            return False
    except Exception as e:
        logger.error(f"[FAIL] {service_name} socket check failed: {e}")
        return False


def check_postgres_pgvector() -> bool:
    """Connects to PostgreSQL and verifies the pgvector extension."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            dbname=settings.POSTGRES_DB,
            connect_timeout=5,
        )
        cur = conn.cursor()
        cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row:
            extname, version = row
            logger.info(f"[PASS] PostgreSQL connected. pgvector extension active (version {version}).")
            return True
        else:
            logger.error("[FAIL] PostgreSQL connected, but 'vector' extension is NOT installed!")
            return False
    except ImportError:
        logger.warning("[WARN] psycopg2 not installed in current environment. Performing socket check only.")
        return check_tcp_port(settings.POSTGRES_HOST, settings.POSTGRES_PORT, "PostgreSQL")
    except Exception as e:
        logger.error(f"[FAIL] PostgreSQL verification failed: {e}")
        return False


def check_caldav() -> bool:
    """Validates the Radicale CalDAV endpoint."""
    caldav_url = settings.CALDAV_URL
    parsed = urlparse(caldav_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5232

    tcp_ok = check_tcp_port(host, port, "Radicale CalDAV")
    if not tcp_ok:
        return False

    try:
        req = urllib.request.Request(f"{caldav_url}/.web/", headers={"User-Agent": "RealEstateAI-Verifier/1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status in (200, 401):
                logger.info(f"[PASS] CalDAV HTTP service is responsive at {caldav_url} (HTTP {response.status}).")
                return True
    except urllib.error.HTTPError as e:
        if e.code in (200, 401, 403):
            logger.info(f"[PASS] CalDAV HTTP service is responsive at {caldav_url} (HTTP {e.code}).")
            return True
        logger.error(f"[FAIL] CalDAV HTTP endpoint returned error code: {e.code}")
        return False
    except Exception as e:
        logger.warning(f"[WARN] CalDAV HTTP check: {e}")
        return tcp_ok


def check_redis() -> bool:
    """Validates the Redis cache endpoint."""
    return check_tcp_port(settings.REDIS_HOST, settings.REDIS_PORT, "Redis")


def check_llm_endpoint() -> bool:
    """Checks the LLM endpoint (Ollama / Cloud GPU)."""
    llm_url = settings.LLM_BASE_URL.rstrip("/")
    parsed = urlparse(llm_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 11434)

    try:
        # Check Ollama tags endpoint
        tags_url = f"{llm_url}/api/tags"
        req = urllib.request.Request(tags_url, headers={"User-Agent": "RealEstateAI-Verifier/1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                models = [m.get("name") for m in data.get("models", [])]
                logger.info(f"[PASS] LLM endpoint responsive at {llm_url}. Available models: {models}")
                return True
    except Exception:
        # Fallback to root or socket check
        tcp_ok = check_tcp_port(host, port, f"LLM Endpoint ({llm_url})")
        if tcp_ok:
            logger.info(f"[PASS] LLM host {host}:{port} is reachable.")
            return True
        else:
            logger.warning(
                f"[INFO] LLM endpoint at {llm_url} is currently offline or unconfigured. "
                f"Set LLM_BASE_URL in .env to your Cloud GPU IP when running inference."
            )
            return True  # Non-blocking for local infrastructure verification


def main() -> int:
    logger.info("=" * 60)
    logger.info("Starting Infrastructure Verification for RealEstateAI...")
    logger.info(f"Target Environment : {settings.ENVIRONMENT}")
    logger.info(f"Database Port      : {settings.POSTGRES_PORT} (DB: {settings.POSTGRES_DB})")
    logger.info(f"CalDAV URL         : {settings.CALDAV_URL}")
    logger.info(f"LLM Endpoint       : {settings.LLM_BASE_URL} ({settings.LLM_MODEL_NAME})")
    logger.info("=" * 60)

    pg_ok = check_postgres_pgvector()
    caldav_ok = check_caldav()
    redis_ok = check_redis()
    llm_ok = check_llm_endpoint()

    logger.info("=" * 60)
    if pg_ok and caldav_ok and redis_ok:
        logger.info(">>> ALL CORE INFRASTRUCTURE SERVICES PASSED VERIFICATION! <<<")
        return 0
    else:
        logger.error(">>> ONE OR MORE CORE INFRASTRUCTURE SERVICES FAILED! <<<")
        return 1


if __name__ == "__main__":
    sys.exit(main())
