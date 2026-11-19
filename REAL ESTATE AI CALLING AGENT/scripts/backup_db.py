"""Cross-Platform PostgreSQL Database Backup Script.

Creates timestamped compressed dumps with automatic retention management.
Can run on both Windows host and Linux servers.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timezone
from src.core.config import settings
from src.core.logging import logger


def run_backup(retention_days: int = 7) -> int:
    backup_dir = Path(settings.STORAGE_DIR) / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    custom_dump = backup_dir / f"realestate_ai_{timestamp}.dump"

    host = settings.POSTGRES_HOST
    port = str(settings.POSTGRES_PORT)
    user = settings.POSTGRES_USER
    db = settings.POSTGRES_DB
    password = settings.POSTGRES_PASSWORD

    env = os.environ.copy()
    env["PGPASSWORD"] = password

    logger.info(f"Starting PostgreSQL backup for '{db}' on {host}:{port}...")

    # Run pg_dump
    cmd = [
        "pg_dump",
        "-h", host,
        "-p", port,
        "-U", user,
        "-d", db,
        "-F", "c",
        "-b",
        "-f", str(custom_dump),
    ]

    t0 = time.perf_counter()
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        elapsed = time.perf_counter() - t0
        size_bytes = custom_dump.stat().st_size
        logger.info(f"Backup created: {custom_dump} ({size_bytes / 1024:.1f} KB in {elapsed:.2f}s)")
    except FileNotFoundError:
        # If pg_dump not on host PATH, try inside the running docker container
        logger.warning("Local pg_dump executable not found on host PATH. Attempting via docker exec...")
        docker_cmd = [
            "docker", "exec", "realestate_postgres",
            "pg_dump", "-U", user, "-d", db, "-F", "c", "-b",
        ]
        try:
            with open(custom_dump, "wb") as f:
                subprocess.run(docker_cmd, stdout=f, check=True)
            size_bytes = custom_dump.stat().st_size
            logger.info(f"Backup created via docker container: {custom_dump} ({size_bytes / 1024:.1f} KB)")
        except Exception as e:
            logger.error(f"Backup via docker failed: {e}")
            return 1
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return 1

    # Prune older than retention_days
    cutoff = time.time() - (retention_days * 86400)
    pruned = 0
    for f in backup_dir.glob("realestate_ai_*.dump"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            pruned += 1

    if pruned > 0:
        logger.info(f"Pruned {pruned} backups older than {retention_days} days.")

    return 0


if __name__ == "__main__":
    sys.exit(run_backup())
