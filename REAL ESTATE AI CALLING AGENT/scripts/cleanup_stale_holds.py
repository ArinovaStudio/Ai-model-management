"""CLI Command for Manual or Cron-based Stale Slot Hold Cleanup."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import logger
from src.workers.stale_hold_cleanup import run_stale_hold_cleanup_once


async def main():
    logger.info("Executing manual stale slot hold reclamation sweep...")
    reclaimed = await run_stale_hold_cleanup_once()
    print(f"Reclaimed {reclaimed} expired slot holds.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
