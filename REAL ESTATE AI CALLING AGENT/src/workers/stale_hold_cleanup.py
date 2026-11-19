"""Stale Reservation Hold Cleanup Worker.

Periodically releases expired 5-minute reservation holds where prospects
dropped off or failed to provide verbal confirmation, restoring slots to AVAILABLE.
"""

import asyncio
from src.core.logging import logger
from src.db.scheduling_manager import scheduling_manager


async def run_stale_hold_cleanup_once() -> int:
    """Performs a single reclamation sweep of expired holds, retries CalDAV syncs, and reaps zombie sessions."""
    try:
        count = await scheduling_manager.reclaim_expired_holds()
        await scheduling_manager.retry_pending_caldav_syncs()
        from src.telephony.session_manager import session_manager
        session_manager.reap_stale_sessions(max_idle_seconds=300)
        return count
    except Exception as e:
        logger.error(f"Error executing stale hold cleanup sweep: {e}", exc_info=True)
        return 0


async def start_stale_hold_cleanup_loop(interval_seconds: int = 60) -> None:
    """Continuous background loop for ASGI application lifecycles."""
    logger.info(f"Starting stale hold cleanup background worker (interval: {interval_seconds}s)...")
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            reclaimed = await run_stale_hold_cleanup_once()
            if reclaimed > 0:
                logger.info(f"[StaleHoldCleanup] Successfully released {reclaimed} expired holds.")
        except asyncio.CancelledError:
            logger.info("Stale hold cleanup background worker stopping gracefully.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in stale hold cleanup loop: {e}", exc_info=True)
