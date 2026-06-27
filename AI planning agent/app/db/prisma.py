import logging

from prisma import Prisma

logger = logging.getLogger(__name__)

db = Prisma(auto_register=True)


async def connect_db() -> None:
    if not db.is_connected():
        await db.connect()
        logger.info("Connected to database")


async def disconnect_db() -> None:
    if db.is_connected():
        await db.disconnect()
        logger.info("Disconnected from database")
