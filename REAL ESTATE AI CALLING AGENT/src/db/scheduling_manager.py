"""Transactional Booking & Concurrency Manager for Site Visits.

Implements strict database row-level locking (SELECT ... FOR UPDATE)
to prevent double-booking or race conditions between concurrent prospects.
"""

from datetime import datetime, timezone, timedelta
from typing import Any
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.session import AsyncSessionLocal
from src.db.models.calendar import CalendarSlot
from src.db.models.lead import Lead
from src.db.models.property import PropertyListing
from src.db.models.enums import SlotStatus
from src.db.caldav_sync import caldav_sync
from src.core.logging import logger


class SchedulingManager:
    """Manages calendar slot verification, 5-minute atomic holds, and booking confirmations."""

    def __init__(self, session_factory=AsyncSessionLocal):
        self.session_factory = session_factory

    async def check_availability(
        self,
        start_time: datetime,
        end_time: datetime,
        session: AsyncSession | None = None,
    ) -> list[CalendarSlot]:
        """Returns all available slots between start_time and end_time.
        
        A slot is considered available if its status is AVAILABLE,
        or if it is HELD but the hold duration has expired.
        """
        now = datetime.now(timezone.utc)

        async def _execute(sess: AsyncSession) -> list[CalendarSlot]:
            query = (
                select(CalendarSlot)
                .where(
                    and_(
                        CalendarSlot.slot_start >= start_time,
                        CalendarSlot.slot_end <= end_time,
                        or_(
                            CalendarSlot.status == SlotStatus.AVAILABLE,
                            and_(
                                CalendarSlot.status == SlotStatus.HELD,
                                CalendarSlot.hold_expires_at < now,
                            ),
                        ),
                    )
                )
                .order_by(CalendarSlot.slot_start.asc())
            )
            result = await sess.execute(query)
            return list(result.scalars().all())

        if session:
            return await _execute(session)
        async with self.session_factory() as sess:
            return await _execute(sess)

    async def acquire_temporary_hold(
        self,
        slot_id: int,
        prospect_id: int | None,
        idempotency_key: str,
        hold_duration_seconds: int = 300,
        session: AsyncSession | None = None,
    ) -> dict[str, Any]:
        """Atomically acquires a temporary hold on a calendar slot using row-level locking.
        
        Acquires row-level lock (FOR UPDATE). If the slot is free or has an expired hold,
        reserves it with the prospect's idempotency key for hold_duration_seconds.
        If conflicted, returns alternative slots without deadlocking.
        """
        now = datetime.now(timezone.utc)
        hold_until = now + timedelta(seconds=hold_duration_seconds)

        async def _execute(sess: AsyncSession) -> dict[str, Any]:
            # Step 1: Acquire row-level lock on the target slot
            query = (
                select(CalendarSlot)
                .where(CalendarSlot.id == slot_id)
                .with_for_update()
            )
            result = await sess.execute(query)
            slot = result.scalar_one_or_none()

            if not slot:
                return {
                    "success": False,
                    "conflict": False,
                    "error": f"Slot id {slot_id} does not exist.",
                    "slot": None,
                    "alternative_slots": [],
                }

            # Step 2: Check idempotency (same prospect re-requesting their own active hold)
            if slot.status == SlotStatus.HELD and slot.hold_idempotency_key == idempotency_key:
                logger.info(f"Slot {slot_id} re-confirmed under same idempotency key '{idempotency_key}'.")
                slot.hold_expires_at = hold_until
                await sess.commit()
                await sess.refresh(slot)
                return {
                    "success": True,
                    "conflict": False,
                    "slot": slot,
                    "hold_expires_at": hold_until,
                    "idempotency_key": idempotency_key,
                }

            slot_start_time = slot.slot_start
            is_available = (
                slot.status == SlotStatus.AVAILABLE
                or (slot.status == SlotStatus.HELD and slot.hold_expires_at and slot.hold_expires_at < now)
            )

            if not is_available:
                # CONFLICT DETECTED: Release lock by ending transaction and fetch alternatives
                logger.warning(
                    f"Conflict detected for slot {slot_id} (status: {slot.status}, hold_expires_at: {slot.hold_expires_at})."
                )
                day_start = slot_start_time.replace(hour=0, minute=0, second=0, microsecond=0)
                day_end = day_start + timedelta(days=1)
                await sess.rollback()

                # Query alternative available slots on the same day or nearby
                alt_query = (
                    select(CalendarSlot)
                    .where(
                        and_(
                            CalendarSlot.id != slot_id,
                            CalendarSlot.slot_start >= day_start,
                            CalendarSlot.slot_end <= day_end,
                            or_(
                                CalendarSlot.status == SlotStatus.AVAILABLE,
                                and_(
                                    CalendarSlot.status == SlotStatus.HELD,
                                    CalendarSlot.hold_expires_at < now,
                                ),
                            ),
                        )
                    )
                    .order_by(CalendarSlot.slot_start.asc())
                    .limit(5)
                )
                alt_result = await sess.execute(alt_query)
                alternatives = list(alt_result.scalars().all())

                return {
                    "success": False,
                    "conflict": True,
                    "error": "The requested time slot is currently reserved or occupied.",
                    "slot": None,
                    "alternative_slots": alternatives,
                }

            # Step 4: Slot is available -> Apply temporary hold
            slot.status = SlotStatus.HELD
            slot.hold_expires_at = hold_until
            slot.hold_idempotency_key = idempotency_key
            slot.booked_by_lead_id = prospect_id
            slot.updated_at = now

            await sess.commit()
            await sess.refresh(slot)
            logger.info(
                f"Successfully acquired 5-minute hold on slot {slot_id} with idempotency key '{idempotency_key}'."
            )

            return {
                "success": True,
                "conflict": False,
                "slot": slot,
                "hold_expires_at": hold_until,
                "idempotency_key": idempotency_key,
            }

        if session:
            return await _execute(session)
        async with self.session_factory() as sess:
            return await _execute(sess)

    async def confirm_visit_booking(
        self,
        slot_id: int,
        idempotency_key: str,
        lead_id: int | None = None,
        notes: str | None = None,
        session: AsyncSession | None = None,
    ) -> dict[str, Any]:
        """Commits a confirmed site visit appointment after verifying active hold and idempotency key."""
        now = datetime.now(timezone.utc)

        async def _execute(sess: AsyncSession) -> dict[str, Any]:
            query = (
                select(CalendarSlot)
                .where(CalendarSlot.id == slot_id)
                .with_for_update()
            )
            result = await sess.execute(query)
            slot = result.scalar_one_or_none()

            if not slot:
                return {"success": False, "error": f"Slot id {slot_id} not found."}

            # Verify that the slot is held by this idempotency key
            if slot.status != SlotStatus.HELD or slot.hold_idempotency_key != idempotency_key:
                return {
                    "success": False,
                    "error": "Slot is not held under the provided idempotency key.",
                }

            # Verify that the hold hasn't expired
            if slot.hold_expires_at and slot.hold_expires_at < now:
                # Mark as available again
                slot.status = SlotStatus.AVAILABLE
                slot.hold_idempotency_key = None
                slot.hold_expires_at = None
                await sess.commit()
                return {
                    "success": False,
                    "error": "Reservation hold has expired. Please re-select a slot.",
                }

            # Commit the booking
            slot.status = SlotStatus.CONFIRMED
            slot.hold_expires_at = None  # Clear temporary hold
            if lead_id:
                slot.booked_by_lead_id = lead_id
            if notes:
                slot.notes = notes
            slot.updated_at = now

            await sess.commit()
            await sess.refresh(slot)
            logger.info(f"Confirmed appointment committed for slot {slot_id} (Lead ID: {lead_id}).")

            # Dispatch CalDAV sync notice
            caldav_status = await self._sync_to_caldav(slot)

            return {
                "success": True,
                "slot": slot,
                "caldav_synced": caldav_status,
            }

        if session:
            return await _execute(session)
        async with self.session_factory() as sess:
            return await _execute(sess)

    async def _sync_to_caldav(self, slot: CalendarSlot) -> bool:
        """Helper to sync confirmed slot with local CalDAV server."""
        try:
            prospect_name = "Interested Prospect"
            phone_number = f"+9198{abs(hash(str(slot.id))) % 100000000:08d}"
            property_title = "Residential Property"
            location = "Bengaluru"

            async with self.session_factory() as sess:
                if slot.booked_by_lead_id:
                    lead_res = await sess.execute(select(Lead).where(Lead.id == slot.booked_by_lead_id))
                    lead = lead_res.scalar_one_or_none()
                    if lead:
                        prospect_name = lead.name or prospect_name
                        phone_number = lead.phone_number or phone_number
                if slot.property_id:
                    prop_res = await sess.execute(select(PropertyListing).where(PropertyListing.id == slot.property_id))
                    prop = prop_res.scalar_one_or_none()
                    if prop:
                        property_title = prop.title or property_title
                        location = prop.location or location

            uid = await caldav_sync.sync_appointment_event_async(
                slot_id=slot.id,
                prospect_name=prospect_name,
                phone_number=phone_number,
                property_title=property_title,
                start_dt=slot.slot_start,
                end_dt=slot.slot_end,
                location=location,
            )
            logger.info(f"Successfully synced CalDAV VEVENT for slot {slot.id} (UID: {uid}).")

            # Persist CalDAV UID on slot and register Appointment ledger record
            try:
                from src.db.models.calendar import Appointment
                async with self.session_factory() as sess:
                    target_slot = await sess.get(CalendarSlot, slot.id)
                    if target_slot:
                        target_slot.caldav_event_uid = uid
                        appointment = Appointment(
                            lead_id=target_slot.booked_by_lead_id,
                            slot_id=target_slot.id,
                            property_id=target_slot.property_id,
                            caldav_event_uid=uid,
                            confirmation_status="confirmed",
                        )
                        sess.add(appointment)
                        await sess.commit()
            except Exception as e:
                logger.debug(f"Appointment record persistence notice: {e}")

            return True
        except Exception as e:
            logger.warning(f"CalDAV sync notice for slot {slot.id}: {e}. Tagging for automated background retry.")
            try:
                async with self.session_factory() as sess:
                    target_slot = await sess.get(CalendarSlot, slot.id)
                    if target_slot:
                        curr_notes = target_slot.notes or ""
                        if "[PENDING_CALDAV_SYNC]" not in curr_notes:
                            target_slot.notes = f"[PENDING_CALDAV_SYNC] {curr_notes}".strip()
                            await sess.commit()
            except Exception:
                pass
            return False

    async def retry_pending_caldav_syncs(self) -> int:
        """Retries syncing confirmed appointments tagged with [PENDING_CALDAV_SYNC]."""
        synced_count = 0
        try:
            async with self.session_factory() as sess:
                query = select(CalendarSlot).where(
                    and_(
                        CalendarSlot.status == SlotStatus.CONFIRMED,
                        CalendarSlot.notes.like("%[PENDING_CALDAV_SYNC]%"),
                    )
                )
                result = await sess.execute(query)
                pending_slots = list(result.scalars().all())

                for slot in pending_slots:
                    # Clear tag before attempting to avoid infinite re-tag loop
                    slot.notes = (slot.notes or "").replace("[PENDING_CALDAV_SYNC]", "").strip()
                    await sess.commit()
                    success = await self._sync_to_caldav(slot)
                    if success:
                        synced_count += 1
                if synced_count > 0:
                    logger.info(f"Successfully retried and synced {synced_count} pending CalDAV appointments.")
        except Exception as exc:
            logger.warning(f"Error during CalDAV sync retry sweep: {exc}")
        return synced_count

    async def reclaim_expired_holds(self) -> int:
        """Reclaims expired 5-minute reservation holds back to AVAILABLE status.
        
        Returns:
            Count of released slots.
        """
        now = datetime.now(timezone.utc)
        async with self.session_factory() as sess:
            query = (
                select(CalendarSlot)
                .where(
                    CalendarSlot.status == SlotStatus.HELD,
                    CalendarSlot.hold_expires_at < now,
                )
                .with_for_update()
            )
            result = await sess.execute(query)
            expired_slots = result.scalars().all()
            reclaimed_count = 0

            for slot in expired_slots:
                slot.status = SlotStatus.AVAILABLE
                slot.hold_expires_at = None
                slot.hold_idempotency_key = None
                slot.booked_by_lead_id = None
                slot.updated_at = now
                reclaimed_count += 1

            if reclaimed_count > 0:
                await sess.commit()
                logger.info(f"Reclaimed {reclaimed_count} expired reservation holds back to AVAILABLE status.")

            return reclaimed_count


scheduling_manager = SchedulingManager()
