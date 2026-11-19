"""Concurrency Stress Test for Conflict-Safe Visit Scheduling.

Validates that when 10 concurrent async workers attempt to claim the exact same
calendar slot at the same millisecond:
1. Exactly 1 worker successfully acquires the 5-minute hold.
2. 9 workers receive clean conflict detections with alternative open slots.
3. Zero race conditions, zero duplicate bookings, and zero deadlocks occur.
4. The winning worker successfully commits the confirmation.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from src.db.session import AsyncSessionLocal
from src.db.models import CalendarSlot, SlotStatus, Lead
from src.db.scheduling_manager import scheduling_manager


@pytest.mark.asyncio
async def test_concurrent_slot_booking_race_condition():
    """Stress test: 10 concurrent prospects race to book the exact same slot."""
    async with AsyncSessionLocal() as session:
        # Step 1: Ensure test lead exists
        lead_query = await session.execute(
            select(Lead).where(Lead.phone_number == "+919876500001")
        )
        test_lead = lead_query.scalar_one_or_none()
        if not test_lead:
            test_lead = Lead(
                phone_number="+919876500001",
                name="Concurrent Test Buyer",
                niche_preference="residential_plot",
            )
            session.add(test_lead)
            await session.commit()
            await session.refresh(test_lead)
        lead_id = test_lead.id

        # Step 2: Find a clean available slot for testing
        result = await session.execute(
            select(CalendarSlot).where(CalendarSlot.status == SlotStatus.AVAILABLE).limit(1)
        )
        target_slot = result.scalar_one_or_none()
        assert target_slot is not None, "At least one available slot must exist in the database."
        slot_id = target_slot.id

    # Step 3: Define worker function simulating a prospect call turn
    async def worker(worker_id: int):
        idempotency_key = f"key_worker_{worker_id}_{slot_id}"
        # Small delay to simulate realistic asynchronous network packet arrival
        await asyncio.sleep(0.001 * (worker_id % 3))
        res = await scheduling_manager.acquire_temporary_hold(
            slot_id=slot_id,
            prospect_id=lead_id,
            idempotency_key=idempotency_key,
            hold_duration_seconds=300,
        )
        return {"worker_id": worker_id, "idempotency_key": idempotency_key, **res}

    # Step 4: Launch 10 workers concurrently at the exact same time
    tasks = [worker(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # Step 5: Verify assertions
    successful_holds = [r for r in results if r["success"] is True]
    conflicted_attempts = [r for r in results if r["success"] is False and r["conflict"] is True]

    # ASSERTION 1: Exactly 1 worker got the hold
    assert len(successful_holds) == 1, (
        f"CRITICAL FAILURE: Expected exactly 1 successful hold, but got {len(successful_holds)}!"
    )

    # ASSERTION 2: Exactly 9 workers received conflict detection
    assert len(conflicted_attempts) == 9, (
        f"CRITICAL FAILURE: Expected exactly 9 conflicts, but got {len(conflicted_attempts)}!"
    )

    winner = successful_holds[0]
    # ASSERTION 3: Conflicted callers received alternative suggestions
    for loser in conflicted_attempts:
        assert len(loser["alternative_slots"]) > 0, (
            f"Worker {loser['worker_id']} had a conflict but received no alternative slots."
        )

    # Step 6: Verify database state matches the winner
    async with AsyncSessionLocal() as session:
        check = await session.execute(select(CalendarSlot).where(CalendarSlot.id == slot_id))
        db_slot = check.scalar_one()
        assert db_slot.status == SlotStatus.HELD
        assert db_slot.hold_idempotency_key == winner["idempotency_key"]

    # Step 7: Winner confirms the booking
    confirm_res = await scheduling_manager.confirm_visit_booking(
        slot_id=slot_id,
        idempotency_key=winner["idempotency_key"],
        lead_id=lead_id,
        notes="Confirmed by concurrent stress test winner.",
    )
    assert confirm_res["success"] is True, f"Winning confirmation failed: {confirm_res.get('error')}"

    # Step 8: Verify final database state is CONFIRMED
    async with AsyncSessionLocal() as session:
        check = await session.execute(select(CalendarSlot).where(CalendarSlot.id == slot_id))
        db_slot = check.scalar_one()
        assert db_slot.status == SlotStatus.CONFIRMED

    # Step 9: Verify a new attempt on the now-confirmed slot immediately fails
    new_attempt = await scheduling_manager.acquire_temporary_hold(
        slot_id=slot_id,
        prospect_id=lead_id,
        idempotency_key="late_comer_key",
    )
    assert new_attempt["success"] is False
    assert new_attempt["conflict"] is True


import uuid

@pytest.mark.asyncio
async def test_expired_hold_reclamation():
    """Validates that a slot with an expired 5-minute hold can be reclaimed by a new caller."""
    unique_abandoned_key = f"old_abandoned_{uuid.uuid4().hex[:8]}"
    unique_reclaim_key = f"reclaim_key_{uuid.uuid4().hex[:8]}"

    async with AsyncSessionLocal() as session:
        # Create test lead
        lead_query = await session.execute(
            select(Lead).where(Lead.phone_number == "+919876500002")
        )
        test_lead = lead_query.scalar_one_or_none()
        if not test_lead:
            test_lead = Lead(
                phone_number="+919876500002",
                name="Reclaim Test Buyer",
            )
            session.add(test_lead)
            await session.commit()
            await session.refresh(test_lead)
        lead_id = test_lead.id

        # Create a test slot with an expired hold in the past
        past_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        expired_slot = CalendarSlot(
            representative_id="agent_rep_01",
            slot_start=datetime.now(timezone.utc) + timedelta(days=2),
            slot_end=datetime.now(timezone.utc) + timedelta(days=2, hours=1),
            status=SlotStatus.HELD,
            hold_expires_at=past_time,
            hold_idempotency_key=unique_abandoned_key,
            booked_by_lead_id=lead_id,
            notes="Expired test slot",
        )
        session.add(expired_slot)
        await session.commit()
        await session.refresh(expired_slot)
        expired_slot_id = expired_slot.id

    # New caller attempts to claim the expired slot
    new_claim = await scheduling_manager.acquire_temporary_hold(
        slot_id=expired_slot_id,
        prospect_id=lead_id,
        idempotency_key=unique_reclaim_key,
        hold_duration_seconds=300,
    )

    assert new_claim["success"] is True
    assert new_claim["conflict"] is False
    assert new_claim["slot"].hold_idempotency_key == unique_reclaim_key
    assert new_claim["slot"].status == SlotStatus.HELD
