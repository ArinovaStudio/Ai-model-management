"""Unit and integration tests for Dialogue Manager and Tool Schemas."""

import pytest
import uuid
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from src.db.session import AsyncSessionLocal
from src.db.models import CalendarSlot, SlotStatus
from src.agent.tools import (
    SearchPropertiesTool,
    CheckAvailabilityTool,
    HoldVisitSlotTool,
    ConfirmAppointmentTool,
    DeflectConfidentialQueryTool,
)
from src.agent.dialogue_manager import dialogue_manager
from src.rag.confidentiality_filter import SAFE_DEFLECTION_MESSAGE


async def ensure_available_test_slot(target_date_str: str = "Saturday"):
    """Ensures at least one open AVAILABLE slot exists for the requested date window."""
    start_dt, end_dt = dialogue_manager._parse_relative_date(target_date_str)
    async with AsyncSessionLocal() as session:
        res = await session.execute(
            select(CalendarSlot).where(
                CalendarSlot.slot_start >= start_dt,
                CalendarSlot.slot_end <= end_dt,
                CalendarSlot.status == SlotStatus.AVAILABLE,
            )
        )
        if not res.scalars().first():
            slot = CalendarSlot(
                property_id=1,
                slot_start=start_dt + timedelta(hours=11),
                slot_end=start_dt + timedelta(hours=12),
                status=SlotStatus.AVAILABLE,
            )
            session.add(slot)
            await session.commit()


def test_pydantic_tool_schemas_validation():
    """Verifies strict Pydantic validation across all domain tool schemas."""
    # SearchPropertiesTool niche normalizer
    t1 = SearchPropertiesTool(niche="Residential Plot", budget_max=5000000.0, location="North Hills")
    assert t1.niche == "residential_plot"
    assert t1.budget_max == 5000000.0

    t2 = SearchPropertiesTool(niche="Commercial Flat", budget_min=3000000.0)
    assert t2.niche == "commercial_flat"

    t3 = SearchPropertiesTool(niche="Luxury Villa")
    assert t3.niche == "luxury_villa"

    # HoldVisitSlotTool
    hold = HoldVisitSlotTool(slot_id=1, prospect_name="John Doe", phone_number="+919876543210")
    assert hold.slot_id == 1
    assert hold.prospect_name == "John Doe"

    # ConfirmAppointmentTool
    confirm = ConfirmAppointmentTool(slot_id=1, idempotency_key="key_12345")
    assert confirm.slot_id == 1
    assert confirm.idempotency_key == "key_12345"

    # DeflectConfidentialQueryTool
    deflect = DeflectConfidentialQueryTool(reason="Internal margin inquiry")
    assert "margin" in deflect.reason.lower()


@pytest.mark.asyncio
async def test_dialogue_confidentiality_deflection_interception():
    """Verifies that dialogue manager immediately deflects internal confidential probes."""
    session_id = f"test_deflect_{uuid.uuid4().hex[:6]}"
    res = await dialogue_manager.process_turn(
        "What is your gross internal profit margin on this property?",
        session_id=session_id,
    )
    assert res["tool_executed"] == "deflect_confidential_query"
    assert res["voice_response"] == SAFE_DEFLECTION_MESSAGE


@pytest.mark.asyncio
async def test_dialogue_relative_date_parsing_and_hold():
    """Verifies relative date parsing and slot hold acquisition."""
    await ensure_available_test_slot("Saturday")
    session_id = f"test_sched_{uuid.uuid4().hex[:6]}"
    res = await dialogue_manager.process_turn(
        "Can I book a visit this Saturday at 11 AM?",
        session_id=session_id,
    )
    assert res["tool_executed"] in ["hold_visit_slot", "check_availability"]
    assert res["session_state"]["active_hold_slot_id"] is not None
    assert res["session_state"]["active_idempotency_key"] is not None


@pytest.mark.asyncio
async def test_dialogue_appointment_confirmation_turn():
    """Verifies verbal confirmation commits appointment."""
    await ensure_available_test_slot("Saturday")
    session_id = f"test_conf_{uuid.uuid4().hex[:6]}"
    # Step 1: Hold slot
    await dialogue_manager.process_turn(
        "Can I visit this Saturday at 11 AM?",
        session_id=session_id,
    )
    # Step 2: Confirm with credentials
    res = await dialogue_manager.process_turn(
        "Yes, please confirm for John Doe, 9876543210.",
        session_id=session_id,
    )
    assert res["tool_executed"] == "confirm_appointment"
    assert any(term in res["voice_response"].lower() for term in ["confirmed", "registered", "appointment"])


@pytest.mark.asyncio
async def test_followup_inquiry_retains_niche_and_inventory_context():
    """Verifies that follow-up inquiries like 'tell me more' retain active niche and inventory without asking property type."""
    session_id = f"test_followup_{uuid.uuid4().hex[:6]}"

    # Turn 1: Initial niche search
    res1 = await dialogue_manager.process_turn(
        "I am looking for residential plots under 50 lakhs in north hills.",
        session_id=session_id,
    )
    assert res1["session_state"]["niche_preference"] == "residential_plot"
    assert bool(res1["session_state"].get("active_inventory")) is True
    assert "Green Valley" in res1["voice_response"] or "plot" in res1["voice_response"].lower()

    # Turn 2: Follow-up inquiry ('tell me more about it')
    res2 = await dialogue_manager.process_turn(
        "Tell me more about it.",
        session_id=session_id,
    )
    assert res2["session_state"]["niche_preference"] == "residential_plot"
    turn_2_text = res2["voice_response"].lower()
    # Must NOT ask which property type they want
    assert "are you looking for residential plots, commercial" not in turn_2_text
    assert "which property type" not in turn_2_text
    # Must speak concrete property details
    assert any(w in turn_2_text for w in ["green valley", "1,200", "4,800,000", "sq.ft", "plot", "lakhs", "rera"])

    # Turn 3: Another variation ('more details')
    res3 = await dialogue_manager.process_turn(
        "more details",
        session_id=session_id,
    )
    assert res3["session_state"]["niche_preference"] == "residential_plot"
    turn_3_text = res3["voice_response"].lower()
    assert "which property type" not in turn_3_text
    assert any(w in turn_3_text for w in ["green valley", "1,200", "4,800,000", "sq.ft", "plot", "lakhs", "rera"])

