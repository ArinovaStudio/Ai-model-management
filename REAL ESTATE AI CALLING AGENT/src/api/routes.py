"""FastAPI Telephony and Call Lifecycle REST/WebSocket Router."""

import uuid
from typing import Optional, List
from fastapi import APIRouter, WebSocket, Query, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from src.core.logging import logger
from src.db.session import AsyncSessionLocal
from src.db.models import OperatorNotification, Lead, PropertyListing, CalendarSlot
from src.telephony.session_manager import session_manager
from src.telephony.event_router import handle_telephony_websocket

router = APIRouter()


class CallStartRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Unique session ID. Auto-generated if omitted.")
    caller_phone: Optional[str] = Field(default=None, description="Caller's phone number")
    caller_name: Optional[str] = Field(default=None, description="Caller's name if known")


class CallEndRequest(BaseModel):
    session_id: str = Field(description="Active session ID to terminate")


class NotificationResponse(BaseModel):
    id: int
    session_id: str
    phone_number: Optional[str]
    reason: str
    partial_summary: str
    urgency: str
    status: str
    created_at: str


from fastapi.responses import PlainTextResponse
from src.core.metrics import metrics_registry


@router.get("/health")
async def health_check():
    """Health check endpoint for container probes."""
    return {"status": "ok", "service": "realestate-ai-voice-agent"}


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus exposition metrics scraping endpoint."""
    text = await metrics_registry.generate_metrics_text()
    return PlainTextResponse(content=text, media_type="text/plain; version=0.0.4; charset=utf-8")


@router.post("/api/calls/start")
async def start_call(req: CallStartRequest):
    """Initializes session state and registers call initiation."""
    session_id = req.session_id or f"call_{uuid.uuid4().hex[:12]}"
    session = session_manager.create_session(
        session_id=session_id,
        caller_phone=req.caller_phone,
        caller_name=req.caller_name,
    )
    metrics_registry.record_call_start()
    return {
        "session_id": session_id,
        "status": session["status"],
        "caller_phone": session["caller_phone"],
        "start_time": session["start_time"].isoformat(),
    }


@router.post("/api/calls/end")
async def end_call(req: CallEndRequest):
    """Gracefully terminates a call session and flushes state."""
    session = session_manager.get_session(req.session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {req.session_id} not found",
        )
    res = session_manager.end_session_normal(req.session_id)
    metrics_registry.record_call_end()
    return {"session_id": req.session_id, "status": res["status"]}


@router.get("/api/notifications", response_model=List[NotificationResponse])
async def list_notifications():
    """Retrieves operator notifications for human callback triage."""
    async with AsyncSessionLocal() as sess:
        query = select(OperatorNotification).order_by(OperatorNotification.created_at.desc()).limit(50)
        result = await sess.execute(query)
        notifications = result.scalars().all()
        return [
            NotificationResponse(
                id=n.id,
                session_id=n.session_id,
                phone_number=n.phone_number,
                reason=n.reason,
                partial_summary=n.partial_summary,
                urgency=n.urgency,
                status=n.status,
                created_at=n.created_at.isoformat(),
            )
            for n in notifications
        ]


@router.websocket("/ws/audio/{session_id}")
async def telephony_audio_websocket(
    websocket: WebSocket,
    session_id: str,
    phone: Optional[str] = Query(None),
):
    """Full-duplex binary audio WebSocket bridge."""
    cleaned_phone = phone.strip() if phone else None
    if cleaned_phone and not cleaned_phone.startswith("+") and not cleaned_phone.startswith("0") and len(cleaned_phone) >= 10:
        cleaned_phone = f"+{cleaned_phone.lstrip('+ ')}"
    await handle_telephony_websocket(
        websocket=websocket,
        session_id=session_id,
        caller_phone=cleaned_phone,
    )


@router.get("/api/properties")
async def list_properties():
    """Returns verified property inventory for the UI console."""
    async with AsyncSessionLocal() as sess:
        query = select(PropertyListing).order_by(PropertyListing.id.asc())
        result = await sess.execute(query)
        properties = result.scalars().all()
        return [
            {
                "id": p.id,
                "title": p.title,
                "project_name": p.project_name,
                "niche": p.niche.value if hasattr(p.niche, "value") else str(p.niche),
                "location": p.location,
                "price": float(p.price),
                "dimensions": p.dimensions,
                "legal_approval_status": p.legal_approval_status,
                "paperwork_checklist": p.paperwork_checklist,
            }
            for p in properties
        ]


@router.get("/api/slots")
async def list_slots():
    """Returns upcoming calendar slots for the UI console."""
    async with AsyncSessionLocal() as sess:
        query = select(CalendarSlot).order_by(CalendarSlot.slot_start.asc()).limit(30)
        result = await sess.execute(query)
        slots = result.scalars().all()
        return [
            {
                "id": s.id,
                "property_id": s.property_id,
                "start": s.slot_start.isoformat(),
                "end": s.slot_end.isoformat(),
                "status": s.status.value if hasattr(s.status, "value") else str(s.status),
                "hold_expires_at": s.hold_expires_at.isoformat() if s.hold_expires_at else None,
            }
            for s in slots
        ]


# --- LEAD-BASED CALLING REST ENDPOINTS (Additive) ---

class LeadSubmitRequest(BaseModel):
    phone_number: str = Field(description="Prospect phone number")
    name: Optional[str] = Field(default=None, description="Prospect full name")
    niche: Optional[str] = Field(default=None, description="Property niche interest (e.g. luxury_villa, residential_plot)")
    budget_min: Optional[float] = Field(default=None, description="Minimum budget")
    budget_max: Optional[float] = Field(default=None, description="Maximum budget")
    preferred_location: Optional[str] = Field(default=None, description="Target location/locality")
    inquiry_notes: Optional[str] = Field(default=None, description="Specific questions or notes from website form")


@router.post("/api/leads/submit")
async def submit_website_lead(req: LeadSubmitRequest):
    """Ingests and triages website form leads.

    Automatically identifies and flags poor/junk leads.
    """
    from src.agent.lead_service import lead_service
    res = await lead_service.ingest_lead(
        phone_number=req.phone_number,
        name=req.name,
        niche=req.niche,
        budget_min=req.budget_min,
        budget_max=req.budget_max,
        preferred_location=req.preferred_location,
        inquiry_notes=req.inquiry_notes,
    )
    return res


@router.get("/api/leads")
async def list_leads():
    """Returns all leads categorized by quality status for operator review."""
    async with AsyncSessionLocal() as sess:
        query = select(Lead).order_by(Lead.created_at.desc()).limit(50)
        result = await sess.execute(query)
        leads = result.scalars().all()
        return [
            {
                "id": l.id,
                "name": l.name,
                "phone_number": l.phone_number,
                "niche_preference": l.niche_preference,
                "budget_max": l.budget_max,
                "preferred_location": l.preferred_location,
                "status": l.status,
                "quality_status": getattr(l, "quality_status", "valid"),
                "flag_reason": getattr(l, "flag_reason", None),
                "call_summary": getattr(l, "call_summary", None),
                "created_at": l.created_at.isoformat() if l.created_at else None,
            }
            for l in leads
        ]


@router.get("/api/leads/{lead_id}/summary")
async def get_lead_call_summary(lead_id: int):
    """Retrieves the 75-200 word post-call summary for a specific lead."""
    async with AsyncSessionLocal() as sess:
        query = select(Lead).where(Lead.id == lead_id)
        result = await sess.execute(query)
        lead = result.scalar_one_or_none()
        if not lead:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Lead {lead_id} not found")
        return {
            "lead_id": lead.id,
            "name": lead.name,
            "phone_number": lead.phone_number,
            "quality_status": getattr(lead, "quality_status", "valid"),
            "call_summary": getattr(lead, "call_summary", None),
        }

