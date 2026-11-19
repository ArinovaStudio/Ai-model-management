"""Pydantic Tool Schemas for Agent Reasoning & Execution.

All arguments extracted from speech turns MUST strictly validate against these
schemas before touching any internal database or domain functions.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class SearchPropertiesTool(BaseModel):
    """Searches verified property listings and paperwork checklists in the database."""
    niche: Optional[str] = Field(
        default=None,
        description="Property niche: 'residential_plot', 'commercial_flat', 'luxury_villa', or 'land_parcel'",
    )
    budget_min: Optional[float] = Field(
        default=None,
        description="Minimum target budget in local currency (e.g., 4000000 for 40 Lakhs)",
    )
    budget_max: Optional[float] = Field(
        default=None,
        description="Maximum target budget in local currency (e.g., 5000000 for 50 Lakhs)",
    )
    location: Optional[str] = Field(
        default=None,
        description="Target locality, corridor, or neighborhood (e.g., 'North Hills')",
    )
    property_type: Optional[str] = Field(
        default=None,
        description="Specific property type: plot, flat, villa, parcel",
    )
    query: Optional[str] = Field(
        default="",
        description="Natural language search query for vector similarity retrieval",
    )

    @field_validator("niche", mode="before")
    @classmethod
    def normalize_niche(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        v_clean = v.lower().strip()
        if "plot" in v_clean:
            return "residential_plot"
        elif "commercial" in v_clean or "flat" in v_clean or "apartment" in v_clean or "office" in v_clean:
            return "commercial_flat"
        elif "villa" in v_clean:
            return "luxury_villa"
        elif "parcel" in v_clean or "land" in v_clean or "acre" in v_clean:
            return "land_parcel"
        return v_clean


class CheckAvailabilityTool(BaseModel):
    """Checks the database for open physical property site-visit slots."""
    preferred_date: str = Field(
        description="Requested visit date (e.g., 'Saturday', 'tomorrow', '2026-10-25')",
    )
    preferred_time_slot: Optional[str] = Field(
        default=None,
        description="Preferred time of day (e.g., '11:00 AM', 'morning', '2 PM')",
    )


class HoldVisitSlotTool(BaseModel):
    """Places an atomic 5-minute hold on a selected calendar slot with an idempotency key."""
    slot_id: int = Field(
        description="Primary key ID of the calendar slot to reserve",
    )
    prospect_name: str = Field(
        description="Full name of the prospect booking the site visit",
    )
    phone_number: str = Field(
        description="Contact phone number of the prospect",
    )

    @field_validator("phone_number")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        clean = "".join(ch for ch in v if ch.isdigit() or ch == "+")
        if len(clean) < 7:
            raise ValueError("Phone number must have at least 7 digits.")
        return clean


class ConfirmAppointmentTool(BaseModel):
    """Commits a confirmed site-visit appointment after client verbal confirmation."""
    slot_id: Optional[int] = Field(
        default=None,
        description="ID of the held calendar slot to confirm",
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description="The unique idempotency key returned during hold acquisition",
    )


class DeflectConfidentialQueryTool(BaseModel):
    """Safely deflects client inquiries probing for confidential company internals."""
    reason: str = Field(
        description="Reason for deflection (e.g., 'internal margins', 'unapproved phase', 'broker commissions')",
    )


class RequestHumanCallbackTool(BaseModel):
    """Requests an urgent human operator callback when caller queries exceed autonomous scope."""
    reason: Optional[str] = Field(
        default="Caller requested to speak with a human agent",
        description="Reason for human operator escalation",
    )
    phone_number: Optional[str] = Field(
        default=None,
        description="Callback contact phone number",
    )


class CheckPaperworkChecklistTool(BaseModel):
    """Retrieves mandatory legal documents, RERA approvals, title deed status, and tax clearance requirements."""
    property_id: Optional[int] = Field(
        default=None,
        description="Optional property ID to check specific checklist",
    )
    niche: Optional[str] = Field(
        default=None,
        description="Property niche: 'residential_plot', 'commercial_flat', 'luxury_villa', or 'land_parcel'",
    )
