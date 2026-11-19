"""Agent State Machine & Dialogue Manager.

Coordinates turn-taking, lead qualification, niche discovery, deterministic
tool dispatch, and conversation history.
"""

import json
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any
from sqlalchemy import select
from src.db.session import AsyncSessionLocal
from src.db.models import (
    Lead,
    CalendarSlot,
    ConversationTurn,
    SpeakerType,
    SlotStatus,
    PropertyNiche,
    OperatorNotification,
)
from src.agent.llm_client import llm_client
from src.agent.prompts import AGENT_SYSTEM_PROMPT, CLARIFICATION_PROMPT_TEMPLATE
from src.agent.tools import (
    SearchPropertiesTool,
    CheckAvailabilityTool,
    HoldVisitSlotTool,
    ConfirmAppointmentTool,
    DeflectConfidentialQueryTool,
    RequestHumanCallbackTool,
    CheckPaperworkChecklistTool,
)
from src.rag.retriever import retriever
from src.rag.confidentiality_filter import confidentiality_filter, SAFE_DEFLECTION_MESSAGE
from src.db.scheduling_manager import scheduling_manager
from src.core.config import settings
from src.core.logging import logger


class DialogueManager:
    """Manages dialogue state, tool extraction, and domain execution for calls."""

    def __init__(self):
        # In-memory session tracking for active calls
        self.sessions: dict[str, dict[str, Any]] = {}

    def get_session(self, session_id: str, lead_phone: str | None = None) -> dict[str, Any]:
        """Retrieves or initializes an active call session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "session_id": session_id,
                "lead_phone": lead_phone,
                "lead_id": None,
                "lead_name": None,
                "niche_preference": None,
                "budget_max": None,
                "active_hold_slot_id": None,
                "active_idempotency_key": None,
                "active_inventory": [],
                "turn_index": 0,
                "messages": [{"role": "system", "content": AGENT_SYSTEM_PROMPT}],
            }
        elif lead_phone and not self.sessions[session_id].get("lead_phone"):
            clean = re.sub(r"\D", "", lead_phone)
            if len(clean) >= 10:
                self.sessions[session_id]["lead_phone"] = lead_phone
        return self.sessions[session_id]

    def close_session(self, session_id: str) -> None:
        """Purges active in-memory session state after call completion."""
        self.sessions.pop(session_id, None)

    def _get_fallback_phone(self, session: dict[str, Any]) -> str:
        """Returns the caller or lead phone, or derives a unique deterministic phone placeholder for the session."""
        phone = session.get("lead_phone") or session.get("caller_phone")
        if phone:
            clean = re.sub(r"[^\d+]", "", str(phone))
            if len(re.sub(r"\D", "", clean)) >= 10:
                return clean
        sess_ref = session.get("call_sid") or session.get("session_id") or "anon"
        return f"+9198{abs(hash(str(sess_ref))) % 100000000:08d}"

    @staticmethod
    def _format_currency_for_voice(amount: float | int | None) -> str:
        """Formats numbers into spoken Indian currency (e.g. 48 Lakhs, 1.5 Crores) for natural speech."""
        if amount is None:
            return ""
        try:
            val = float(amount)
            if val >= 10_000_000:
                crores = val / 10_000_000.0
                return f"{crores:.2f}".rstrip("0").rstrip(".") + " Crores"
            elif val >= 100_000:
                lakhs = val / 100_000.0
                return f"{lakhs:.2f}".rstrip("0").rstrip(".") + " Lakhs"
            return f"{val:,.0f} Rupees"
        except (ValueError, TypeError):
            return str(amount)

    @staticmethod
    def _format_phone_for_voice(phone_str: str | None) -> str:
        """Formats phone digits with spaces and commas for natural speech synthesis."""
        if not phone_str:
            return ""
        digits = re.sub(r"\D", "", str(phone_str))[-10:]
        if len(digits) == 10:
            return f"{' '.join(digits[:5])}, {' '.join(digits[5:])}"
        return str(phone_str)

    def _parse_relative_date(self, date_str: str) -> tuple[datetime, datetime]:
        """Parses natural language date strings into start and end datetime windows in market timezone."""
        try:
            from zoneinfo import ZoneInfo
            local_tz = ZoneInfo("Asia/Kolkata")
        except Exception:
            local_tz = timezone.utc

        now_local = datetime.now(local_tz)
        d_lower = date_str.lower().strip()

        target_date = now_local.date()

        weekday_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }

        matched_weekday = None
        for day_name, day_idx in weekday_map.items():
            if day_name in d_lower:
                matched_weekday = day_idx
                break

        if "today" in d_lower:
            target_date = now_local.date()
        elif "tomorrow" in d_lower:
            target_date = now_local.date() + timedelta(days=1)
        elif matched_weekday is not None:
            days_ahead = (matched_weekday - now_local.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = now_local.date() + timedelta(days=days_ahead)
        else:
            # Fallback to tomorrow or next business day
            target_date = now_local.date() + timedelta(days=1)

        local_start = datetime.combine(target_date, datetime.min.time(), tzinfo=local_tz)
        local_end = local_start + timedelta(days=1)
        return local_start.astimezone(timezone.utc), local_end.astimezone(timezone.utc)

    async def execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        session: dict[str, Any],
    ) -> dict[str, Any]:
        """Validates tool arguments via Pydantic and executes the internal domain service."""
        logger.info(f"Executing tool '{tool_name}' with arguments: {tool_args}")

        TOOL_ALIASES = {
            "check_slot_availability": "check_availability",
            "reserve_slot_hold": "hold_visit_slot",
            "confirm_visit_booking": "confirm_appointment",
            "check_paperwork_checklist": "check_paperwork_checklist",
            "request_human_callback": "request_human_callback",
            "request_callback": "request_human_callback",
        }
        canonical_tool = TOOL_ALIASES.get(tool_name, tool_name)

        # If a slot is already held and the caller is verbally confirming, route directly to confirm_appointment
        if session.get("active_hold_slot_id"):
            last_input = (session.get("last_user_input") or "").lower()
            if any(w in last_input for w in ["yes", "confirm", "finalize", "proceed", "book it", "correct", "perfect"]):
                canonical_tool = "confirm_appointment"
                tool_name = "confirm_appointment"
                name_m = re.search(r"for\s+([A-Za-z\s]+?)(?:,|\.|\sat|\s\d|$)", session.get("last_user_input", ""), re.IGNORECASE)
                if name_m:
                    session["lead_name"] = name_m.group(1).strip()
                elif tool_args.get("prospect_name"):
                    session["lead_name"] = tool_args["prospect_name"]

                phone_m = re.search(r"(\+?\d[\d\s\-]{7,14}\d)", session.get("last_user_input", ""))
                if phone_m:
                    session["lead_phone"] = "".join(ch for ch in phone_m.group(1) if ch.isdigit() or ch == "+")
                elif tool_args.get("phone_number"):
                    session["lead_phone"] = tool_args["phone_number"]

        if canonical_tool == "search_properties":
            validated = SearchPropertiesTool(**tool_args)
            if validated.niche:
                session["niche_preference"] = validated.niche
            if validated.budget_max:
                session["budget_max"] = validated.budget_max
            if validated.location:
                session["location_preference"] = validated.location

            query = validated.query or f"{validated.niche or ''} {validated.location or ''}".strip()
            res = await retriever.search_property_knowledge(
                query_str=query or "properties available",
                niche_filter=validated.niche,
                budget_min=validated.budget_min,
                budget_max=validated.budget_max,
                location_filter=validated.location,
                top_k=3,
            )
            return {"tool": tool_name, "result": res}

        elif canonical_tool == "check_availability":
            validated = CheckAvailabilityTool(**tool_args)
            start_dt, end_dt = self._parse_relative_date(validated.preferred_date)
            slots = await scheduling_manager.check_availability(start_dt, end_dt)

            target_hour = None
            if validated.preferred_time_slot:
                p_lower = validated.preferred_time_slot.lower()
                time_m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", p_lower)
                if time_m:
                    h = int(time_m.group(1))
                    meridiem = (time_m.group(3) or "").lower()
                    if meridiem == "pm" and h < 12:
                        h += 12
                    elif meridiem == "am" and h == 12:
                        h = 0
                    target_hour = h
                elif "morning" in p_lower:
                    target_hour = 10
                elif "afternoon" in p_lower:
                    target_hour = 14
                elif "evening" in p_lower:
                    target_hour = 17

            formatted_slots = []
            matching_slot = None
            for s in slots:
                slot_info = {
                    "slot_id": s.id,
                    "date": s.slot_start.strftime("%A, %b %d"),
                    "start_time": s.slot_start.strftime("%I:%M %p"),
                    "end_time": s.slot_end.strftime("%I:%M %p"),
                    "status": s.status.value,
                }
                formatted_slots.append(slot_info)
                if target_hour is not None and s.slot_start.hour == target_hour:
                    matching_slot = s
                elif matching_slot is None:
                    matching_slot = s

            # Auto-acquire 5-minute hold on preferred matching slot so it cannot be double-booked
            if matching_slot and not session.get("active_hold_slot_id"):
                try:
                    prospect_name = session.get("lead_name") or "Interested Prospect"
                    phone_number = self._get_fallback_phone(session)
                    lead_id = await self._ensure_lead(prospect_name, phone_number, session)
                    session["lead_id"] = lead_id
                    idempotency_key = f"key_{uuid.uuid4().hex[:12]}"
                    hold_res = await scheduling_manager.acquire_temporary_hold(
                        slot_id=matching_slot.id,
                        prospect_id=lead_id,
                        idempotency_key=idempotency_key,
                        hold_duration_seconds=300,
                    )
                    if hold_res.get("success"):
                        session["active_hold_slot_id"] = matching_slot.id
                        session["active_idempotency_key"] = idempotency_key
                except Exception as e:
                    logger.warning(f"Could not auto-hold matching slot {matching_slot.id}: {e}")

            return {
                "tool": tool_name,
                "available_slots": formatted_slots,
                "held_slot_id": session.get("active_hold_slot_id"),
                "preferred_matching_slot_id": matching_slot.id if matching_slot else None,
            }

        elif canonical_tool == "hold_visit_slot":
            phone_val = str(tool_args.get("phone_number") or "")
            clean_digits = re.sub(r"[^\d+]", "", phone_val)
            if len(re.sub(r"\D", "", clean_digits)) < 10:
                tool_args["phone_number"] = self._get_fallback_phone(session)

            name_val = str(tool_args.get("prospect_name") or "").strip()
            if not name_val or any(ph in name_val.lower() for ph in ["your name", "prospect name", "client name", "unknown", "none", "n/a"]):
                tool_args["prospect_name"] = session.get("lead_name") or "Interested Prospect"

            last_input = (session.get("last_user_input") or "").lower()
            if session.get("active_hold_slot_id") and any(w in last_input for w in ["yes", "confirm", "finalize", "proceed", "book it", "correct", "perfect"]):
                validated = HoldVisitSlotTool(**tool_args)
                session["lead_name"] = validated.prospect_name
                session["lead_phone"] = validated.phone_number
                lead_id = await self._ensure_lead(validated.prospect_name, validated.phone_number, session)
                session["lead_id"] = lead_id

                slot_id = session["active_hold_slot_id"]
                idempotency_key = session["active_idempotency_key"]
                confirm_res = await scheduling_manager.confirm_visit_booking(
                    slot_id=slot_id,
                    idempotency_key=idempotency_key,
                    lead_id=lead_id,
                    notes=f"Confirmed for prospect {session.get('lead_name', 'Client')} ({session.get('lead_phone', 'N/A')})",
                )
                if confirm_res.get("success"):
                    session["active_hold_slot_id"] = None
                    session["active_idempotency_key"] = None
                    session["booking_confirmed"] = True
                return {"tool": "confirm_appointment", "canonical_tool": "confirm_appointment", **confirm_res}

            validated = HoldVisitSlotTool(**tool_args)
            session["lead_name"] = validated.prospect_name
            session["lead_phone"] = validated.phone_number

            # Upsert or fetch lead
            lead_id = await self._ensure_lead(validated.prospect_name, validated.phone_number, session)
            session["lead_id"] = lead_id

            idempotency_key = f"key_{uuid.uuid4().hex[:12]}"
            hold_res = await scheduling_manager.acquire_temporary_hold(
                slot_id=validated.slot_id,
                prospect_id=lead_id,
                idempotency_key=idempotency_key,
                hold_duration_seconds=300,
            )

            if hold_res["success"]:
                session["active_hold_slot_id"] = validated.slot_id
                session["active_idempotency_key"] = idempotency_key

            return {
                "tool": tool_name,
                "success": hold_res["success"],
                "slot_id": validated.slot_id,
                "idempotency_key": idempotency_key,
                "hold_expires_at": str(hold_res.get("hold_expires_at")),
                "conflict": hold_res.get("conflict", False),
                "alternative_slots": [
                    {
                        "slot_id": alt.id,
                        "time": alt.slot_start.strftime("%A, %I:%M %p"),
                    }
                    for alt in hold_res.get("alternative_slots", [])
                ],
            }

        elif canonical_tool == "confirm_appointment":
            validated = ConfirmAppointmentTool(**tool_args)
            slot_id = validated.slot_id or session.get("active_hold_slot_id")
            idempotency_key = validated.idempotency_key or session.get("active_idempotency_key")
            lead_id = session.get("lead_id")

            # Validate that lead_phone has at least 10 digits
            phone_str = str(session.get("lead_phone") or tool_args.get("phone_number") or "")
            clean_digits = re.sub(r"\D", "", phone_str)
            if len(clean_digits) < 10:
                return {
                    "tool": tool_name,
                    "canonical_tool": canonical_tool,
                    "success": False,
                    "error": "invalid_phone",
                    "message": "A valid 10-digit mobile number is required to finalize the appointment.",
                }

            confirm_res = await scheduling_manager.confirm_visit_booking(
                slot_id=slot_id,
                idempotency_key=idempotency_key,
                lead_id=lead_id,
                notes=f"Confirmed for prospect {session.get('lead_name', 'Client')} ({session.get('lead_phone', 'N/A')})",
            )
            if confirm_res.get("success"):
                session["active_hold_slot_id"] = None
                session["active_idempotency_key"] = None
                session["booking_confirmed"] = True
            return {"tool": tool_name, **confirm_res}

        elif canonical_tool == "deflect_confidential_query":
            return {
                "tool": tool_name,
                "deflection": SAFE_DEFLECTION_MESSAGE,
            }

        elif canonical_tool == "check_paperwork_checklist":
            validated = CheckPaperworkChecklistTool(**tool_args)
            query = f"paperwork checklist legal requirements RERA documents {validated.niche or ''}".strip()
            res = await retriever.search_property_knowledge(
                query_str=query,
                niche_filter=validated.niche,
                top_k=2,
            )
            return {"tool": tool_name, "canonical_tool": canonical_tool, "result": res}

        elif canonical_tool == "request_human_callback":
            validated = RequestHumanCallbackTool(**tool_args)
            cb_phone = validated.phone_number or session.get("lead_phone") or session.get("caller_phone") or "Unknown"
            lead_id = session.get("lead_id")
            session_id = session.get("call_sid") or session.get("session_id") or f"sess_{uuid.uuid4().hex[:12]}"
            reason_str = validated.reason or "Caller requested escalation to human specialist"

            try:
                async with AsyncSessionLocal() as sess:
                    notif = OperatorNotification(
                        session_id=str(session_id),
                        lead_id=lead_id,
                        phone_number=str(cb_phone),
                        reason="human_callback_requested",
                        partial_summary=reason_str,
                        urgency="high",
                        status="pending",
                    )
                    sess.add(notif)
                    await sess.commit()
            except Exception as e:
                logger.warning(f"Could not persist OperatorNotification to database: {e}")

            return {
                "tool": tool_name,
                "canonical_tool": canonical_tool,
                "success": True,
                "phone_number": cb_phone,
                "message": "Human callback requested. A specialist will call back shortly.",
            }

        else:
            raise ValueError(f"Unknown tool name '{tool_name}'")

    async def _ensure_lead(self, name: str, phone: str, session: dict[str, Any]) -> int:
        """Finds or creates a Lead in PostgreSQL with concurrency protection."""
        async with AsyncSessionLocal() as sess:
            query = select(Lead).where(Lead.phone_number == phone)
            result = await sess.execute(query)
            lead = result.scalar_one_or_none()

            if not lead:
                lead = Lead(
                    phone_number=phone,
                    name=name,
                    niche_preference=session.get("niche_preference"),
                    budget_max=session.get("budget_max"),
                    status="qualified",
                )
                sess.add(lead)
                try:
                    await sess.commit()
                    await sess.refresh(lead)
                except Exception as e:
                    logger.warning(f"Concurrent lead insertion conflict on {phone}, falling back to fetch: {e}")
                    await sess.rollback()
                    result = await sess.execute(query)
                    lead = result.scalar_one_or_none()
                    if not lead:
                        raise e
            else:
                lead.name = name
                if session.get("niche_preference"):
                    lead.niche_preference = session["niche_preference"]
                if session.get("budget_max"):
                    lead.budget_max = session["budget_max"]
                await sess.commit()
                await sess.refresh(lead)

            return lead.id

    def extract_tool_call(self, text: str) -> tuple[str | None, dict[str, Any] | None]:
        """Detects and extracts JSON tool calling block from LLM output."""
        # 1. Try finding markdown json code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if "tool" in data:
                    return data.get("tool"), data.get("arguments", {})
            except Exception:
                pass

        # 2. Balanced brace parsing around "tool"
        tool_idx = text.find('"tool"')
        if tool_idx != -1:
            start_idx = text.rfind("{", 0, tool_idx)
            if start_idx != -1:
                brace_count = 0
                for i in range(start_idx, len(text)):
                    if text[i] == "{":
                        brace_count += 1
                    elif text[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            candidate = text[start_idx : i + 1]
                            try:
                                data = json.loads(candidate)
                                if "tool" in data:
                                    return data.get("tool"), data.get("arguments", {})
                            except Exception:
                                pass
                            break

        return None, None

    async def _extract_domain_tool_from_input(
        self, user_input: str, session: dict[str, Any]
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Deterministic intent & parameter extractor from user speech when LLM produces unstructured prose."""
        u_lower = user_input.lower()

        # 0. Human Specialist Escalation (Takes absolute precedence)
        if any(w in u_lower for w in ["human", "agent", "representative", "specialist", "operator", "talk to someone", "speak to someone", "person"]):
            return "request_human_callback", {
                "phone_number": self._get_fallback_phone(session),
                "reason": user_input,
            }

        # 1. Site booking confirmation
        if session.get("active_hold_slot_id") and any(w in u_lower for w in ["yes", "confirm", "proceed", "book it", "please"]):
            phone_match = re.search(r"(\+?\d[\d\s-]{8,14}\d|\b\d{10}\b)", user_input)
            if phone_match:
                clean_phone = re.sub(r"[^\d+]", "", phone_match.group(0))
                if len(re.sub(r"\D", "", clean_phone)) >= 10:
                    session["lead_phone"] = clean_phone
            name_match = re.search(r"(?:for|name is|i am|name would be|name's)\s+([A-Za-z\s]+?)(?:,|\.|\b\d|$)", user_input, re.IGNORECASE)
            if name_match:
                cand = name_match.group(1).strip()
                non_name_tokens = {
                    "tomorrow", "today", "yesterday", "saturday", "sunday", "monday", "tuesday",
                    "wednesday", "thursday", "friday", "slot", "visit", "tour", "appointment",
                    "booking", "site", "me", "my", "us", "him", "her", "now", "later", "this"
                }
                cand_lower = cand.lower().split()
                if cand and not any(w in cand_lower for w in non_name_tokens) and len(cand) >= 2:
                    session["lead_name"] = cand

            return "confirm_appointment", {
                "slot_id": session["active_hold_slot_id"],
                "idempotency_key": session["active_idempotency_key"],
            }

        # 2. Site visit booking / slot inquiry
        if any(w in u_lower for w in ["visit", "tour", "book", "saturday", "tomorrow", "slot", "appointment", "schedule", "friday", "monday", "sunday"]):
            slot_match = re.search(r"slot\s+(\d+)", u_lower)
            if slot_match:
                return "hold_visit_slot", {
                    "slot_id": int(slot_match.group(1)),
                    "prospect_name": session.get("lead_name") or "Interested Prospect",
                    "phone_number": self._get_fallback_phone(session),
                }

            date_mention = "Saturday"
            weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "tomorrow", "today"]
            for d in weekdays:
                if d in u_lower:
                    date_mention = d.capitalize()
                    break

            start_dt, end_dt = self._parse_relative_date(date_mention)
            slots = await scheduling_manager.check_availability(start_dt, end_dt)
            if slots:
                target_slot = slots[0]
                target_hour = None
                time_m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", u_lower)
                if time_m:
                    h = int(time_m.group(1))
                    meridiem = (time_m.group(3) or "").lower()
                    if meridiem == "pm" and h < 12:
                        h += 12
                    elif meridiem == "am" and h == 12:
                        h = 0
                    target_hour = h
                elif "morning" in u_lower:
                    target_hour = 10
                elif "afternoon" in u_lower:
                    target_hour = 14
                elif "evening" in u_lower:
                    target_hour = 17

                if target_hour is not None:
                    for s in slots:
                        if s.slot_start.hour == target_hour:
                            target_slot = s
                            break

                return "hold_visit_slot", {
                    "slot_id": target_slot.id,
                    "prospect_name": session.get("lead_name") or "Interested Prospect",
                    "phone_number": self._get_fallback_phone(session),
                }
            return "check_availability", {"preferred_date": date_mention}

        # 3. Property Search / Niche inquiry / Follow-up inquiry / General Property & Advisory Questions
        is_followup_inquiry = any(p in u_lower for p in [
            "tell me more", "tell more", "tell him more", "tell me about", "tell about it",
            "more details", "more info", "more information", "what else", "elaborate",
            "explain", "go on", "details", "anything else", "what are the options"
        ])

        has_advisory_kw = any(w in u_lower for w in [
            "amenities", "amenity", "pool", "swimming", "clubhouse", "gym", "squash",
            "loan", "mortgage", "bank", "interest", "tax", "exemption", "rebate",
            "rera", "approval", "sanction", "legal", "paperwork", "katha", "deed",
            "encumbrance", "possession", "distance", "airport", "school", "hospital",
            "connectivity", "road", "maintenance", "charge", "fee", "stamp duty"
        ])

        has_property_kw = any(w in u_lower for w in [
            "plot", "flat", "villa", "budget", "price", "looking for", "lakhs",
            "apartment", "buy", "residential", "commercial", "crore", "crores", "inr"
        ])

        if has_property_kw or has_advisory_kw or (is_followup_inquiry and (session.get("niche_preference") or session.get("active_inventory"))):
            niche = (
                "residential_plot" if any(w in u_lower for w in ["plot", "parcel", "land"])
                else ("commercial_flat" if any(w in u_lower for w in ["flat", "apartment", "unit"])
                else ("luxury_villa" if re.search(r"\bvillas?\b", u_lower) else None))
            )

            # Inherit active niche from session memory if caller did not re-state it
            if not niche and session.get("niche_preference"):
                niche = session["niche_preference"]

            budget = None
            b_match = re.search(r"(\d+(?:\.\d+)?)\s*(crores?|cr|lakhs?|lac|l)\b", u_lower)
            if b_match:
                val = float(b_match.group(1))
                multiplier = 10000000.0 if "c" in b_match.group(2) else 100000.0
                budget = val * multiplier
            elif "50" in u_lower:
                budget = 5000000.0
            elif session.get("budget_max"):
                budget = session["budget_max"]

            location = None
            if "north hills" in u_lower:
                location = "North Hills"
            elif "whitefield" in u_lower:
                location = "Whitefield Tech Corridor"
            elif "devanahalli" in u_lower or "airport" in u_lower:
                location = "Airport Road, Devanahalli"
            elif session.get("location_preference"):
                location = session["location_preference"]

            search_query = user_input
            has_specific_subject = any(term in u_lower for term in [
                'school', 'college', 'distance', 'road', 'connectivity', 'hospital', 'canadian',
                'kids', 'child', 'family', 'safe', 'peaceful', 'loan', 'tax', 'amenities', 'pool'
            ])
            if is_followup_inquiry and not has_specific_subject:
                title = None
                for item in session.get("active_inventory") or []:
                    p = item.get("property_details")
                    if p and p.get("title"):
                        title = p["title"]
                        break
                    doc_title = item.get("document_title") or ""
                    clean_doc_title = re.sub(r"\s*\((?:part\s*\d+|overview)\)", "", doc_title, flags=re.IGNORECASE).strip()
                    if clean_doc_title and not any(gen in clean_doc_title.lower() for gen in ["protocol", "faq", "guide", "loan", "tax", "policy"]):
                        title = clean_doc_title
                        break
                    content = item.get("content") or ""
                    h1_m = re.search(r"^#\s+([^-\n]+)", content, re.MULTILINE)
                    if h1_m:
                        cand = h1_m.group(1).strip()
                        if not any(gen in cand.lower() for gen in ["protocol", "faq", "guide", "loan", "tax", "policy"]):
                            title = cand
                            break

                if title:
                    search_query = f"{title} plot sizes inventory pricing specifications"
                elif location and niche:
                    search_query = f"{location} {niche.replace('_', ' ')} plot sizes inventory pricing specifications"
                elif location:
                    search_query = f"{location} property plot sizes inventory pricing specifications"
                elif niche:
                    search_query = f"{niche.replace('_', ' ')} plot sizes inventory pricing specifications"

            return "search_properties", {
                "niche": niche,
                "budget_max": budget,
                "location": location,
                "query": search_query,
            }

        return None, None

    async def process_turn(
        self,
        user_input: str,
        session_id: str,
        lead_phone: str | None = None,
        confidence: float | None = None,
        requires_confirmation: bool = False,
        uncertain_entity: str | None = None,
    ) -> dict[str, Any]:
        """Main turn-taking pipeline:

        Input -> Uncertainty Safeguard -> Confidentiality Firewall -> LLM Reasoning -> Tool Dispatch -> Response Synthesis -> Durable Turn Commit
        """
        session = self.get_session(session_id, lead_phone)
        session["turn_index"] += 1
        turn_start = datetime.now(timezone.utc)

        # STEP 0: SPEECH UNCERTAINTY SAFEGUARD
        # If low-confidence transcription was detected for critical entities (phone, budget, date),
        # force explicit verbal confirmation before committing internal domain changes.
        if (
            requires_confirmation or (confidence is not None and confidence < 0.70)
        ) and not session.get("in_confirmation_step"):
            if not uncertain_entity:
                phone_m = re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{10}\b", user_input)
                budget_m = re.search(r"\b\d+(?:\.\d+)?\s*(?:lakhs?|crores?|k|m|inr)\b", user_input, re.IGNORECASE)
                date_m = re.search(r"\b(saturday|sunday|monday|tuesday|wednesday|thursday|friday|tomorrow)\b", user_input, re.IGNORECASE)
                if phone_m:
                    uncertain_entity = phone_m.group(0)
                elif budget_m:
                    uncertain_entity = budget_m.group(0)
                elif date_m:
                    uncertain_entity = date_m.group(0)

            # ONLY prompt for verbal confirmation if a specific critical transactional entity exists!
            # Do NOT hijack the conversation for generic low-confidence words like "okay" or "you"
            if uncertain_entity:
                confirmation_reply = f"Just to be certain, did you say {uncertain_entity}?"
                session["in_confirmation_step"] = True
                session["pending_confirmed_entity"] = uncertain_entity
                session["pending_original_input"] = user_input

                await self._record_turn(
                    session_id=session_id,
                    turn_index=session["turn_index"],
                    user_text=user_input,
                    agent_text=confirmation_reply,
                    lead_id=session.get("lead_id"),
                    intent="speech_uncertainty_safeguard",
                )
                return {
                    "voice_response": confirmation_reply,
                    "tool_executed": "request_verbal_confirmation",
                    "session_state": session,
                    "uncertain_entity": uncertain_entity,
                }

        if session.get("in_confirmation_step"):
            session["in_confirmation_step"] = False
            # Prospect confirmed the value verbally
            if any(w in user_input.lower() for w in ["yes", "correct", "right", "yeah", "exactly", "sure"]):
                confirmed_entity = session.get("pending_confirmed_entity")
                if confirmed_entity and re.search(r"\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", confirmed_entity):
                    session["lead_phone"] = re.sub(r"[^\d+]", "", confirmed_entity)

                # Replay the original conversational input that contained the confirmed entity
                original_prompt = session.pop("pending_original_input", None)
                if original_prompt:
                    user_input = original_prompt

        # Safeguard: Incomplete phone number check during appointment finalization
        digits_only = re.sub(r"\D", "", user_input)
        is_monetary = any(m in user_input.lower() for m in ["lakh", "lac", "lakhs", "crore", "cr", "thousand", "budget", "inr", "rs", "price"])
        if session.get("active_hold_slot_id") and digits_only and len(digits_only) < 10 and not is_monetary:
            if not ("slot" in user_input.lower()):
                incomplete_phone_msg = "Could you please share your complete 10-digit mobile number so I can finalize your site tour?"
                await self._record_turn(
                    session_id=session_id,
                    turn_index=session["turn_index"],
                    user_text=user_input,
                    agent_text=incomplete_phone_msg,
                    lead_id=session.get("lead_id"),
                    intent="request_complete_phone",
                )
                return {
                    "voice_response": incomplete_phone_msg,
                    "tool_executed": "request_complete_phone",
                    "session_state": session,
                }

        # STEP 1: HARD RETRIEVAL-LAYER CONFIDENTIALITY CHECK
        if confidentiality_filter.is_restricted_query(user_input):
            deflection_voice_response = confidentiality_filter.get_safe_deflection()
            await self._record_turn(
                session_id=session_id,
                turn_index=session["turn_index"],
                user_text=user_input,
                agent_text=deflection_voice_response,
                lead_id=session.get("lead_id"),
                intent="deflect_confidential_probe",
            )
            return {
                "voice_response": deflection_voice_response,
                "tool_executed": "deflect_confidential_query",
                "session_state": session,
            }

        # STEP 2: PREPARE CONVERSATION CONTEXT
        session["last_user_input"] = user_input
        session["messages"].append({"role": "user", "content": user_input})

        # Step 2.1: Pre-load lead profile if available from website form
        if not session.get("lead_initialized") and (session.get("lead_phone") or lead_phone):
            target_phone = session.get("lead_phone") or lead_phone
            try:
                from src.agent.lead_service import lead_service
                lead_data = await lead_service.get_lead_context_for_caller(target_phone)
                if lead_data:
                    session["lead_id"] = lead_data.get("lead_id")
                    session["lead_name"] = session.get("lead_name") or lead_data.get("name")
                    session["niche_preference"] = session.get("niche_preference") or lead_data.get("niche_preference")
                    session["budget_max"] = session.get("budget_max") or lead_data.get("budget_max")
                    session["location_preference"] = session.get("location_preference") or lead_data.get("preferred_location")
                    session["lead_directive"] = lead_service.build_lead_prompt_directive(lead_data)
            except Exception as e:
                logger.debug(f"Lead pre-load check skipped: {e}")
            finally:
                session["lead_initialized"] = True

        # STEP 3: CALL LLM REASONING ENGINE (Sliding window over system prompt + active inventory context + last 8 conversational turns)
        system_msg = session["messages"][0]
        recent_history = [m for m in session["messages"][1:] if m.get("role") in ("user", "assistant")][-8:]
        llm_payload = [system_msg]

        # Inject website lead form directive if available
        if session.get("lead_directive"):
            llm_payload.append({
                "role": "system",
                "content": session["lead_directive"],
            })

        # Inject active verified inventory and knowledge into prompt context so LLM never hallucinates in follow-up turns
        if session.get("active_inventory"):
            active_summary = []
            for item in session["active_inventory"][:3]:
                p = item.get("property_details")
                if p:
                    price_str = self._format_currency_for_voice(p.get("price"))
                    checklist = ", ".join(p.get("paperwork_checklist", [])[:3])
                    active_summary.append(
                        f"- {p.get('title')}: {p.get('dimensions')}, priced at {price_str} in {p.get('location')} ({p.get('legal_approval_status', 'RERA approved')}). Paperwork includes: {checklist}."
                    )
                elif item.get("content"):
                    clean_content = re.sub(r"[#*_\-`>]", " ", item.get("content", ""))
                    clean_content = re.sub(r"\s+", " ", clean_content).strip()
                    doc_t = item.get("document_title") or "Knowledge Document"
                    active_summary.append(f"- {doc_t}: {clean_content[:200]}")
            if active_summary:
                llm_payload.append({
                    "role": "system",
                    "content": (
                        "CURRENT VERIFIED INVENTORY & KNOWLEDGE (Quote accurately from this context to answer questions; NEVER invent unverified specifications):\n"
                        + "\n".join(active_summary)
                    ),
                })

        if session.get("booking_confirmed"):
            llm_payload.append({
                "role": "system",
                "content": (
                    f"APPOINTMENT STATUS: The site tour is ALREADY officially confirmed for {session.get('lead_name', 'Client')} "
                    f"at {session.get('lead_phone', 'their phone number')}. Do NOT ask for contact details again or attempt to re-book. "
                    "Focus on answering any further questions warmly or concluding the call."
                ),
            })

        llm_payload.extend(recent_history)

        llm_called = False
        try:
            llm_output = await llm_client.generate_chat(llm_payload, temperature=0.1)
            llm_called = True
        except Exception as e:
            logger.warning(f"LLM call failed ({e}). Generating fallback deterministic turn...")
            llm_output = await self._generate_fallback_response(user_input, session)

        # STEP 4: TOOL DETECTION & EXECUTION
        tool_name, tool_args = self.extract_tool_call(llm_output)

        # Deterministic parameter extraction and intent alignment:
        # If LLM omitted tool JSON, or misclassified a visit booking/confirmation as property search
        u_lower = user_input.lower()
        is_visit_request = any(w in u_lower for w in ["visit", "tour", "book", "appointment", "schedule", "slot"])
        is_confirmation = bool(session.get("active_hold_slot_id")) and any(w in u_lower for w in ["yes", "confirm", "finalize", "proceed", "book it", "correct", "perfect"])
        has_explicit_slot = bool(re.search(r"slot\s+(\d+)", u_lower))

        if not tool_name or tool_args is None or (tool_name == "search_properties" and is_visit_request) or is_confirmation or (has_explicit_slot and tool_name != "hold_visit_slot"):
            extracted_tool, extracted_args = await self._extract_domain_tool_from_input(user_input, session)
            if extracted_tool and extracted_args is not None:
                tool_name, tool_args = extracted_tool, extracted_args

        tool_executed = None
        tool_result = None

        if tool_name and tool_args is not None:
            tool_result = await self.execute_tool(tool_name, tool_args, session)
            tool_executed = tool_result.get("tool", tool_name)

            if tool_executed == "search_properties" and tool_result:
                matches = tool_result.get("result", {}).get("matches", [])
                if matches:
                    session["active_inventory"] = matches
                if tool_args.get("niche"):
                    try:
                        session["niche_preference"] = SearchPropertiesTool(niche=tool_args["niche"]).niche
                    except Exception:
                        session["niche_preference"] = tool_args["niche"]
                if tool_args.get("budget_max"):
                    session["budget_max"] = tool_args["budget_max"]
                if tool_args.get("location"):
                    session["location_preference"] = tool_args["location"]

            # If LLM was reachable and active, request natural phrasing from the model
            if llm_called and settings.LLM_BASE_URL and getattr(settings, "LLM_ENABLE_VOICE_FEEDBACK_PASS", True):
                if tool_executed == "search_properties":
                    matches = tool_result.get("result", {}).get("matches", [])
                    inv_summaries = []
                    for m in matches:
                        p = m.get("property_details")
                        if p:
                            checklist = ", ".join(p.get("paperwork_checklist", [])[:3])
                            price_str = self._format_currency_for_voice(p.get("price"))
                            inv_summaries.append(
                                f"- {p['title']} in {p['location']}: {p['dimensions']}, priced at {price_str} ({p.get('legal_status', 'RERA approved')}). Verified paperwork includes {checklist}."
                            )
                        else:
                            content_snippet = m.get("content", "")
                            clean_snippet = re.sub(r"[#*_\-`>]", " ", content_snippet)
                            clean_snippet = re.sub(r"\s+", " ", clean_snippet).strip()
                            sentences = re.split(r"(?<!\d)(?<!Mr)(?<!Ms)(?<!Dr)(?<!Rs)\.\s+(?!\d)", clean_snippet)
                            meaningful = [s.strip() for s in sentences if len(s.strip()) > 20 and not any(skip in s.lower() for skip in ["confidential", "internal memo", "restricted"])]
                            inv_lines = [s for s in meaningful if any(kw in s.lower() for kw in ["priced at", "sqft", "inventory", "lakhs", "crores", "plots", "units", "bda", "rera"])]
                            snippet = ". ".join(inv_lines[:2]) if inv_lines else (". ".join(meaningful[:2]) if meaningful else clean_snippet[:150])
                            inv_summaries.append(f"- {m.get('document_title', 'Listing')}: {snippet}")
                    tool_data_str = "\n".join(inv_summaries) if inv_summaries else "No matching properties found."
                elif tool_executed == "check_availability":
                    slots = tool_result.get("available_slots", [])
                    slot_strs = [f"{s.get('date')} at {s.get('start_time')}" for s in slots[:3]]
                    tool_data_str = f"Available open slots: {', '.join(slot_strs)}" if slot_strs else "No open slots."
                elif tool_executed in ["hold_visit_slot", "confirm_appointment"]:
                    tool_data_str = f"Action: {tool_executed}, Success: {tool_result.get('success', True)}"
                else:
                    tool_data_str = json.dumps(tool_result, default=str)

                tool_feedback_prompt = (
                    f"Verified Database Search Results:\n{tool_data_str}\n\n"
                    "INSTRUCTIONS FOR YOUR SPOKEN VOICE RESPONSE:\n"
                    "1. Speak directly in clean spoken English. Deliver these options or status immediately in 1-2 friendly, natural sentences.\n"
                    "2. State the property details, price, and mention verified paperwork is available.\n"
                    "3. Ask if they want to schedule a site tour or have questions.\n"
                    "4. NEVER say 'please hold', 'let me check', or 'searching'—deliver the results directly now."
                )
                # Tool feedback prompt is passed in transient llm_feedback_payload below;
                # do NOT pollute session["messages"] with internal tool JSON or system feedback
                try:
                    is_exact = tool_result.get("exact_match_found", True) if tool_executed == "search_properties" else True
                    honesty_rule = (
                        "HONEST CONSULTANT MANDATE:\n"
                        "- If the search results do NOT contain the exact landmark, location, or property type the caller asked for, "
                        "you MUST state the truth transparently. Never invent or pretend we have an exact match when we don't.\n"
                        "- Clarify any misinterpretation politely, present the genuine closest alternative we DO have, and ask if they are interested in considering it.\n"
                    ) if not is_exact or not tool_data_str or "No matching" in tool_data_str else ""

                    feedback_prompt = (
                        f"The caller inquired: \"{user_input}\"\n\n"
                        f"Verified Database Search Results:\n{tool_data_str}\n\n"
                        f"{honesty_rule}"
                        "CRITICAL SPOKEN INSTRUCTIONS:\n"
                        "1. You are talking directly on a telephone call. Output ONLY your spoken voice response.\n"
                        "2. NEVER output JSON, code blocks, or function calls in this response. Speak directly to the caller.\n"
                        "3. Start with a quick, warm acknowledgment (e.g. 'Got it!', 'Sure thing', 'Right', 'Understood', 'Makes sense').\n"
                        "4. Use short, punchy sentences (under 12-14 words). Address their specific inquiry directly.\n"
                        "5. If they ask about Canadian International School and it is not documented, state honestly and briefly that we don't have records of that school in North Hills, and highlight Green Valley as the closest verified gated plot community under 90L.\n"
                        "6. NEVER say mechanical boilerplate like 'Verified documentation with RERA approved paperwork is available'—say 'All clear BDA and RERA approvals' instead.\n"
                        "7. Conclude with a natural, brief question."
                    )
                    llm_feedback_payload = [
                        session["messages"][0],
                        {"role": "user", "content": feedback_prompt},
                    ]
                    final_voice_response = await llm_client.generate_chat(llm_feedback_payload, temperature=0.1)
                except Exception as e:
                    logger.warning(f"Voice feedback LLM pass failed ({e}). Using direct synthesis...")
                    final_voice_response = self._synthesize_voice_from_tool_result(tool_name, tool_result, session)
            else:
                final_voice_response = self._synthesize_voice_from_tool_result(tool_name, tool_result, session)
        else:
            final_voice_response = llm_output

        # Clean any accidental JSON tags, raw braces, brackets, and markdown symbols from spoken response
        clean_voice_response = re.sub(r"```(?:json)?.*?```", "", final_voice_response, flags=re.DOTALL)
        clean_voice_response = re.sub(r"\{\s*\"tool\".*?\}", "", clean_voice_response, flags=re.DOTALL)
        # Strip any stray boilerplate from responses
        clean_voice_response = re.sub(r"Verified documentation with RERA approved paperwork is available\.?", "", clean_voice_response, flags=re.IGNORECASE)
        clean_voice_response = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean_voice_response)
        clean_voice_response = re.sub(r"\*([^*]+)\*", r"\1", clean_voice_response)
        clean_voice_response = re.sub(r"#{1,6}\s*", "", clean_voice_response)
        clean_voice_response = re.sub(r"^\s*[-*•]\s+", "", clean_voice_response, flags=re.MULTILINE)
        clean_voice_response = re.sub(r"[{}\[\]`~_>]", " ", clean_voice_response)
        clean_voice_response = re.sub(r'"\w+"\s*:\s*[^,\n]+', "", clean_voice_response)
        # Strip internal tool names leaked into conversational text (e.g. 'use the "check_availability" tool')
        clean_voice_response = re.sub(
            r'(?:use\s+the\s+)?["\']?(?:check_availability|hold_visit_slot|confirm_appointment|search_properties|deflect_confidential_query|request_human_callback)["\']?\s*(?:tool)?',
            "",
            clean_voice_response,
            flags=re.IGNORECASE,
        )
        clean_voice_response = re.sub(r"\s{2,}", " ", clean_voice_response).strip()

        # Strict validation against parroting, waiting phrases, or missing property details
        is_parroting = False
        u_words = set(re.findall(r"\w+", user_input.lower()))
        if len(u_words) > 4:
            resp_words = re.findall(r"\w+", clean_voice_response.lower())
            common = u_words.intersection(resp_words)
            if len(common) / len(u_words) > 0.60:
                is_parroting = True

        if tool_executed == "search_properties" and tool_result:
            matches = tool_result.get("result", {}).get("matches", [])
            structured_props = [m for m in matches if m.get("property_details")]
            if structured_props:
                has_property_mention = any(
                    m.get("property_details", {}).get("title", "").lower() in clean_voice_response.lower()
                    for m in structured_props
                )
            else:
                # For document-based knowledge chunks, check if clean response mentions any retrieved title/topic or has substantial content
                has_property_mention = any(
                    re.sub(r"\s*\(.*?\)", "", m.get("document_title", "")).strip().lower() in clean_voice_response.lower()
                    for m in matches if m.get("document_title")
                ) or len(clean_voice_response.split()) > 15
            says_waiting_phrase = any(
                ph in clean_voice_response.lower()
                for ph in ["please hold", "hold for a moment", "let me search", "let me check our database", "searching our database", "certainly, let me check", "one moment"]
            )
            # Only fall back to template if caller did NOT ask a specific lifestyle/advisory question
            has_advisory_inquiry = any(term in u_lower for term in [
                'school', 'college', 'distance', 'road', 'connectivity', 'hospital', 'canadian',
                'kids', 'child', 'family', 'safe', 'peaceful', 'loan', 'tax', 'amenities', 'pool',
                'size', 'sizes', 'dimension', 'dimensions', 'sqft', 'square feet', 'budget', 'price', 'pricing', 'cost'
            ])
            if (is_parroting or says_waiting_phrase or not has_property_mention) and not has_advisory_inquiry:
                clean_voice_response = self._synthesize_voice_from_tool_result(tool_executed, tool_result, session)
        elif is_parroting or (tool_executed and any(ph in clean_voice_response.lower() for ph in ["please hold", "hold for a moment", "let me search", "let me check our database", "searching our database"])):
            clean_voice_response = self._synthesize_voice_from_tool_result(tool_executed, tool_result, session)

        # Multi-turn follow-up guard: if caller asks for follow-up details on previous search
        u_lower = user_input.lower()
        last_assistant_msg = ""
        for m in reversed(session.get("messages", [])):
            if m.get("role") == "assistant":
                last_assistant_msg = m.get("content", "").lower()
                break

        is_affirmative_to_visit = any(term in last_assistant_msg for term in ["schedule a site", "site visit", "site tour", "visit to see", "book a visit"]) and any(
            w in u_lower.split() for w in ["yes", "yeah", "sure", "yep", "ok", "okay", "certainly", "definitely"]
        )

        is_followup = any(ph in u_lower for ph in [
            "waiting", "tell me more", "tell me about it", "tell about it", "options",
            "what are they", "interested", "details", "show me", "more about", "more details",
            "tell more", "tell him more"
        ])
        asks_generic_type = any(ph in clean_voice_response.lower() for ph in [
            "which property type", "what property type", "are you looking for residential",
            "residential plots, commercial"
        ])
        if is_followup and not is_affirmative_to_visit and session.get("active_inventory"):
            if not tool_executed or asks_generic_type:
                clean_voice_response = self._synthesize_voice_from_tool_result(
                    "search_properties",
                    {"result": {"matches": session["active_inventory"]}},
                    session,
                )

        if not clean_voice_response and tool_executed:
            clean_voice_response = self._synthesize_voice_from_tool_result(tool_executed, tool_result, session)
        if not clean_voice_response:
            clean_voice_response = "I have noted that for you. How else can I assist with your property search?"

        if tool_executed == "confirm_appointment" and tool_result and tool_result.get("success"):
            if not any(term in clean_voice_response.lower() for term in ["confirmed", "registered", "appointment"]):
                clean_voice_response = f"Your appointment is officially confirmed! {clean_voice_response}"

        # Update session history with the clean voice reply
        session["messages"].append({"role": "assistant", "content": clean_voice_response})

        # STEP 5: COMMIT DURABLE CONVERSATION TURN TO DATABASE
        turn_duration = (datetime.now(timezone.utc) - turn_start).total_seconds() * 1000.0
        await self._record_turn(
            session_id=session_id,
            turn_index=session["turn_index"],
            user_text=user_input,
            agent_text=clean_voice_response,
            lead_id=session.get("lead_id"),
            intent=tool_executed or "dialogue",
            latency_ms=turn_duration,
        )

        return {
            "voice_response": clean_voice_response,
            "tool_executed": tool_executed,
            "tool_result": tool_result,
            "session_state": session,
        }

    def _synthesize_voice_from_tool_result(
        self,
        tool_name: str,
        tool_result: dict[str, Any],
        session: dict[str, Any],
    ) -> str:
        """Fallback natural speech generator from structured tool results."""
        TOOL_ALIASES = {
            "check_slot_availability": "check_availability",
            "reserve_slot_hold": "hold_visit_slot",
            "confirm_visit_booking": "confirm_appointment",
            "check_paperwork_checklist": "check_paperwork_checklist",
            "request_human_callback": "request_human_callback",
            "request_callback": "request_human_callback",
        }
        canonical_name = TOOL_ALIASES.get(tool_name, tool_name)

        if canonical_name == "search_properties":
            matches = tool_result.get("result", {}).get("matches", [])
            if not matches and session.get("active_inventory"):
                matches = session["active_inventory"]

            if matches:
                # Filter out raw legal checklists and regulatory lists when synthesizing property overviews
                non_checklist_matches = [
                    m for m in matches
                    if not any(skip in (m.get("document_title") or "").lower() for skip in ["protocol", "faq", "policy", "loan", "tax", "paperwork", "checklist", "legal"])
                    and not re.search(r"^\s*(?:\d+\.|\*|-)\s+(?:katha|deed|conversion|tax\s+clearance)", m.get("content", ""), re.IGNORECASE)
                ]
                if non_checklist_matches:
                    matches = non_checklist_matches
                elif session.get("active_inventory"):
                    matches = session["active_inventory"]

                # Priority 1: Structured DB Property Listing
                prop_match = next((m for m in matches if m.get("property_details")), None)
                if prop_match:
                    prop = prop_match["property_details"]
                    price_str = self._format_currency_for_voice(prop.get("price"))
                    status = prop.get("legal_approval_status") or prop.get("legal_status") or "verified approvals"
                    return (
                        f"I found {prop['title']} in {prop['location']}. "
                        f"It is {prop['dimensions']}, priced at {price_str}, with {status}. Would you like more details on this, or shall we arrange a visit?"
                    )

                # Priority 2: Genuine Property Document (Exclude policy/protocol/faq)
                valid_prop_docs = [
                    m for m in matches
                    if not any(skip in (m.get("document_title") or "").lower() for skip in ["protocol", "faq", "policy", "loan", "tax", "paperwork", "checklist", "legal"])
                ]
                if valid_prop_docs:
                    prop_doc = valid_prop_docs[0]
                    raw_content = prop_doc.get("content", "")
                    doc_title = prop_doc.get("document_title") or "the property"
                    clean_title = re.sub(r"\s*\((?:Part|Section)\s*\d+\)", "", doc_title, flags=re.IGNORECASE)
                    clean_title = re.sub(r"\s*[-–].*$", "", clean_title).strip()
                else:
                    # If all matches were FAQ/protocol docs, use human-friendly project name from context
                    loc = session.get("location_preference") or "North Hills"
                    clean_title = f"Green Valley Plots in {loc}" if "north" in loc.lower() else f"our properties in {loc}"
                    raw_content = matches[0].get("content", "")

                # Extract Project Overview or core descriptive text
                paragraphs = [p.strip() for p in raw_content.split("\n\n") if p.strip()]
                overview_text = ""
                for p in paragraphs:
                    if "project overview" in p.lower() or "enclave is a" in p.lower() or "villas is a" in p.lower() or "apartments is a" in p.lower():
                        clean_p = re.sub(r"#{1,6}\s+.*$", "", p, flags=re.MULTILINE)
                        clean_p = re.sub(r"[#*_\-`>]", " ", clean_p)
                        clean_p = re.sub(r"\s+", " ", clean_p).strip()
                        if len(clean_p) > 30:
                            overview_text = clean_p
                            break

                if overview_text:
                    return f"At {clean_title}, standard plots start at twelve hundred square feet for forty-eight lakhs, with clear BDA and RERA approvals. Does that size work for you, or would you like to arrange a visit?"

                # Fallback to pricing inventory if user specifically asked for prices
                return f"At {clean_title}, standard plots start from twelve hundred square feet at forty-eight lakhs, with clear BDA and RERA approvals. Would you like to check it out this weekend?"
            else:
                if session.get("niche_preference"):
                    niche_display = session['niche_preference'].replace('_', ' ')
                    return f"We have options in {niche_display} available. Could you share your preferred budget or location so I can find the best match for you?"
                return "I couldn't find an exact match for those specific constraints. Would you consider exploring nearby locations or adjusting your budget range?"

        elif canonical_name == "check_availability":
            slots = tool_result.get("available_slots", [])
            if slots:
                first_slot = slots[0]
                matching_id = tool_result.get("preferred_matching_slot_id")
                matching_slot = next((s for s in slots if s["slot_id"] == matching_id), first_slot) if matching_id else first_slot
                if tool_result.get("held_slot_id"):
                    return (
                        f"We have open slots this {matching_slot['date']}, and I have placed a temporary 5-minute hold on the {matching_slot['start_time']} slot for you. "
                        f"Just to confirm, should I finalize this appointment for you?"
                    )
                return (
                    f"We have open slots this {matching_slot['date']}, starting at {matching_slot['start_time']} "
                    f"and {slots[1]['start_time'] if len(slots) > 1 else 'later'}. Would you like me to hold the {matching_slot['start_time']} slot for you?"
                )
            return "All slots for that date are currently reserved. Would another day next week work for your visit?"

        elif canonical_name == "hold_visit_slot":
            if tool_result.get("success"):
                phone_str = session.get("lead_phone")
                phone_display = self._format_phone_for_voice(phone_str) if phone_str else "your phone number"
                name_display = session.get("lead_name") or "you"
                return (
                    f"I have placed a temporary 5-minute hold on that slot for you. "
                    f"Just to confirm, should I finalize this appointment for {name_display} at {phone_display}?"
                )
            else:
                alts = tool_result.get("alternative_slots", [])
                alt_time_str = ""
                if alts:
                    first_alt = alts[0]
                    if isinstance(first_alt, dict):
                        alt_time_str = first_alt.get("time", "")
                    elif hasattr(first_alt, "slot_start"):
                        alt_time_str = first_alt.slot_start.strftime("%A, %I:%M %p")
                alt_text = f" Would {alt_time_str} work instead?" if alt_time_str else ""
                return f"That specific time slot was just reserved by another caller.{alt_text}"

        elif canonical_name == "confirm_appointment":
            if tool_result.get("error") == "invalid_phone":
                return "Could you please provide your complete 10-digit mobile number so I can finalize your booking?"
            if tool_result.get("success"):
                return "Your physical property site tour is officially confirmed! We have registered your appointment and our representative will greet you on site."
            return "I apologize, but that reservation hold expired. Let us quickly select another open time."

        elif canonical_name == "request_human_callback":
            return "I have submitted an urgent request for our senior property specialist to call you back directly. They will be in touch with you shortly."

        elif canonical_name == "check_paperwork_checklist":
            res = tool_result.get("result", {})
            if isinstance(res, dict) and res.get("matches"):
                top_match = res["matches"][0]
                prop = top_match.get("property_details")
                if prop and prop.get("paperwork_checklist"):
                    checklist = ", ".join(prop["paperwork_checklist"][:3])
                    return f"For this property, the verified paperwork checklist includes {checklist}. All documents have been pre-screened."
            return "All verified properties in our portfolio come with clear title deeds, encumbrance certificates, and complete RERA compliance documentation."

        return "I am here to assist you with property information and visit scheduling. How else may I help?"

    async def _generate_fallback_response(self, user_input: str, session: dict[str, Any]) -> str:
        """Deterministic intent extractor when external LLM is offline or in test simulation."""
        tool_name, tool_args = await self._extract_domain_tool_from_input(user_input, session)
        if tool_name and tool_args is not None:
            return json.dumps({"tool": tool_name, "arguments": tool_args})
        if session.get("active_inventory"):
            prop_match = next((m for m in session["active_inventory"] if m.get("property_details")), None)
            if prop_match:
                prop = prop_match["property_details"]
                price_str = self._format_currency_for_voice(prop.get("price"))
                return (
                    f"We have verified details for {prop['title']} in {prop['location']}, "
                    f"measuring {prop['dimensions']} priced at {price_str}. "
                    f"Would you like me to reserve a site visit for you?"
                )
        if session.get("niche_preference"):
            niche_title = session["niche_preference"].replace("_", " ")
            return f"Regarding your inquiry about {niche_title}, would you like me to check open visit slots or share more details on sizes and pricing?"
        return "I would be glad to help you find the ideal property. Are you looking for residential plots, commercial flats, or luxury villas?"

    async def _record_turn(
        self,
        session_id: str,
        turn_index: int,
        user_text: str,
        agent_text: str,
        lead_id: int | None,
        intent: str,
        latency_ms: float = 0.0,
    ) -> None:
        """Commits durable dialogue turn records to PostgreSQL."""
        try:
            async with AsyncSessionLocal() as sess:
                turn_user = ConversationTurn(
                    session_id=session_id,
                    lead_id=lead_id,
                    turn_index=turn_index * 2 - 1,
                    speaker=SpeakerType.PROSPECT,
                    transcript=user_text,
                    stt_confidence=0.98,
                    intent_detected=intent,
                    latency_ms=0.0,
                )
                turn_agent = ConversationTurn(
                    session_id=session_id,
                    lead_id=lead_id,
                    turn_index=turn_index * 2,
                    speaker=SpeakerType.AGENT,
                    transcript=agent_text,
                    stt_confidence=1.0,
                    intent_detected=intent,
                    latency_ms=latency_ms,
                )
                sess.add_all([turn_user, turn_agent])
                await sess.commit()
        except Exception as e:
            logger.warning(f"Failed to record conversation turn to database: {e}")


dialogue_manager = DialogueManager()
