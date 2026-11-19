"""Telephony Session Manager & Mid-Call Disconnect Recovery Subsystem.

Tracks call lifecycles, persists session states, and handles unexpected VoIP
drops/WebSocket disconnects by generating partial conversational summaries
and queueing urgent operator callback notifications in PostgreSQL.
"""

from datetime import datetime, timezone
from typing import Any, Optional
from sqlalchemy import select
from src.core.logging import logger
from src.db.session import AsyncSessionLocal
from src.db.models import ConversationTurn, OperatorNotification, Lead
from src.agent.llm_client import llm_client
from src.agent.dialogue_manager import dialogue_manager
from src.core.config import settings


class TelephonySessionManager:
    """Manages telephony call sessions and resilient failure recovery."""

    def __init__(self):
        self._active_sessions: dict[str, dict[str, Any]] = {}

    def create_session(
        self,
        session_id: str,
        caller_phone: str | None = None,
        caller_name: str | None = None,
    ) -> dict[str, Any]:
        """Initializes a new call session."""
        if caller_phone:
            cleaned = caller_phone.strip()
            if not cleaned.startswith("+") and not cleaned.startswith("0") and len(cleaned) >= 10:
                cleaned = f"+{cleaned}"
            caller_phone = cleaned

        session = {
            "session_id": session_id,
            "caller_phone": caller_phone,
            "caller_name": caller_name,
            "start_time": datetime.now(timezone.utc),
            "status": "active",  # active, completed, dropped
            "turn_count": 0,
            "last_active": datetime.now(timezone.utc),
        }
        self._active_sessions[session_id] = session
        # Link into dialogue manager
        dialogue_manager.get_session(session_id=session_id, lead_phone=caller_phone)
        logger.info(f"Initialized telephony session: {session_id} (caller: {caller_phone})")
        return session

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Retrieves active session state."""
        return self._active_sessions.get(session_id)

    def mark_turn_activity(self, session_id: str) -> None:
        """Updates last active timestamp and increment turn count."""
        session = self._active_sessions.get(session_id)
        if session:
            session["turn_count"] += 1
            session["last_active"] = datetime.now(timezone.utc)

    def end_session_normal(self, session_id: str) -> dict[str, Any]:
        """Gracefully completes a call session and evicts from active memory."""
        session = self._active_sessions.pop(session_id, None)
        dialogue_manager.close_session(session_id)
        if session:
            session["status"] = "completed"
            session["end_time"] = datetime.now(timezone.utc)
            logger.info(f"Completed call session gracefully: {session_id}")
            
            # Asynchronously trigger 75-200 word executive post-call summary
            try:
                import asyncio
                from src.agent.lead_service import lead_service
                asyncio.create_task(lead_service.generate_post_call_summary(session_id))
            except Exception as e:
                logger.debug(f"Post-call summary background trigger error: {e}")

            return session
        return {"session_id": session_id, "status": "completed"}

    def reap_stale_sessions(self, max_idle_seconds: int = 300) -> int:
        """Evicts zombie call sessions abandoned without graceful hangup or close."""
        now = datetime.now(timezone.utc)
        stale_ids = []
        for sid, s in list(self._active_sessions.items()):
            last_active = s.get("last_active") or s.get("start_time") or now
            if (now - last_active).total_seconds() > max_idle_seconds:
                stale_ids.append(sid)

        for sid in stale_ids:
            logger.info(f"Reaping stale/abandoned session {sid} (idle > {max_idle_seconds}s)")
            self.end_session_normal(sid)
        return len(stale_ids)

    async def handle_mid_call_disconnect(
        self,
        session_id: str,
        disconnect_reason: str = "unexpected_websocket_disconnect",
    ) -> dict[str, Any]:
        """Recovers from unexpected call drops / network disconnects:

        1. Preserves session state and marks status as 'dropped'.
        2. Retrieves all durable dialogue turns from PostgreSQL.
        3. Formulates a concise partial call summary of prospect intent.
        4. Queues an urgent record in `operator_notifications` for human callback.
        """
        session = self._active_sessions.get(session_id, {
            "session_id": session_id,
            "caller_phone": None,
            "caller_name": None,
            "turn_count": 0,
        })
        session["status"] = "dropped"
        session["disconnect_reason"] = disconnect_reason
        from src.core.metrics import metrics_registry
        metrics_registry.record_call_drop()
        logger.warning(
            f"Mid-call disconnect detected for session {session_id}! Initiating recovery pipeline..."
        )

        try:
            # Retrieve conversation turns from database
            turns_summary_list = []
            lead_id = None
            caller_phone = session.get("caller_phone")

            async with AsyncSessionLocal() as sess:
                query = (
                    select(ConversationTurn)
                    .where(ConversationTurn.session_id == session_id)
                    .order_by(ConversationTurn.turn_index.asc())
                )
                result = await sess.execute(query)
                turns = result.scalars().all()

                for t in turns:
                    if t.lead_id and not lead_id:
                        lead_id = t.lead_id
                    turns_summary_list.append(f"{t.speaker.value.upper()}: {t.transcript}")

            # Check dialogue manager session state if phone wasn't captured in telephony session
            dm_session = dialogue_manager.sessions.get(session_id, {})
            if not caller_phone and dm_session.get("lead_phone"):
                caller_phone = dm_session.get("lead_phone")
            if not lead_id and dm_session.get("lead_id"):
                lead_id = dm_session.get("lead_id")

            # Generate partial summary
            transcript_text = "\n".join(turns_summary_list)
            partial_summary = await self._generate_partial_summary(transcript_text, dm_session)

            # Dynamic triage urgency scoring: mid-call disconnects default to urgent for rapid human operator callback
            calculated_urgency = "urgent"

            # Commit urgent operator notification
            async with AsyncSessionLocal() as sess:
                notification = OperatorNotification(
                    session_id=session_id,
                    lead_id=lead_id,
                    phone_number=caller_phone,
                    reason=disconnect_reason,
                    partial_summary=partial_summary,
                    urgency=calculated_urgency,
                    status="pending",
                )
                sess.add(notification)
                await sess.commit()
                await sess.refresh(notification)

                notification_id = notification.id
                logger.info(
                    f"Created urgent OperatorNotification #{notification_id} for session {session_id} "
                    f"(Status: pending, Urgency: urgent, Phone: {caller_phone})"
                )

            return {
                "session_id": session_id,
                "notification_id": notification_id,
                "status": "dropped",
                "partial_summary": partial_summary,
                "phone_number": caller_phone,
                "turns_recovered": len(turns_summary_list),
            }
        except Exception as e:
            logger.error(f"Error handling mid-call disconnect for session {session_id}: {e}", exc_info=True)
            raise
        finally:
            dialogue_manager.close_session(session_id)
            self._active_sessions.pop(session_id, None)

    async def _generate_partial_summary(
        self,
        transcript_text: str,
        dm_session: dict[str, Any],
    ) -> str:
        """Invokes LLM (or deterministic summary fallback) for interrupted calls."""
        if not transcript_text:
            return "Call disconnected immediately after connection before any conversational turns."

        # In local dev environment with CPU LLM, generate deterministic summary instantly
        if settings.LLM_BASE_URL and "localhost" in settings.LLM_BASE_URL:
            niche = dm_session.get("niche_preference", "residential plots")
            held_slot = dm_session.get("active_hold_slot_id")
            slot_status = f"requested visit slot (Hold Slot #{held_slot})" if held_slot else "inquired about site visits"
            return (
                f"Prospect was interested in {niche}, {slot_status}; "
                f"call dropped before final verbal confirmation could be completed."
            )

        summary_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a real estate operations assistant. A prospective client's phone call was abruptly cut off mid-conversation. "
                    "Analyze the partial transcript and summarize in 1-2 concise sentences what the prospect wanted, what properties or visit dates they discussed, "
                    "and the exact state where the call dropped for an urgent operator callback."
                ),
            },
            {
                "role": "user",
                "content": f"Partial Transcript:\n{transcript_text}\n\nSummary for operator:",
            },
        ]

        try:
            summary = await llm_client.generate_chat(summary_prompt, max_tokens=150)
            if summary and len(summary.strip()) > 15:
                return summary.strip()
        except Exception as e:
            logger.warning(f"LLM summary generation failed ({e}). Using deterministic template.")

        # Deterministic fallback summary
        niche = dm_session.get("niche_preference", "property options")
        held_slot = dm_session.get("active_hold_slot_id")
        slot_status = f"requested visit slot (Hold Slot #{held_slot})" if held_slot else "inquired about site visits"
        return (
            f"Prospect was interested in {niche}, {slot_status}; "
            f"call dropped before final verbal confirmation could be completed."
        )


# Global session manager singleton
session_manager = TelephonySessionManager()
