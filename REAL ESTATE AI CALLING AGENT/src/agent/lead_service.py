"""Lead Service: Ingestion, Quality Triage & Post-Call Summarization Subsystem.

Provides:
1. Automated lead quality & completeness evaluation (filters out junk/poor leads).
2. Lead conversational brief generation for the AI Voice Agent.
3. 75-to-200 word accurate post-call executive summarizer.
"""

import re
from typing import Any, Optional
from datetime import datetime, timezone
from sqlalchemy import select
from src.core.logging import logger


class LeadService:
    """Handles lead-based calling intelligence and post-call analytics."""

    @staticmethod
    def evaluate_lead_quality(
        name: Optional[str],
        phone_number: str,
        niche: Optional[str] = None,
        budget: Optional[float] = None,
        location: Optional[str] = None,
        inquiry_notes: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Evaluates whether incoming lead data is sufficient for a discovery call.

        Returns (is_valid, flag_reason).
        """
        # 1. Clean and validate phone number
        clean_phone = re.sub(r"[^\d+]", "", phone_number or "")
        digits_only = re.sub(r"\D", "", clean_phone)

        if len(digits_only) < 10:
            return False, "Invalid phone number: Must contain at least 10 digits."

        # Check for obvious dummy repeating sequences (e.g., 0000000000, 1111111111, 1234567890)
        clean_digits = re.sub(r"^91", "", digits_only) if len(digits_only) > 10 and digits_only.startswith("91") else digits_only
        if len(set(clean_digits)) <= 2 or clean_digits in ["1234567890", "0123456789", "9876543210"]:
            return False, "Invalid phone number: Identified as placeholder or dummy digits."

        # 2. Check name if provided
        if name:
            clean_name = name.strip()
            if len(clean_name) < 2:
                return False, "Invalid name: Name must be at least 2 characters."
            if re.match(r"^(test|test\s+user|asdf|qwerty|n/a|na|none|unknown|foo|bar)$", clean_name, re.IGNORECASE):
                return False, f"Invalid name: Identified as test/dummy name '{clean_name}'."

        # 3. Check for substantive discovery signal
        # A valid lead needs at least one clear piece of intent: niche, budget, location, or descriptive note
        has_niche = bool(niche and len(niche.strip()) > 2)
        has_budget = bool(budget and budget > 0)
        has_location = bool(location and len(location.strip()) > 2)
        has_inquiry = bool(inquiry_notes and len(inquiry_notes.strip()) >= 5)

        substantive_signals = sum([has_niche, has_budget, has_location, has_inquiry])

        if substantive_signals == 0:
            return False, "Insufficient lead data: No property niche, budget, location, or inquiry note provided."

        return True, None

    async def ingest_lead(
        self,
        phone_number: str,
        name: Optional[str] = None,
        niche: Optional[str] = None,
        budget_min: Optional[float] = None,
        budget_max: Optional[float] = None,
        preferred_location: Optional[str] = None,
        inquiry_notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """Ingests and triages an incoming website lead."""
        cleaned_phone = phone_number.strip()
        if not cleaned_phone.startswith("+") and not cleaned_phone.startswith("0") and len(cleaned_phone) >= 10:
            cleaned_phone = f"+{cleaned_phone}"

        is_valid, flag_reason = self.evaluate_lead_quality(
            name=name,
            phone_number=cleaned_phone,
            niche=niche,
            budget=budget_max or budget_min,
            location=preferred_location,
            inquiry_notes=inquiry_notes,
        )

        quality_status = "valid" if is_valid else "flagged_poor_data"
        lead_status = "new" if is_valid else "ignored"

        from src.db.session import AsyncSessionLocal
        from src.db.models import Lead

        async with AsyncSessionLocal() as sess:
            query = select(Lead).where(Lead.phone_number == cleaned_phone)
            res = await sess.execute(query)
            lead = res.scalar_one_or_none()

            if lead:
                # Update existing lead with latest inquiry data
                if name:
                    lead.name = name.strip()
                if niche:
                    lead.niche_preference = niche
                if budget_min is not None:
                    lead.budget_min = budget_min
                if budget_max is not None:
                    lead.budget_max = budget_max
                if preferred_location:
                    lead.preferred_location = preferred_location
                if inquiry_notes:
                    lead.form_inquiry = inquiry_notes
                lead.quality_status = quality_status
                lead.flag_reason = flag_reason
                lead.status = lead_status
            else:
                lead = Lead(
                    phone_number=cleaned_phone,
                    name=name.strip() if name else None,
                    niche_preference=niche,
                    budget_min=budget_min,
                    budget_max=budget_max,
                    preferred_location=preferred_location,
                    form_inquiry=inquiry_notes,
                    quality_status=quality_status,
                    flag_reason=flag_reason,
                    status=lead_status,
                )
                sess.add(lead)

            await sess.commit()
            await sess.refresh(lead)

            logger.info(
                f"Ingested lead ID {lead.id} ({lead.phone_number}): quality={quality_status}, status={lead_status}"
            )

            return {
                "lead_id": lead.id,
                "phone_number": lead.phone_number,
                "name": lead.name,
                "quality_status": lead.quality_status,
                "flag_reason": lead.flag_reason,
                "status": lead.status,
            }

    async def get_lead_context_for_caller(self, phone_number: str) -> Optional[dict[str, Any]]:
        """Retrieves lead data to prepare a pre-call conversational context."""
        if not phone_number:
            return None

        clean = re.sub(r"[^\d+]", "", phone_number)
        from src.db.session import AsyncSessionLocal
        from src.db.models import Lead

        async with AsyncSessionLocal() as sess:
            query = select(Lead).where(Lead.phone_number == clean)
            res = await sess.execute(query)
            lead = res.scalar_one_or_none()

            if not lead:
                return None

            # Ignore flagged poor leads
            if lead.quality_status != "valid":
                return None

            return {
                "lead_id": lead.id,
                "name": lead.name,
                "phone_number": lead.phone_number,
                "niche_preference": lead.niche_preference,
                "budget_max": lead.budget_max,
                "preferred_location": lead.preferred_location,
                "form_inquiry": lead.form_inquiry,
            }

    def build_lead_prompt_directive(self, lead_data: dict[str, Any]) -> str:
        """Constructs a natural voice directive for the agent based on lead form data."""
        name_str = lead_data.get("name") or "the prospect"
        details = []
        if lead_data.get("name"):
            details.append(f"- Buyer Name: {lead_data['name']}")
        if lead_data.get("niche_preference"):
            niche_disp = lead_data['niche_preference'].replace('_', ' ')
            details.append(f"- Property Interest / Niche: {niche_disp}")
        if lead_data.get("budget_max"):
            details.append(f"- Stated Budget: INR {lead_data['budget_max']:,.0f}")
        if lead_data.get("preferred_location"):
            details.append(f"- Preferred Location: {lead_data['preferred_location']}")
        if lead_data.get("form_inquiry"):
            details.append(f"- Inquiry Note Submitted: \"{lead_data['form_inquiry']}\"")

        details_str = "\n".join(details)
        return (
            "WEBSITE LEAD FORM CONTEXT (This buyer previously submitted an inquiry online):\n"
            f"{details_str}\n\n"
            f"INITIAL GREETING DIRECTIVE:\n"
            f"Greet {name_str} warmly by name. Acknowledge that you have their website inquiry about their preferred property/niche, "
            "and immediately let them know you're happy to answer any questions or share the latest availability."
        )

    async def generate_post_call_summary(self, session_id: str) -> Optional[str]:
        """Generates a concise, accurate 75-to-200 word executive call summary and persists it."""
        try:
            from src.db.session import AsyncSessionLocal
            from src.db.models import Lead, ConversationTurn
            from src.agent.llm_client import llm_client

            async with AsyncSessionLocal() as sess:
                # 1. Fetch transcript turns
                query = (
                    select(ConversationTurn)
                    .where(ConversationTurn.session_id == session_id)
                    .order_by(ConversationTurn.turn_index.asc())
                )
                res = await sess.execute(query)
                turns = res.scalars().all()

                if not turns:
                    return None

                transcript_lines = [f"{t.speaker.value.upper()}: {t.transcript}" for t in turns]
                transcript_text = "\n".join(transcript_lines)
                lead_id = next((t.lead_id for t in turns if t.lead_id), None)

                # 2. Fetch lead if available
                lead = None
                if lead_id:
                    lead_query = select(Lead).where(Lead.id == lead_id)
                    lead_res = await sess.execute(lead_query)
                    lead = lead_res.scalar_one_or_none()

            # 3. Generate summary via LLM
            client_name = lead.name if lead and lead.name else "Prospective Buyer"
            summary_prompt = (
                f"You are an expert real estate sales intelligence auditor.\n"
                f"Review the following telephone call transcript between AI Representative Aria and {client_name}.\n\n"
                f"TRANSCRIPT:\n{transcript_text}\n\n"
                "TASK:\n"
                "Generate an accurate, professional executive call summary strictly between 75 and 200 words.\n"
                "Include:\n"
                "1. Buyer Profile & Expressed Intent (Niche, budget flexibility, timeline).\n"
                "2. Inquiries & Key Discussion Points (Specific properties discussed, amenities, paperwork, or loans).\n"
                "3. Call Outcome & Next Steps (Site tour status, scheduled appointment, or follow-up needed).\n"
                "Strictly maintain high accuracy. Do not exceed 200 words."
            )

            messages = [
                {"role": "system", "content": "You provide structured, concise, and highly accurate real estate call summaries."},
                {"role": "user", "content": summary_prompt},
            ]

            summary = await llm_client.generate_chat(messages, temperature=0.1)
            clean_summary = summary.strip()

            # 4. Save to Lead record
            if lead_id:
                async with AsyncSessionLocal() as sess:
                    lead_update_query = select(Lead).where(Lead.id == lead_id)
                    res = await sess.execute(lead_update_query)
                    target_lead = res.scalar_one_or_none()
                    if target_lead:
                        target_lead.call_summary = clean_summary
                        await sess.commit()
                        logger.info(f"Saved post-call summary to Lead {lead_id} ({len(clean_summary.split())} words)")

            return clean_summary

        except Exception as e:
            logger.warning(f"Error generating post-call summary for session {session_id}: {e}")
            return None


lead_service = LeadService()

