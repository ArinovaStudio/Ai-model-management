"""Demonstration & Verification Script for Lead-Based Conversational Flow.

Creates:
1. High-intent valid leads directly tied to data/docs (Palm Meadows, Emerald Lakeview, Logistics Corridor).
2. Poor/Junk leads (dummy phone, spam name, zero discovery signals) to test triage.
3. Tests lead evaluation and prompt directive synthesis.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Direct import avoiding src.agent package eager loading
import importlib.util

spec = importlib.util.spec_from_file_location("lead_service", str(Path(__file__).resolve().parent.parent / "src" / "agent" / "lead_service.py"))
lead_mod = importlib.util.module_from_spec(spec)
sys.modules["lead_service"] = lead_mod
spec.loader.exec_module(lead_mod)
lead_service = lead_mod.LeadService()

SAMPLE_LEADS = [
    # 1. High Intent Luxury Villa Lead (Palm Meadows Villas)
    {
        "name": "Dr. Vikram Sethi",
        "phone_number": "+919845012345",
        "niche": "luxury_villa",
        "budget_min": 30000000.0,
        "budget_max": 48500000.0,
        "preferred_location": "Airport Road, Devanahalli",
        "inquiry_notes": "Interested in the 5BHK Presidential Villa at Palm Meadows. Need to verify BIAAPA approvals and private pool maintenance before scheduling a weekend visit.",
    },
    # 2. Family Townhome Lead (Emerald Lakeview Townhomes)
    {
        "name": "Pooja Banerjee",
        "phone_number": "+919820098765",
        "niche": "residential_plot",
        "budget_min": 8500000.0,
        "budget_max": 12000000.0,
        "preferred_location": "North Hills",
        "inquiry_notes": "Looking for a 3BHK lakeview townhome or plot close to international schools. Inquiring about SBI pre-approved home loan options.",
    },
    # 3. Commercial Investor Lead (Logistics Corridor Land)
    {
        "name": "Karan Mehra",
        "phone_number": "+919711223344",
        "niche": "land_parcel",
        "budget_min": 50000000.0,
        "budget_max": 90000000.0,
        "preferred_location": "Whitefield Tech Corridor",
        "inquiry_notes": "Looking for 2 to 5 acre industrial/logistics parcel with NH road frontage. Need 40-year parent deed clear title details.",
    },
    # 4. Poor Lead 1: Dummy repeating phone number (Spam)
    {
        "name": "Test User",
        "phone_number": "+910000000000",
        "niche": "luxury_villa",
        "budget_min": 1000000.0,
        "budget_max": 2000000.0,
        "preferred_location": "Downtown",
        "inquiry_notes": "call me",
    },
    # 5. Poor Lead 2: Junk name & zero discovery intent
    {
        "name": "asdf",
        "phone_number": "+919988776655",
        "niche": None,
        "budget_min": None,
        "budget_max": None,
        "preferred_location": None,
        "inquiry_notes": "",
    },
]


def test_lead_evaluation():
    print("=" * 70)
    print("LEAD QUALITY & TRIAGE VERIFICATION TEST")
    print("=" * 70)

    for i, lead in enumerate(SAMPLE_LEADS, 1):
        is_valid, reason = lead_service.evaluate_lead_quality(
            name=lead["name"],
            phone_number=lead["phone_number"],
            niche=lead.get("niche"),
            budget=lead.get("budget_max"),
            location=lead.get("preferred_location"),
            inquiry_notes=lead.get("inquiry_notes"),
        )

        status_str = "VALID (READY FOR CALL)" if is_valid else "FLAGGED & IGNORED"
        print(f"\nLead #{i}: {lead['name']} ({lead['phone_number']})")
        print(f"Status: [{status_str}]")
        if not is_valid:
            print(f"Reason: {reason}")
        else:
            directive = lead_service.build_lead_prompt_directive(lead)
            print("Generated Agent System Prompt Context:")
            for line in directive.split("\n"):
                print(f"  | {line}")

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS EXECUTED CLEANLY")
    print("=" * 70)


if __name__ == "__main__":
    test_lead_evaluation()
