"""Database seeding script for RealEstateAI.

Populates:
1. Properties across residential plots, commercial flats, villas, and land parcels.
2. Calendar tour slots for the next 7 business days.
3. Knowledge chunks tagged as 'customer_shareable' and 'internal_restricted'.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timezone, timedelta
import random
from src.db.session import SyncSessionLocal
from src.db.models import (
    PropertyListing,
    PropertyNiche,
    KnowledgeChunk,
    AccessLabel,
    CalendarSlot,
    SlotStatus,
)
from src.rag.embeddings import embed_text
from src.core.logging import logger


def generate_synthetic_embedding(dim: int = 384) -> list[float]:
    """Generates a normalized random vector of specified dimension."""
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def get_chunk_embedding(text: str) -> list[float]:
    """Computes real 384-dimensional FastEmbed embedding with fallback."""
    try:
        return embed_text(text)
    except Exception as e:
        logger.warning(f"FastEmbed embedding notice ({e}). Using normalized fallback vector.")
        return generate_synthetic_embedding(384)


def seed_database() -> None:
    session = SyncSessionLocal()
    try:
        logger.info("Checking existing records...")
        existing_props = session.query(PropertyListing).count()
        if existing_props > 0:
            logger.info(f"Database already contains {existing_props} property listings. Skipping seed.")
            return

        logger.info("Seeding property listings...")
        properties = [
            PropertyListing(
                title="Green Valley Premium Residential Plot 42",
                project_name="Green Valley Enclave",
                niche=PropertyNiche.RESIDENTIAL_PLOT,
                location="North Hills, Bengaluru",
                price=4800000.00,  # 48 Lakhs
                dimensions="1200 sqft (30x40 ft)",
                legal_approval_status="RERA & BDA Approved (PRM/KA/RERA/1251/309/PR/190525)",
                paperwork_checklist=[
                    "Registered Title Deed",
                    "RERA Registration Certificate",
                    "Encumbrance Certificate (15 Years)",
                    "Land Conversion Certificate",
                    "Katha Certificate & Extract (A-Katha)",
                    "Property Tax Paid Receipts",
                ],
            ),
            PropertyListing(
                title="Green Valley Corner Villa Plot 108",
                project_name="Green Valley Enclave",
                niche=PropertyNiche.RESIDENTIAL_PLOT,
                location="North Hills, Bengaluru",
                price=8500000.00,  # 85 Lakhs
                dimensions="2400 sqft (40x60 ft)",
                legal_approval_status="RERA Approved",
                paperwork_checklist=[
                    "Registered Title Deed",
                    "RERA Approval Certificate",
                    "Up-to-date Encumbrance Certificate",
                    "A-Katha Certificate",
                ],
            ),
            PropertyListing(
                title="Skyline Heights 3BHK Luxury Flat - Unit 1204",
                project_name="Skyline Heights Tower A",
                niche=PropertyNiche.COMMERCIAL_FLAT,
                location="Whitefield Tech Corridor",
                price=13500000.00,  # 1.35 Cr
                dimensions="1650 sqft Super Built-up",
                legal_approval_status="RERA Approved & Occupancy Certificate Issued",
                paperwork_checklist=[
                    "Occupancy Certificate (OC)",
                    "Commencement Certificate (CC)",
                    "Sanctioned Building Plan",
                    "No Objection Certificate (Fire & Pollution)",
                    "Draft Sale Agreement",
                ],
            ),
            PropertyListing(
                title="Palm Meadows Signature Villa 07",
                project_name="Palm Meadows Estates",
                niche=PropertyNiche.LUXURY_VILLA,
                location="Airport Road, Devanahalli",
                price=32000000.00,  # 3.20 Cr
                dimensions="3800 sqft on 4000 sqft plot",
                legal_approval_status="BIAAPA Approved",
                paperwork_checklist=[
                    "Title Deed with Parent Deeds",
                    "BIAAPA Plan Sanction",
                    "NOC from Airport Authority",
                    "Environmental Clearance",
                ],
            ),
            PropertyListing(
                title="Industrial Warehouse Land Parcel",
                project_name="Logistics Corridor Phase 1",
                niche=PropertyNiche.LAND_PARCEL,
                location="Dabaspete Highway",
                price=45000000.00,  # 4.50 Cr
                dimensions="2.5 Acres Commercial Zone",
                legal_approval_status="KIADB Zoned",
                paperwork_checklist=[
                    "KIADB Allotment Letter",
                    "Non-Agricultural (NA) Industrial Order",
                    "Pollution Control Board Clearance",
                ],
            ),
        ]
        session.add_all(properties)
        session.flush()
        logger.info(f"Inserted {len(properties)} properties.")

        # Seed Knowledge Chunks with strict access labels
        logger.info("Seeding knowledge chunks with customer_shareable and internal_restricted labels...")
        p1 = properties[0]
        chunks = [
            # CUSTOMER SHAREABLE CHUNKS
            KnowledgeChunk(
                property_id=p1.id,
                document_title="Green Valley Plot 42 Brochure & Specs",
                chunk_content=(
                    "Green Valley Enclave Plot 42 is a 1200 sqft east-facing residential plot in North Hills. "
                    "Priced at 48 Lakhs (INR 4,000 per sqft). Fully approved by RERA under registration "
                    "PRM/KA/RERA/1251/309/PR/190525. Paperwork available for immediate buyer verification includes "
                    "Title Deed, A-Katha, and 15-year Encumbrance Certificate."
                ),
                embedding=get_chunk_embedding(
                    "Green Valley Enclave Plot 42 is a 1200 sqft east-facing residential plot in North Hills. "
                    "Priced at 48 Lakhs (INR 4,000 per sqft). Fully approved by RERA under registration "
                    "PRM/KA/RERA/1251/309/PR/190525. Paperwork available for immediate buyer verification includes "
                    "Title Deed, A-Katha, and 15-year Encumbrance Certificate."
                ),
                access_label=AccessLabel.CUSTOMER_SHAREABLE,
                metadata_payload={"section": "pricing_and_specs", "visibility": "public"},
            ),
            KnowledgeChunk(
                property_id=p1.id,
                document_title="Green Valley Legal & Paperwork Checklist",
                chunk_content=(
                    "Official purchase requirements for Green Valley Plots: "
                    "1. Booking advance 10%. 2. Sale agreement within 15 days. "
                    "3. Registration upon full balance. Documents provided: Original Title Deed, "
                    "RERA Registration Copy, Non-Encumbrance Certificate, Conversion Order, and Tax Receipts."
                ),
                embedding=get_chunk_embedding(
                    "Official purchase requirements for Green Valley Plots: "
                    "1. Booking advance 10%. 2. Sale agreement within 15 days. "
                    "3. Registration upon full balance. Documents provided: Original Title Deed, "
                    "RERA Registration Copy, Non-Encumbrance Certificate, Conversion Order, and Tax Receipts."
                ),
                access_label=AccessLabel.CUSTOMER_SHAREABLE,
                metadata_payload={"section": "legal_paperwork", "visibility": "public"},
            ),
            # INTERNAL RESTRICTED CHUNKS (SHOULD NEVER LEAK)
            KnowledgeChunk(
                property_id=p1.id,
                document_title="INTERNAL MEMO: Broker Commission & Margins",
                chunk_content=(
                    "CONFIDENTIAL INTERNAL ONLY: Direct sales commission is 3.5% (approx INR 1,68,000). "
                    "External broker referral fee is capped at 1.5%. Builder base cost is INR 2,800/sqft, "
                    "yielding an internal gross profit margin of 30%. Never quote less than INR 3,850/sqft."
                ),
                embedding=get_chunk_embedding(
                    "CONFIDENTIAL INTERNAL ONLY: Direct sales commission is 3.5% (approx INR 1,68,000). "
                    "External broker referral fee is capped at 1.5%. Builder base cost is INR 2,800/sqft, "
                    "yielding an internal gross profit margin of 30%. Never quote less than INR 3,850/sqft."
                ),
                access_label=AccessLabel.INTERNAL_RESTRICTED,
                metadata_payload={"section": "internal_margins", "confidential": True},
            ),
            KnowledgeChunk(
                property_id=p1.id,
                document_title="INTERNAL MEMO: Unapproved Phase 2 Timeline",
                chunk_content=(
                    "INTERNAL RESTRICTED: Phase 2 land acquisition is pending final court settlement. "
                    "Do not mention Phase 2 launch dates or projected INR 3,200/sqft pre-launch prices to clients "
                    "until final clearances are gazetted in Q4."
                ),
                embedding=get_chunk_embedding(
                    "INTERNAL RESTRICTED: Phase 2 land acquisition is pending final court settlement. "
                    "Do not mention Phase 2 launch dates or projected INR 3,200/sqft pre-launch prices to clients "
                    "until final clearances are gazetted in Q4."
                ),
                access_label=AccessLabel.INTERNAL_RESTRICTED,
                metadata_payload={"section": "future_projects", "confidential": True},
            ),
        ]
        session.add_all(chunks)
        session.flush()
        logger.info(f"Inserted {len(chunks)} knowledge chunks (2 public, 2 internal restricted).")

        # Seed Calendar Slots for next 7 days across all properties
        logger.info("Seeding calendar slots for next 7 business days...")
        now = datetime.now(timezone.utc)
        start_date = now.replace(minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        slot_hours = [(10, 11), (11, 12), (14, 15), (15, 16), (17, 18)]
        calendar_slots = []

        for day_offset in range(7):
            current_day = start_date + timedelta(days=day_offset)
            # Skip Sundays
            if current_day.weekday() == 6:
                continue

            for slot_idx, (start_h, end_h) in enumerate(slot_hours):
                s_start = current_day.replace(hour=start_h)
                s_end = current_day.replace(hour=end_h)
                assigned_prop = properties[(day_offset + slot_idx) % len(properties)]
                slot = CalendarSlot(
                    representative_id="primary_agent_rep_01",
                    slot_start=s_start,
                    slot_end=s_end,
                    status=SlotStatus.AVAILABLE,
                    property_id=assigned_prop.id,
                    notes=f"Physical property site visit slot for {assigned_prop.title} ({s_start.strftime('%A, %b %d %I:%M %p')})",
                )
                calendar_slots.append(slot)

        session.add_all(calendar_slots)
        session.commit()
        logger.info(f"Inserted {len(calendar_slots)} available calendar visit slots across all properties.")
        logger.info(">>> DATABASE SEEDING COMPLETED SUCCESSFULLY! <<<")

    except Exception as e:
        session.rollback()
        logger.error(f"Seeding failed: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    seed_database()
