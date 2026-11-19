"""Secure Vector Retrieval Engine.

Enforces retrieval-layer security guardrails:
1. Filters out probe attacks via ConfidentialityFilter.
2. Hardcodes `WHERE access_label = 'customer_shareable'` at the SQL query level,
   making it mathematically impossible for internal corporate secrets to leak.
3. Retrieves verified property details, dimensions, and paperwork checklists.
"""

from typing import Any
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.session import AsyncSessionLocal
from src.db.models import KnowledgeChunk, PropertyListing, AccessLabel, PropertyNiche
from src.rag.embeddings import embed_text
from src.rag.confidentiality_filter import confidentiality_filter
from src.core.logging import logger


class SecureRetriever:
    """Retrieves verified property domain knowledge with strict confidentiality enforcement."""

    def __init__(self, session_factory=AsyncSessionLocal):
        self.session_factory = session_factory

    async def search_property_knowledge(
        self,
        query_str: str,
        niche_filter: PropertyNiche | str | None = None,
        budget_min: float | None = None,
        budget_max: float | None = None,
        location_filter: str | None = None,
        top_k: int = 3,
        session: AsyncSession | None = None,
    ) -> dict[str, Any]:
        """Executes vector similarity search with query-level customer_shareable filtering.
        
        Returns:
            dict containing:
            - is_deflected (bool): True if restricted query was intercepted
            - deflection_message (str | None): Safe fallback deflection if intercepted
            - matches (list[dict]): Verified retrieved chunks, prices, and paperwork
        """
        # Step 1: Query-level deflection firewall check
        if confidentiality_filter.is_restricted_query(query_str):
            logger.info(f"Query intercepted by confidentiality firewall: '{query_str}'")
            return {
                "is_deflected": True,
                "deflection_message": confidentiality_filter.get_safe_deflection(),
                "matches": [],
            }

        # Step 2: Compute embedding vector for the search query
        query_vector = embed_text(query_str)

        async def _execute(sess: AsyncSession) -> dict[str, Any]:
            # Step 3: Construct SQL query with MANDATORY hardcoded access_label guardrail
            # Using cosine distance operator `<=>` (or `<->` for L2)
            cosine_distance = KnowledgeChunk.embedding.cosine_distance(query_vector)

            stmt = (
                select(KnowledgeChunk, PropertyListing, cosine_distance.label("distance"))
                .outerjoin(PropertyListing, KnowledgeChunk.property_id == PropertyListing.id)
                .where(
                    # HARD SECURITY GUARDRAIL: Only customer_shareable data is ever selected
                    KnowledgeChunk.access_label == AccessLabel.CUSTOMER_SHAREABLE
                )
            )

            # Niche filter: If caller specifies a niche, enforce it strictly on property listings
            # and only allow unlinked knowledge chunks if they are general guides (loan, tax, protocol)
            # or explicitly mention the relevant niche keywords
            if niche_filter:
                niche_val = niche_filter.value if isinstance(niche_filter, PropertyNiche) else str(niche_filter).lower().strip()
                if "plot" in niche_val or "residential" in niche_val:
                    niche_val = PropertyNiche.RESIDENTIAL_PLOT.value
                    niche_keywords = ["plot", "green valley", "residential", "loan", "tax", "protocol", "faq", "paperwork"]
                elif "commercial" in niche_val or "flat" in niche_val or "apartment" in niche_val or "office" in niche_val:
                    niche_val = PropertyNiche.COMMERCIAL_FLAT.value
                    niche_keywords = ["flat", "apartment", "emerald", "commercial", "suite", "loan", "tax", "protocol", "faq", "paperwork"]
                elif "villa" in niche_val or "luxury" in niche_val:
                    niche_val = PropertyNiche.LUXURY_VILLA.value
                    niche_keywords = ["villa", "palm meadows", "luxury", "loan", "tax", "protocol", "faq", "paperwork"]
                elif "land" in niche_val or "parcel" in niche_val:
                    niche_val = PropertyNiche.LAND_PARCEL.value
                    niche_keywords = ["parcel", "logistics", "land", "industrial", "protocol", "faq", "paperwork"]
                else:
                    niche_keywords = []

                if niche_val in [e.value for e in PropertyNiche]:
                    doc_conditions = [KnowledgeChunk.document_title.ilike(f"%{kw}%") for kw in niche_keywords]
                    content_conditions = [KnowledgeChunk.chunk_content.ilike(f"%{kw}%") for kw in niche_keywords]
                    allowed_doc_chunk = and_(
                        KnowledgeChunk.property_id.is_(None),
                        or_(*doc_conditions, *content_conditions) if niche_keywords else False
                    )
                    stmt = stmt.where(
                        or_(PropertyListing.niche == niche_val, allowed_doc_chunk)
                    )

            # Budget and location filters: Prioritize properties matching constraints
            if budget_max is not None:
                stmt = stmt.where(
                    or_(PropertyListing.price <= budget_max, KnowledgeChunk.property_id.is_(None))
                )
            if budget_min is not None:
                stmt = stmt.where(
                    or_(PropertyListing.price >= budget_min, KnowledgeChunk.property_id.is_(None))
                )
            if location_filter:
                clean_loc = location_filter.split(",")[0].strip()
                stmt = stmt.where(
                    or_(
                        PropertyListing.location.ilike(f"%{clean_loc}%"),
                        and_(
                            KnowledgeChunk.property_id.is_(None),
                            or_(
                                KnowledgeChunk.document_title.ilike(f"%{clean_loc}%"),
                                KnowledgeChunk.chunk_content.ilike(f"%{clean_loc}%"),
                                KnowledgeChunk.document_title.ilike("%guide%"),
                                KnowledgeChunk.document_title.ilike("%faq%"),
                                KnowledgeChunk.document_title.ilike("%protocol%"),
                                KnowledgeChunk.document_title.ilike("%tax%"),
                                KnowledgeChunk.document_title.ilike("%loan%")
                            )
                        )
                    )
                )

            # Order by closest vector match (smallest cosine distance), fetching wider pool for distinct property deduplication
            stmt = stmt.order_by("distance").limit(max(top_k * 3, 10))

            result = await sess.execute(stmt)
            rows = result.all()

            def _build_match_item(c: KnowledgeChunk, p: PropertyListing | None, d: float) -> dict[str, Any]:
                item = {
                    "document_title": c.document_title,
                    "content": c.chunk_content,
                    "similarity_score": round(1.0 - float(d), 4),
                    "access_label": c.access_label.value,
                    "property_details": None,
                }
                if p:
                    item["property_details"] = {
                        "id": p.id,
                        "title": p.title,
                        "project_name": p.project_name,
                        "niche": p.niche.value,
                        "location": p.location,
                        "price": float(p.price),
                        "dimensions": p.dimensions,
                        "legal_status": p.legal_approval_status,
                        "paperwork_checklist": p.paperwork_checklist,
                    }
                return confidentiality_filter.sanitize_retrieved_payload(item)

            matches = []
            seen_property_ids = set()

            # Pass 1: Prioritize distinct property listings so one property does not crowd out others
            # Enforce similarity threshold to reject garbage/out-of-scope vector matches
            MIN_SIMILARITY_THRESHOLD = 0.45
            for chunk, prop, dist in rows:
                sim = round(1.0 - float(dist), 4)
                if sim < MIN_SIMILARITY_THRESHOLD:
                    continue
                if len(matches) >= top_k:
                    break
                if prop:
                    if prop.id in seen_property_ids:
                        continue
                    seen_property_ids.add(prop.id)
                    matches.append(_build_match_item(chunk, prop, dist))

            # Pass 2: If capacity remains, backfill with general knowledge chunks or additional listings above threshold
            if len(matches) < top_k:
                for chunk, prop, dist in rows:
                    sim = round(1.0 - float(dist), 4)
                    if sim < MIN_SIMILARITY_THRESHOLD:
                        continue
                    if len(matches) >= top_k:
                        break
                    match_item = _build_match_item(chunk, prop, dist)
                    if not any(m["document_title"] == chunk.document_title for m in matches):
                        matches.append(match_item)

            logger.info(f"Retrieved {len(matches)} customer-shareable chunks for query: '{query_str[:50]}'")
            return {
                "is_deflected": False,
                "deflection_message": None,
                "exact_match_found": len(matches) > 0,
                "requested_criteria": {
                    "niche": niche_filter.value if isinstance(niche_filter, PropertyNiche) else niche_filter,
                    "budget_max": budget_max,
                    "location": location_filter,
                },
                "matches": matches,
            }

        if session:
            return await _execute(session)
        async with self.session_factory() as sess:
            return await _execute(sess)


retriever = SecureRetriever()
