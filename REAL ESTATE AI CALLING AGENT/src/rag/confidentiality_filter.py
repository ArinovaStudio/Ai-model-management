"""Domain Confidentiality & Guardrail Filter.

Detects unauthorized queries probing for confidential internal company data
(margins, broker commissions, unapproved project phases, employee notes)
and generates deterministic safe deflections without hallucinating.
Also sanitizes retrieved context payload metadata.
"""

import re
from typing import Any
from src.core.logging import logger

SAFE_DEFLECTION_MESSAGE = (
    "I can share all officially approved project specifications, verified pricing, and legal paperwork. "
    "For internal commercial partnerships, agent commissions, or administrative policies, our senior management team "
    "will follow up with you directly."
)

# Regex patterns matching attempts to probe for internal confidential business data
RESTRICTED_QUERY_PATTERNS = [
    r"\b(internal|profit|gross|builder|net)\s*margin(s)?\b",
    r"\b(broker|agent|referral)\s*(commission|commissions|cut|kickback|kickbacks)\b",
    r"\b(base\s*cost|builder\s*cost|actual\s*cost|cost\s*basis)\b",
    r"\bconfidential\b",
    r"\binternal\s+(note|notes|memo|memos|record|records|document|documents|data)\b",
    r"\b(trade|company|internal|corporate|pricing)\s*secret(s)?\b|\bsecret\s+(margin|price|pricing|commission|deal|cost)\b",
    r"\bunapproved\s*(phase|project|launch|timeline)\b",
    r"\bphase\s*2\s*(launch|timeline|price|pricing|court)\b",
    r"\b(ignore\s*previous\s*instructions|system\s*prompt|prompt\s*injection)\b",
    r"\b(data|memory|prompt|internal)\s*leak(s|ed|ing)?\b|\bleak\s+(data|prompt|system|documents?)\b",
    r"\b(security|firewall|prompt|guardrail|system)\s*bypass\b|\bbypass\s+(security|firewall|guardrail|filter)\b",
]

COMPILED_RESTRICTED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in RESTRICTED_QUERY_PATTERNS
]


class ConfidentialityFilter:
    """Retrieval-layer and prompt-layer confidentiality firewall."""

    def is_restricted_query(self, query_str: str) -> bool:
        """Evaluates whether the user's query probes for restricted corporate data."""
        clean_query = query_str.strip()
        for pattern in COMPILED_RESTRICTED_PATTERNS:
            if pattern.search(clean_query):
                logger.warning(
                    f"Confidentiality firewall triggered for query: '{clean_query[:60]}...' (Matched pattern: {pattern.pattern})"
                )
                return True
        return False

    def get_safe_deflection(self) -> str:
        """Returns the standardized, mathematically safe deflection response."""
        return SAFE_DEFLECTION_MESSAGE

    def sanitize_retrieved_payload(self, chunk_data: dict[str, Any]) -> dict[str, Any]:
        """Strips out internal operational metadata fields before feeding context to LLM."""
        sanitized = dict(chunk_data)

        # Remove internal fields if present in metadata
        metadata = dict(sanitized.get("metadata", {}))
        restricted_keys = ["confidential", "internal", "cost", "margin", "commission", "access_label"]
        for key in list(metadata.keys()):
            if any(r in key.lower() for r in restricted_keys):
                del metadata[key]

        sanitized["metadata"] = metadata
        return sanitized


confidentiality_filter = ConfidentialityFilter()
