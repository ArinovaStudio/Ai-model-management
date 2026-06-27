import re
import logging
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

CONSTRAINT_PATTERNS: list[str] = [
    r"(?:must|shall|should|will)\s+(?:be\s+)?(?:not\s+)?(?:able|required|completed|delivered|approved|reviewed|tested|deployed)",
    r"(?:cannot|can't|must not|mustn't|shall not|shan't|should not|shouldn't)",
    r"(?:limited|restricted|constrained)\s+(?:to|by)",
    r"(?:requires?|required|mandatory|compulsory|prerequisite)",
    r"(?:strict|hard|firm)\s+(?:deadline|limit|constraint|requirement)",
    r"(?:budget|cost|time|resource)\s+(?:limit|constraint|restriction|bound)",
    r"(?:within\s+\d+\s+(?:days?|weeks?|months?|years?))",
    r"(?:no\s+(?:later|earlier|more|less|fewer)\s+than)",
    r"(?:must\s+(?:comply|adhere|follow|meet|satisfy))",
    r"(?:outside\s+(?:the\s+)?(?:scope|budget|timeline|schedule))",
]

SCOPE_PATTERNS: list[str] = [
    r"(?:scope|objective|goal|purpose|mission)\s*(?::|is|includes?|covers?|encompasses?)",
    r"(?:deliverable|artifact|output|outcome|result)",
    r"(?:feature|functionality|capability|module|component|subsystem)",
    r"(?:this\s+(?:project|system|application|solution|platform))\s+(?:will|shall|should|aims?|intends?)",
    r"(?:key\s+(?:area|component|feature|functionality|deliverable|objective))",
    r"(?:responsible\s+(?:for|to))",
    r"(?:in\s+scope|out\s+of\s+scope|within\s+scope)",
    r"(?:the\s+(?:project|team|system|solution))\s+(?:will\s+deliver|provides?|offers?|handles?|manages?|processes?|supports?)",
    r"(?:technical\s+(?:requirement|specification|approach|stack|architecture|design))",
    r"(?:user\s+(?:story|requirement|need|expectation|workflow))",
]

MILESTONE_PATTERNS: list[str] = [
    r"(?:milestone|phase|sprint|release|iteration)\s*:?\s*([^.]*?(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})[^.]*?\.)",
    r"(?:by|before|after|target|due)\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s*:?\s*([^.]*?\.)",
]


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _extract_between(text: str, patterns: list[str]) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    for sentence in _split_sentences(text):
        for pattern in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                key = sentence.lower().strip()
                if key not in seen:
                    seen.add(key)
                    results.append(sentence.strip())
                break
    return results


def extract_timeline(
    start_date: date,
    deadline: date,
    texts: Optional[list[str]] = None,
) -> dict:
    duration_days = (deadline - start_date).days
    result: dict = {
        "start_date": start_date.isoformat(),
        "deadline": deadline.isoformat(),
        "duration_days": max(duration_days, 0),
    }

    if texts:
        milestones: list[dict] = []
        for text in texts:
            for pattern in MILESTONE_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    milestones.append({"description": match.group(0).strip()})
        if milestones:
            result["milestones"] = milestones

    return result


def extract_constraints(texts: list[str]) -> list[str]:
    constraints: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for sentence in _split_sentences(text):
            for pattern in CONSTRAINT_PATTERNS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    key = sentence.lower().strip()
                    if key not in seen:
                        seen.add(key)
                        constraints.append(sentence.strip())
                    break
    return constraints


def extract_scope(texts: list[str]) -> list[str]:
    scope_items: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for sentence in _split_sentences(text):
            for pattern in SCOPE_PATTERNS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    key = sentence.lower().strip()
                    if key not in seen:
                        seen.add(key)
                        scope_items.append(sentence.strip())
                    break
    return scope_items


def build_summary_text(
    project_name: str,
    description: Optional[str],
    timeline: dict,
    scope_items: list[str],
    constraints: list[str],
    plan_docs_count: int,
    docs_count: int,
) -> str:
    parts: list[str] = [f"Project: {project_name}"]

    if description:
        parts.append(f"Description: {description}")

    if timeline:
        parts.append(
            f"Timeline: {timeline.get('start_date', 'N/A')} to "
            f"{timeline.get('deadline', 'N/A')} "
            f"({timeline.get('duration_days', 'unknown')} days)"
        )

    if scope_items:
        parts.append("Scope: " + " | ".join(scope_items[:5]))

    if constraints:
        parts.append("Constraints: " + " | ".join(constraints[:5]))

    parts.append(f"Documents: {plan_docs_count} plan docs, {docs_count} general docs")

    return " | ".join(parts)
