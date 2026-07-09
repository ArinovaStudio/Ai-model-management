import json
import logging
import asyncio
from datetime import date
from typing import Any, Optional

from app.db.prisma import db
from app.services.ollama_service import ollama_service
from app.services.tech_stack import TECHNOLOGY_REGISTRY

logger = logging.getLogger(__name__)


class ProjectNotFoundError(Exception):
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(f"Project {project_id} not found")


def _build_project_context(
    project_name: str,
    description: Optional[str],
    scope: Optional[str],
    constraints: Optional[str],
    start_date: Optional[date],
    deadline: Optional[date],
    features: list[Any],
    technologies: list[Any],
    plan_docs: list[Any],
    docs: list[Any],
) -> str:
    parts: list[str] = [f"Project: {project_name}"]

    if description:
        parts.append(f"Description: {description}")
    if scope:
        parts.append(f"Scope: {scope}")
    if constraints:
        parts.append(f"Constraints: {constraints}")
    if start_date and deadline:
        duration = (deadline - start_date).days
        parts.append(f"Timeline: {start_date.isoformat()} to {deadline.isoformat()} ({duration} days)")

    if features:
        parts.append("\nExtracted Feature Requirements:")
        for f in features:
            ambig = " [AMBIGUOUS]" if getattr(f, "isAmbiguous", False) else ""
            parts.append(f"  - {f.title}{ambig}: {f.description[:200]}")
    else:
        parts.append("\nNo feature requirements extracted.")

    if technologies:
        parts.append("\nDetected Technology Stack:")
        by_cat: dict[str, list[str]] = {}
        for t in technologies:
            by_cat.setdefault(t.category, []).append(t.name)
        for cat, names in sorted(by_cat.items()):
            parts.append(f"  {cat}: {', '.join(names)}")
    else:
        parts.append("\nNo technologies detected.")

    if plan_docs:
        parts.append(f"\nPlan Documents ({len(plan_docs)}):")
        for d in plan_docs:
            parts.append(f"  - {d.title}: {d.content[:200]}")
    if docs:
        parts.append(f"\nSupporting Documents ({len(docs)}):")
        for d in docs[:5]:
            parts.append(f"  - {d.title}: {d.content[:200]}")

    return "\n".join(parts)


def _keyword_gap_analysis(
    features: list[Any],
    technologies: list[Any],
    description: Optional[str],
    scope: Optional[str],
    constraints: Optional[str],
    start_date: Optional[date],
    deadline: Optional[date],
) -> dict[str, list[dict]]:
    from app.services.extraction import _split_sentences

    result: dict[str, list[dict]] = {
        "assumptions": [],
        "clarifications": [],
        "risks": [],
    }
    seen: set[str] = set()

    def add_item(section: str, item: dict) -> None:
        key = item["title"].lower().strip()
        if key not in seen:
            seen.add(key)
            result.setdefault(section, []).append(item)

    ambig_features = [f for f in features if getattr(f, "isAmbiguous", False)]
    for f in ambig_features:
        add_item("clarifications", {
            "title": f"Ambiguous requirement: {f.title}",
            "description": f.clarificationPrompt or f"Requirement '{f.title}' is ambiguous and needs clarification.",
            "category": "ambiguity",
            "severity": "high",
            "related_feature": f.title,
        })

    tech_names = {t.name.lower(): t for t in technologies}
    for t in technologies:
        deps_raw = getattr(t, "dependencies", None)
        if deps_raw:
            try:
                deps = json.loads(deps_raw) if isinstance(deps_raw, str) else deps_raw
            except (json.JSONDecodeError, TypeError):
                deps = []
            for dep in deps:
                if dep.lower() not in tech_names:
                    add_item("risks", {
                        "title": f"Missing dependency: {dep} for {t.name}",
                        "description": f"{t.name} depends on {dep}, but {dep} was not detected in the project documents.",
                        "category": "missing_dependency",
                        "severity": "high",
                        "type": "blockage",
                    })

    tech_categories = {t.category for t in technologies}
    expected_categories = {"backend", "frontend", "devops"}
    missing_cats = expected_categories - tech_categories
    for cat in missing_cats:
        add_item("assumptions", {
            "title": f"No {cat} technology detected",
            "description": f"No {cat} technologies were explicitly mentioned. It may be assumed that {cat} is out of scope or handled by an existing platform.",
            "category": "technology",
            "severity": "medium",
        })

    if start_date and deadline:
        duration = (deadline - start_date).days
        if duration <= 0:
            add_item("risks", {
                "title": "Deadline is before or same as start date",
                "description": f"Project deadline ({deadline.isoformat()}) is not after start date ({start_date.isoformat()}).",
                "category": "unrealistic_deadline",
                "severity": "critical",
                "type": "blockage",
            })
        elif duration < 14:
            add_item("risks", {
                "title": "Very short project timeline",
                "description": f"Project duration is only {duration} days, which may be unrealistic for the scope.",
                "category": "unrealistic_deadline",
                "severity": "high",
                "type": "risk",
            })
        elif duration < 30:
            add_item("risks", {
                "title": "Short project timeline",
                "description": f"Project duration is {duration} days. Consider if this is sufficient.",
                "category": "unrealistic_deadline",
                "severity": "medium",
                "type": "risk",
            })

    if not description and not scope:
        add_item("clarifications", {
            "title": "No project description or scope provided",
            "description": "The project lacks a description and scope statement. This is needed to evaluate completeness.",
            "category": "missing_detail",
            "severity": "high",
            "related_feature": None,
        })

    if not constraints:
        add_item("assumptions", {
            "title": "No constraints documented",
            "description": "No project constraints were provided. It is assumed there are no budget, timeline, or resource limitations.",
            "category": "scope",
            "severity": "medium",
        })

    if not features:
        add_item("risks", {
            "title": "No feature requirements extracted",
            "description": "No functional requirements or features could be extracted from the project documents.",
            "category": "requirement_gap",
            "severity": "critical",
            "type": "blockage",
        })

    if ambig_features:
        total = len(features)
        ratio = len(ambig_features) / total if total > 0 else 1.0
        if ratio > 0.5:
            add_item("risks", {
                "title": "Majority of requirements are ambiguous",
                "description": f"{len(ambig_features)} of {total} extracted requirements ({ratio:.0%}) are marked ambiguous.",
                "category": "requirement_gap",
                "severity": "high",
                "type": "blockage",
            })

    return result


async def perform_gap_analysis(
    project_id: str,
    reindex: bool = False,
) -> dict[str, Any]:
    project = await db.project.find_unique(
        where={"id": project_id},
        include={
            "projectInfo": True,
            "planDocs": True,
            "docs": True,
            "featureRequests": True,
            "technologies": True,
        },
    )

    if not project:
        raise ProjectNotFoundError(project_id)

    if reindex:
        await db.riskblockage.delete_many(where={"projectId": project_id})

    project_info = project.projectInfo
    features = project.featureRequests or []
    technologies = project.technologies or []
    plan_docs = project.planDocs or []
    docs = project.docs or []

    context = _build_project_context(
        project_name=project.name,
        description=project_info.description if project_info else None,
        scope=project_info.scope if project_info else None,
        constraints=project_info.constraints if project_info else None,
        start_date=project_info.startDate if project_info else None,
        deadline=project_info.deadline if project_info else None,
        features=features,
        technologies=technologies,
        plan_docs=plan_docs,
        docs=docs,
    )

    ollama_result = await ollama_service.analyze_gaps(context)

    ollama_assumptions: list[dict] = ollama_result.get("assumptions", []) if ollama_result else []
    ollama_clarifications: list[dict] = ollama_result.get("clarifications", []) if ollama_result else []
    ollama_risks: list[dict] = ollama_result.get("risks", []) if ollama_result else []

    has_llm = bool(ollama_assumptions or ollama_clarifications or ollama_risks)

    if not has_llm:
        logger.info("Ollama gap analysis returned empty; using keyword fallback")
        keyword_result = _keyword_gap_analysis(
            features=features,
            technologies=technologies,
            description=project_info.description if project_info else None,
            scope=project_info.scope if project_info else None,
            constraints=project_info.constraints if project_info else None,
            start_date=project_info.startDate if project_info else None,
            deadline=project_info.deadline if project_info else None,
        )
        ollama_assumptions = keyword_result.get("assumptions", [])
        ollama_clarifications = keyword_result.get("clarifications", [])
        ollama_risks = keyword_result.get("risks", [])
        source = "keyword"
    else:
        source = "ollama"

    stored_assumptions: list[Any] = []
    stored_clarifications: list[Any] = []
    stored_risks: list[Any] = []

    async def store(section: str, items: list[dict], report_type: str) -> list[Any]:
        records: list[Any] = []
        creation_calls = []
        for item in items:
            title = (item.get("title") or "").strip()
            desc = (item.get("description") or "").strip()
            if not title or not desc:
                continue
            feat_title = item.get("related_feature")
            linked_feature = None
            if feat_title:
                linked_feature = next(
                    (f for f in features if f.title.lower() == feat_title.lower()),
                    None,
                )

            cat = item.get("category", "requirement_gap") if section == "risks" else item.get("category", "scope")
            if section == "risks":
                item_type = item.get("type", "risk")
            elif section == "clarifications":
                item_type = item.get("type", "clarification")
            else:
                item_type = item.get("type", "assumption")

            async def create_rb(t=title, d=desc, i_type=item_type, c=cat, lf=linked_feature, ft=feat_title):
                rec = await db.riskblockage.create(
                    data={
                        "projectId": project_id,
                        "clientId": "system",
                        "riskTitle": t,
                        "type": i_type,
                        "category": c,
                        "title": t,
                        "description": d,
                        "severity": item.get("severity", "medium"),
                        "status": "open",
                        "sourceFeatureId": lf.id if lf else None,
                        "reportType": report_type,
                        "metadata": json.dumps({
                            "source": source,
                            "llm_generated": has_llm,
                            "related_feature": ft,
                        }),
                    }
                )
                records.append(rec)
            creation_calls.append(create_rb())

        if creation_calls:
            await asyncio.gather(*creation_calls)
        return records

    stored_assumptions = await store("assumptions", ollama_assumptions, "assumption")
    stored_clarifications = await store("clarifications", ollama_clarifications, "clarification")
    stored_risks = await store("risks", ollama_risks, "risk")

    total_items = len(stored_assumptions) + len(stored_clarifications) + len(stored_risks)
    critical_items = sum(1 for r in stored_risks if r.severity == "critical")
    high_items = sum(1 for r in stored_risks if r.severity == "high")

    summary_parts: list[str] = []
    if total_items == 0:
        summary_parts.append("No gaps, risks, or assumptions identified.")
    else:
        summary_parts.append(f"Identified {total_items} items:")
        summary_parts.append(f"  - {len(stored_assumptions)} assumptions")
        summary_parts.append(f"  - {len(stored_clarifications)} clarifications needed")
        summary_parts.append(f"  - {len(stored_risks)} risks/blockages ({critical_items} critical, {high_items} high)")

    summary = " | ".join(summary_parts)
    logger.info("Gap analysis for project %s: %s", project_id, summary)

    return {
        "assumptions": stored_assumptions,
        "clarifications": stored_clarifications,
        "risks": stored_risks,
        "summary": summary,
    }
