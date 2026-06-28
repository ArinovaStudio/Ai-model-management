import json
import logging
from datetime import timezone
from typing import Any, Optional

from app.db.prisma import db

logger = logging.getLogger(__name__)

# Score weights (total = 100)
REQUIREMENT_WEIGHT: float = 30.0
DOCUMENT_WEIGHT: float = 15.0
DEPENDENCY_WEIGHT: float = 15.0
TIMELINE_WEIGHT: float = 25.0
TECH_WEIGHT: float = 15.0


async def run_validation(
    project_id: str,
    reindex: bool = False,
) -> dict[str, Any]:
    project = await db.project.find_unique(
        where={"id": project_id},
        include={
            "projectInfo": True,
            "featureRequests": True,
            "technologies": True,
            "riskBlockages": True,
            "planDocs": True,
            "docs": True,
            "groupingTasks": {"include": {"tasks": True}},
            "resourceAllocations": True,
        },
    )

    if not project:
        raise ValueError(f"Project {project_id} not found")

    if reindex:
        await db.riskblockage.update_many(
            where={"projectId": project_id},
            data={"confidenceScore": 0, "planningWarnings": None},
        )

    features = project.featureRequests or []
    technologies = project.technologies or []
    risk_blockages = project.riskBlockages or []
    plan_docs = project.planDocs or []
    docs = project.docs or []
    groups = project.groupingTasks or []
    allocations = project.resourceAllocations or []
    info = project.projectInfo

    warnings: list[dict[str, str]] = []
    risk_assessments: list[dict[str, Any]] = []

    # 1. Requirement Completeness (30 pts)
    total_features = len(features)
    ambiguous_count = sum(1 for f in features if getattr(f, "isAmbiguous", False))
    clarified_count = sum(1 for f in features if getattr(f, "status", "") == "clarified")
    non_ambiguous = total_features - ambiguous_count

    if total_features == 0:
        req_score = 0.0
        warnings.append({
            "category": "requirements",
            "message": "No feature requirements extracted from project documents",
            "severity": "critical",
        })
    else:
        completeness_ratio = non_ambiguous / total_features
        req_score = REQUIREMENT_WEIGHT * completeness_ratio
        if ambiguous_count > 0:
            pct = ambiguous_count / total_features * 100
            warnings.append({
                "category": "requirements",
                "message": f"{ambiguous_count} of {total_features} requirements ({pct:.0f}%) are ambiguous",
                "severity": "high" if pct > 30 else "medium",
            })
        if clarified_count > 0:
            req_score = min(REQUIREMENT_WEIGHT, req_score + 5.0)

    # 2. Document Coverage (15 pts)
    total_docs = len(plan_docs) + len(docs)
    if total_docs == 0:
        doc_score = 0.0
        warnings.append({
            "category": "documents",
            "message": "No documents have been ingested for this project",
            "severity": "high",
        })
    elif total_docs == 1:
        doc_score = DOCUMENT_WEIGHT * 0.5
    else:
        doc_score = min(DOCUMENT_WEIGHT, DOCUMENT_WEIGHT * (total_docs / 4))

    # 3. Dependency Clarity (15 pts)
    all_tasks: list[Any] = []
    for g in groups:
        all_tasks.extend(getattr(g, "tasks", None) or [])
    tasks_with_deps = sum(1 for t in all_tasks if t.dependencies)
    if not all_tasks:
        dep_score = 0.0
    else:
        dep_ratio = tasks_with_deps / len(all_tasks)
        dep_score = DEPENDENCY_WEIGHT * dep_ratio
        if dep_ratio < 0.5:
            warnings.append({
                "category": "dependencies",
                "message": f"Only {tasks_with_deps} of {len(all_tasks)} tasks ({dep_ratio * 100:.0f}%) have mapped dependencies",
                "severity": "medium",
            })

    # 4. Timeline Feasibility (25 pts)
    timeline_score = 0.0
    if info and info.startDate and info.deadline:
        start = info.startDate
        deadline = info.deadline
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)

        available_days = max(1, (deadline - start).days)
        total_est = sum(g.totalDurationDays or 0 for g in groups)

        if total_est == 0:
            timeline_score = TIMELINE_WEIGHT * 0.3
            warnings.append({
                "category": "timeline",
                "message": "No effort estimates available; timeline feasibility cannot be fully assessed",
                "severity": "medium",
            })
        elif total_est <= available_days:
            timeline_score = TIMELINE_WEIGHT
        elif total_est <= available_days * 1.2:
            timeline_score = TIMELINE_WEIGHT * 0.6
            warnings.append({
                "category": "timeline",
                "message": f"Estimated duration ({total_est:.1f}d) slightly exceeds timeline ({available_days}d)",
                "severity": "medium",
            })
        else:
            timeline_score = TIMELINE_WEIGHT * 0.2
            warnings.append({
                "category": "timeline",
                "message": f"Estimated duration ({total_est:.1f}d) significantly exceeds timeline ({available_days}d)",
                "severity": "critical",
            })
    else:
        timeline_score = 0.0
        warnings.append({
            "category": "timeline",
            "message": "Project start date or deadline not set",
            "severity": "critical",
        })

    # 5. Technology Certainty (15 pts)
    tech_categories = {t.category for t in technologies}
    if len(tech_categories) >= 3:
        tech_score = TECH_WEIGHT
    elif len(tech_categories) == 2:
        tech_score = TECH_WEIGHT * 0.6
    elif len(tech_categories) == 1:
        tech_score = TECH_WEIGHT * 0.3
    else:
        tech_score = 0.0
        warnings.append({
            "category": "technology",
            "message": "No technologies detected in project documents",
            "severity": "high",
        })

    # Assess risks and update RiskBlockage records
    for risk in risk_blockages:
        confidence_impact = 0.0
        r_warning: Optional[str] = None

        if getattr(risk, "severity", "") == "critical":
            confidence_impact = -15.0
            r_warning = f"Critical risk: {risk.title}"
        elif getattr(risk, "severity", "") == "high":
            confidence_impact = -8.0
        elif getattr(risk, "severity", "") == "medium":
            confidence_impact = -3.0

        if getattr(risk, "status", "") == "resolved":
            confidence_impact = 0.0
            r_warning = None

        await db.riskblockage.update(
            where={"id": risk.id},
            data={
                "confidenceScore": confidence_impact,
                "planningWarnings": r_warning,
            },
        )

        risk_assessments.append({
            "risk_id": risk.id,
            "title": risk.title,
            "type": risk.type,
            "severity": risk.severity,
            "status": risk.status,
            "confidence_impact": confidence_impact,
            "warning": r_warning,
        })

    # Resource adequacy check
    if allocations:
        total_available_hours = sum(
            a.allocatedHours for a in allocations
        )
        if total_available_hours > 0:
            recommended_total = sum(
                a.recommendedCount or a.requiredCount for a in allocations
            )
            if recommended_total > 15:
                warnings.append({
                    "category": "resources",
                    "message": f"Team size recommendation ({recommended_total:.0f}) exceeds typical capacity",
                    "severity": "medium",
                })

    # Compute total confidence score
    base_score = req_score + doc_score + dep_score + timeline_score + tech_score

    # Apply risk penalties (each unresolved risk reduces score)
    total_penalty = 0.0
    for ra in risk_assessments:
        if ra["status"] != "resolved":
            total_penalty += abs(ra["confidence_impact"])
    total_penalty = min(40.0, total_penalty)

    confidence = max(0.0, min(100.0, base_score - total_penalty))

    logger.info(
        "Validation complete for project %s: confidence=%.1f, warnings=%d, risks=%d",
        project_id, confidence, len(warnings), len(risk_assessments),
    )

    # Persist to project summary
    project_summary = {}
    if project.summary:
        try:
            project_summary = json.loads(project.summary)
        except (json.JSONDecodeError, TypeError):
            project_summary = {}
    project_summary["validation"] = {
        "confidence_score": round(confidence, 1),
        "components": {
            "requirement_completeness": round(req_score, 1),
            "document_coverage": round(doc_score, 1),
            "dependency_clarity": round(dep_score, 1),
            "timeline_feasibility": round(timeline_score, 1),
            "technology_certainty": round(tech_score, 1),
        },
        "warnings_count": len(warnings),
        "risks_assessed": len(risk_assessments),
    }
    await db.project.update(
        where={"id": project_id},
        data={"summary": json.dumps(project_summary)},
    )

    return {
        "confidence": {
            "score": round(confidence, 1),
            "requirement_completeness": round(req_score, 1),
            "document_coverage": round(doc_score, 1),
            "dependency_clarity": round(dep_score, 1),
            "timeline_feasibility": round(timeline_score, 1),
            "technology_certainty": round(tech_score, 1),
            "max_score": 100.0,
        },
        "warnings": warnings,
        "risk_assessments": risk_assessments,
    }
