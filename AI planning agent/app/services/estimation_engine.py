import json
import logging
import re
from typing import Any, Optional

from app.db.prisma import db

logger = logging.getLogger(__name__)

TASK_SIZE_FACTORS: dict[str, float] = {
    "Research & Analysis": 0.3,
    "Design & Architecture": 0.6,
    "Implementation": 1.0,
    "Testing & QA": 0.5,
    "Documentation": 0.2,
    "Deployment": 0.3,
}

HIGH_UNCERTAINTY_KEYWORDS: list[str] = [
    "ai", "machine learning", "deep learning", "nlp", "neural",
    "computer vision", "blockchain", "real.time", "distributed",
    "microservices", "kubernetes", "scalable", "high.availability",
    "complex", "predictive", "recommendation", "optimization",
    "cryptography", "zero.downtime", "fault.tolerant",
]

MEDIUM_UNCERTAINTY_KEYWORDS: list[str] = [
    "api", "integration", "third.party", "external", "migration",
    "authentication", "authorization", "database", "cache",
    "websocket", "webhook", "rest", "graphql", "async",
    "background job", "queue", "event.driven", "cron",
]

INTEGRATION_KEYWORDS: list[str] = [
    "integrate", "integration", "connect", "interface",
    "webhook", "rest", "graphql", "api", "sync",
    "federation", "sso", "oauth", "ldap",
]

HOURS_PER_DAY: float = 8.0
MAX_COMPLEXITY: float = 5.0
COMPLEXITY_WEIGHT: float = 0.25


def _compute_task_complexity(
    title: str,
    description: str,
    dependency_count: int,
) -> float:
    combined = f"{title} {description}".lower()

    base_score = 0.5
    for task_type, factor in TASK_SIZE_FACTORS.items():
        if task_type.lower() in title.lower():
            base_score = factor
            break

    high_count = 0
    for kw in HIGH_UNCERTAINTY_KEYWORDS:
        pattern = kw.replace(".", r"[.\s-]")
        if re.search(pattern, combined):
            high_count += 1

    med_count = 0
    for kw in MEDIUM_UNCERTAINTY_KEYWORDS:
        pattern = kw.replace(".", r"[.\s-]")
        if re.search(pattern, combined):
            med_count += 1

    integration_count = 0
    for kw in INTEGRATION_KEYWORDS:
        pattern = kw.replace(".", r"[.\s-]")
        if re.search(pattern, combined):
            integration_count += 1

    uncertainty_bonus = (high_count * 0.3) + (med_count * 0.15)
    integration_bonus = integration_count * 0.2
    dependency_impact = dependency_count * 0.1

    score = base_score + uncertainty_bonus + integration_bonus + dependency_impact
    return round(min(MAX_COMPLEXITY, score), 2)


def _compute_effort(estimated_hours: float, complexity_score: float) -> tuple[float, float]:
    effort = estimated_hours * (1.0 + complexity_score * COMPLEXITY_WEIGHT)
    duration = max(0.5, effort / HOURS_PER_DAY)
    return round(effort, 2), round(duration, 2)


async def run_estimation(
    project_id: str,
    reindex: bool = False,
) -> dict[str, Any]:
    project = await db.project.find_unique(
        where={"id": project_id},
        include={
            "groupingTasks": {"include": {"tasks": True}},
        },
    )

    if not project:
        raise ValueError(f"Project {project_id} not found")

    groups = project.groupingTasks or []
    if not groups:
        logger.info("No WBS groups found for project %s; estimation skipped", project_id)
        return {"groups": [], "summary": {}}

    all_group_data: list[dict[str, Any]] = []

    for group in groups:
        tasks = getattr(group, "tasks", None) or []
        group_task_data: list[dict[str, Any]] = []

        for task in tasks:
            deps: list[str] = []
            if task.dependencies:
                try:
                    deps = json.loads(task.dependencies)
                except (json.JSONDecodeError, TypeError):
                    deps = []

            estimated_hours = task.estimatedHours or 4.0
            complexity = _compute_task_complexity(
                title=task.title,
                description=task.description,
                dependency_count=len(deps),
            )
            effort, duration = _compute_effort(estimated_hours, complexity)

            await db.task.update(
                where={"id": task.id},
                data={
                    "complexityScore": complexity,
                    "effortHours": effort,
                    "durationDays": duration,
                },
            )

            group_task_data.append({
                "task_id": task.id,
                "complexity_score": complexity,
                "effort_hours": effort,
                "duration_days": duration,
            })

        total_effort = sum(t["effort_hours"] for t in group_task_data)
        total_duration = sum(t["duration_days"] for t in group_task_data)
        avg_complexity = (
            sum(t["complexity_score"] for t in group_task_data) / len(group_task_data)
            if group_task_data else 0.0
        )

        await db.groupingtask.update(
            where={"id": group.id},
            data={
                "totalEffortHours": round(total_effort, 2),
                "totalDurationDays": round(total_duration, 2),
                "avgComplexityScore": round(avg_complexity, 2),
            },
        )

        all_group_data.append({
            "group_id": group.id,
            "group_name": group.name or "",
            "total_effort_hours": total_effort,
            "total_duration_days": total_duration,
            "avg_complexity_score": avg_complexity,
            "task_count": len(group_task_data),
            "tasks": group_task_data,
        })

    total_effort = sum(g["total_effort_hours"] for g in all_group_data)
    total_duration = sum(g["total_duration_days"] for g in all_group_data)
    all_scores = [
        t["complexity_score"]
        for g in all_group_data
        for t in g["tasks"]
    ]
    avg_complexity = sum(all_scores) / len(all_scores) if all_scores else 0.0

    summary = {
        "total_effort_hours": round(total_effort, 2),
        "total_duration_days": round(total_duration, 2),
        "avg_complexity_score": round(avg_complexity, 2),
        "total_tasks": len(all_scores),
        "total_groups": len(all_group_data),
    }

    project_summary = {}
    if project.summary:
        try:
            project_summary = json.loads(project.summary)
        except (json.JSONDecodeError, TypeError):
            project_summary = {"raw": project.summary}
    project_summary["effort_estimation"] = summary
    await db.project.update(
        where={"id": project_id},
        data={"summary": json.dumps(project_summary)},
    )

    logger.info(
        "Estimation complete for project %s: %.1f total hours, %.1f days",
        project_id, total_effort, total_duration,
    )

    return {"groups": all_group_data, "summary": summary}
