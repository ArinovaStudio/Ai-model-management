import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

from app.db.prisma import db

logger = logging.getLogger(__name__)

ROLES: list[str] = [
    "Backend",
    "Frontend",
    "Mobile",
    "QA",
    "DevOps",
    "UI/UX",
]

ROLE_KEYWORD_MAP: dict[str, list[str]] = {
    "Backend": [
        "backend", "server", "api", "database", "python", "java", "go",
        "rust", "node", "express", "django", "flask", "spring", "sql",
        "postgresql", "mongodb", "redis", "graphql", "rest", "microservice",
        "authentication", "authorization", "caching", "queue", "cron",
        "implementation", "business logic", "data model", "migration",
    ],
    "Frontend": [
        "frontend", "ui", "react", "vue", "angular", "svelte", "next.js",
        "typescript", "javascript", "css", "html", "tailwind", "bootstrap",
        "component", "redux", "state management", "webpack", "vite",
        "responsive", "design system", "material ui",
    ],
    "Mobile": [
        "mobile", "react native", "flutter", "swift", "kotlin", "android",
        "ios", "app store", "play store", "mobile ui",
    ],
    "QA": [
        "testing", "test", "qa", "quality", "jest", "pytest", "mocha",
        "cypress", "selenium", "playwright", "junit", "eslint",
        "integration test", "unit test", "e2e", "coverage",
    ],
    "DevOps": [
        "devops", "deployment", "ci/cd", "docker", "kubernetes", "jenkins",
        "github actions", "gitlab ci", "terraform", "ansible", "helm",
        "monitoring", "prometheus", "grafana", "infrastructure", "pipeline",
        "release", "production", "staging", "load balancer",
    ],
    "UI/UX": [
        "design", "research", "analysis", "documentation", "figma",
        "sketch", "adobe xd", "wireframe", "prototype", "user research",
        "usability", "accessibility", "storybook", "branding",
    ],
}

HOURS_PER_PERSON_PER_DAY: float = 6.5
MIN_TEAM_SIZE: int = 1
MAX_TEAM_SIZE: int = 15


def _determine_roles(title: str, description: str) -> list[str]:
    combined = f"{title} {description}".lower()
    matched: list[tuple[str, int]] = []

    for role, keywords in ROLE_KEYWORD_MAP.items():
        score = 0
        for kw in keywords:
            pattern = re.escape(kw)
            matches = re.findall(pattern, combined)
            score += len(matches)
        if score > 0:
            matched.append((role, score))

    matched.sort(key=lambda x: x[1], reverse=True)

    if not matched:
        if any(w in combined for w in ["research", "analysis", "documentation"]):
            return ["UI/UX"]
        return ["Backend"]

    top_score = matched[0][1]
    result = [matched[0][0]]
    for role, score in matched[1:]:
        if score >= top_score * 0.6:
            result.append(role)

    return result[:2]


async def run_resource_planning(
    project_id: str,
    reindex: bool = False,
) -> dict[str, Any]:
    project = await db.project.find_unique(
        where={"id": project_id},
        include={
            "groupingTasks": {"include": {"tasks": True}},
            "projectInfo": True,
        },
    )

    if not project:
        raise ValueError(f"Project {project_id} not found")

    if reindex:
        await db.resourceallocation.delete_many(where={"projectId": project_id})

    groups = project.groupingTasks or []
    tasks: list[Any] = []
    for g in groups:
        tasks.extend(getattr(g, "tasks", None) or [])

    if not tasks:
        logger.info("No tasks found for project %s; resource planning skipped", project_id)
        return {"allocations": [], "total_team_size": 0, "total_effort_hours": 0}

    role_hours: dict[str, float] = {r: 0.0 for r in ROLES}

    for task in tasks:
        effort = task.effortHours or task.estimatedHours or 0
        if effort <= 0:
            continue
        matched_roles = _determine_roles(task.title, task.description or "")
        split = effort / len(matched_roles)
        for role in matched_roles:
            role_hours[role] = role_hours.get(role, 0) + split

    deadline = project.projectInfo.deadline if project.projectInfo else None
    total_duration_days = 0.0
    if deadline:
        now = datetime.now(timezone.utc)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        remaining = (deadline - now).total_seconds()
        total_duration_days = max(5.0, remaining / 86400.0)
    else:
        max_duration = 0.0
        for g in groups:
            if g.totalDurationDays and g.totalDurationDays > max_duration:
                max_duration = g.totalDurationDays
        total_duration_days = max(10.0, max_duration)

    total_effort = sum(role_hours.values())
    available_per_person = HOURS_PER_PERSON_PER_DAY * total_duration_days

    stored_allocations: list[Any] = []
    for role in ROLES:
        hours = round(role_hours.get(role, 0), 2)
        if hours <= 0:
            continue

        raw_count = hours / available_per_person if available_per_person > 0 else 1
        recommended = max(MIN_TEAM_SIZE, round(raw_count, 1))
        required = max(MIN_TEAM_SIZE, min(MAX_TEAM_SIZE, math.ceil(raw_count)))
        workload_pct = round((hours / total_effort * 100) if total_effort > 0 else 0, 1)

        record = await db.resourceallocation.create(
            data={
                "projectId": project_id,
                "role": role,
                "requiredCount": required,
                "allocatedHours": hours,
                "workloadPercentage": workload_pct,
                "recommendedCount": recommended,
                "metadata": json.dumps({
                    "total_duration_days": round(total_duration_days, 1),
                    "available_hours_per_person": round(available_per_person, 1),
                }),
            },
        )
        stored_allocations.append(record)

    if not stored_allocations:
        default_role = ROLES[0]
        record = await db.resourceallocation.create(
            data={
                "projectId": project_id,
                "role": default_role,
                "requiredCount": 1,
                "allocatedHours": 0,
                "workloadPercentage": 100,
                "recommendedCount": 1,
                "metadata": json.dumps({"note": "default allocation - no tasks with effort data"}),
            },
        )
        stored_allocations.append(record)

    total_team_size = sum(a.requiredCount for a in stored_allocations)

    logger.info(
        "Resource plan generated for project %s: %d roles, %d team members",
        project_id, len(stored_allocations), total_team_size,
    )

    project_summary = {}
    if project.summary:
        try:
            project_summary = json.loads(project.summary)
        except (json.JSONDecodeError, TypeError):
            project_summary = {}
    project_summary["resource_plan"] = {
        "total_team_size": total_team_size,
        "total_effort_hours": round(total_effort, 2),
        "total_duration_days": round(total_duration_days, 1),
        "roles": [
            {
                "role": a.role,
                "required_count": a.requiredCount,
                "allocated_hours": a.allocatedHours,
                "workload_percentage": a.workloadPercentage,
            }
            for a in stored_allocations
        ],
    }
    await db.project.update(
        where={"id": project_id},
        data={"summary": json.dumps(project_summary)},
    )

    return {
        "allocations": stored_allocations,
        "total_team_size": total_team_size,
        "total_effort_hours": round(total_effort, 2),
    }
