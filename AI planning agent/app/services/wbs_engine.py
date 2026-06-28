import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.db.prisma import db

logger = logging.getLogger(__name__)

WBS_TASK_TEMPLATES: list[dict[str, Any]] = [
    {"title": "Research & Analysis", "order": 1, "estimated_hours": 4.0},
    {"title": "Design & Architecture", "order": 2, "estimated_hours": 8.0},
    {"title": "Implementation", "order": 3, "estimated_hours": 16.0},
    {"title": "Testing & QA", "order": 4, "estimated_hours": 6.0},
    {"title": "Documentation", "order": 5, "estimated_hours": 3.0},
    {"title": "Deployment", "order": 6, "estimated_hours": 2.0},
]


def _extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", text.lower())
    stopwords = {
        "the", "and", "for", "this", "that", "with", "from", "have",
        "will", "should", "shall", "must", "can", "may", "need",
        "feature", "system", "user", "application", "project", "support",
        "also", "each", "their", "them", "they", "its", "would", "could",
    }
    return {w for w in words if w not in stopwords}


def _compute_due_date(
    project_deadline: Optional[datetime],
    task_order: int,
    total_tasks: int,
) -> datetime:
    now = datetime.now(timezone.utc)
    if project_deadline:
        if project_deadline.tzinfo is None:
            project_deadline = project_deadline.replace(tzinfo=timezone.utc)
        span = (project_deadline - now).total_seconds()
        if span > 0:
            offset = span * (task_order / (total_tasks + 1))
            return now + timedelta(seconds=offset)
    return now + timedelta(days=task_order * 5)


async def generate_wbs(
    project_id: str,
    reindex: bool = False,
) -> dict[str, Any]:
    project = await db.project.find_unique(
        where={"id": project_id},
        include={
            "featureRequests": True,
            "projectInfo": True,
        },
    )

    if not project:
        raise ValueError(f"Project {project_id} not found")

    if reindex:
        await db.task.delete_many(
            where={"projectId": project_id, "featureRequestId": {"not": None}},
        )
        await db.groupingtask.delete_many(where={"projectId": project_id})

    features = project.featureRequests or []
    if not features:
        logger.info("No feature requests found for project %s; WBS will be empty", project_id)
        return {"groups": [], "total_groups": 0, "total_tasks": 0}

    deadline = project.projectInfo.deadline if project.projectInfo else None

    groups_map: dict[str, list[Any]] = {}
    for f in features:
        category = f.category or "uncategorized"
        groups_map.setdefault(category, []).append(f)

    stored_groups: list[Any] = []
    global_task_index = 0
    all_stored_tasks: list[Any] = []
    group_task_map: dict[str, list[Any]] = {}

    for group_order, (category, cat_features) in enumerate(sorted(groups_map.items())):
        group_name = category.replace("_", " ").strip().title() + " Features"
        feature_ids = [f.id for f in cat_features]

        group_record = await db.groupingtask.create(
            data={
                "name": group_name,
                "projectId": project_id,
                "description": f"Work package for {category} features ({len(cat_features)} features)",
                "order": group_order,
                "featureIds": json.dumps(feature_ids),
            },
        )

        group_tasks: list[Any] = []
        feature_keywords: dict[str, set[str]] = {}

        for feature in cat_features:
            feat_keywords = _extract_keywords(f"{feature.title} {feature.description or ''}")
            feature_keywords[feature.id] = feat_keywords

            for template in WBS_TASK_TEMPLATES:
                global_task_index += 1
                task_title = f"{template['title']} - {feature.title}"
                task_description = (
                    f"{template['title']} for feature '{feature.title}': "
                    f"{feature.description[:250] if feature.description else 'No description provided'}"
                )

                due = _compute_due_date(deadline, template["order"], len(WBS_TASK_TEMPLATES))

                task_record = await db.task.create(
                    data={
                        "title": task_title,
                        "description": task_description,
                        "assignee": "unassigned",
                        "priority": feature.priority or "medium",
                        "dueDate": due,
                        "status": "pending",
                        "tags": [],
                        "projectId": project_id,
                        "featureRequestId": feature.id,
                        "groupId": group_record.id,
                        "order": template["order"],
                        "estimatedHours": template["estimated_hours"],
                    },
                )
                group_tasks.append(task_record)
                all_stored_tasks.append(task_record)

        # Set within-feature sequential dependencies
        for feature in cat_features:
            feature_tasks = [t for t in group_tasks if t.featureRequestId == feature.id]
            feature_tasks.sort(key=lambda t: t.order)
            for i, task in enumerate(feature_tasks):
                if i > 0:
                    deps = json.dumps([feature_tasks[i - 1].id])
                    await db.task.update(
                        where={"id": task.id},
                        data={"dependencies": deps},
                    )

        # Cross-feature keyword-based dependencies
        for feature in cat_features:
            feature_tasks = [t for t in group_tasks if t.featureRequestId == feature.id]
            feature_tasks.sort(key=lambda t: t.order)
            if not feature_tasks:
                continue
            first_task = feature_tasks[0]

            for other_feat in cat_features:
                if other_feat.id == feature.id:
                    continue
                other_keywords = feature_keywords.get(other_feat.id, set())
                if other_keywords and feature_keywords.get(feature.id, set()) & other_keywords:
                    other_tasks = [t for t in group_tasks if t.featureRequestId == other_feat.id]
                    other_tasks.sort(key=lambda t: t.order, reverse=True)
                    if other_tasks:
                        dep_id = other_tasks[0].id
                        existing_deps: list[str] = []
                        if first_task.dependencies:
                            try:
                                existing_deps = json.loads(first_task.dependencies)
                            except (json.JSONDecodeError, TypeError):
                                existing_deps = []
                        if dep_id not in existing_deps and first_task.id != dep_id:
                            existing_deps.append(dep_id)
                            await db.task.update(
                                where={"id": first_task.id},
                                data={"dependencies": json.dumps(existing_deps)},
                            )

        group_record_with_tasks = await db.groupingtask.find_unique(
            where={"id": group_record.id},
            include={"tasks": True},
        )
        stored_groups.append(group_record_with_tasks)
        group_task_map[group_record.id] = group_tasks

    logger.info(
        "WBS generated for project %s: %d groups, %d tasks",
        project_id,
        len(stored_groups),
        len(all_stored_tasks),
    )

    return {
        "groups": stored_groups,
        "total_groups": len(stored_groups),
        "total_tasks": len(all_stored_tasks),
    }
