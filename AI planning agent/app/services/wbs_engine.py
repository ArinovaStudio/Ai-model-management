import json
import logging
import re
import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.db.prisma import db, get_cached_project, set_cached_project, clear_project_cache
from app.services.ollama_service import ollama_service
from app.services.estimation_engine import _compute_task_complexity, _compute_effort

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
    sprint_duration_days: int = 14,
    productive_hours_per_day: float = 6.0,
) -> dict[str, Any]:
    if reindex:
        clear_project_cache(project_id)
        await db.task.delete_many(
            where={"projectId": project_id, "featureRequestId": {"not": None}},
        )
        await db.groupingtask.delete_many(where={"projectId": project_id})

    project = get_cached_project(project_id)
    if not project or not getattr(project, "featureRequests", None):
        project = await db.project.find_unique(
            where={"id": project_id},
            include={
                "featureRequests": True,
                "projectInfo": True,
                "technologies": True,
                "projectIntelligence": True,
                "groupingTasks": {"include": {"tasks": True}},
                "resourceAllocations": True,
            },
        )
        if project:
            set_cached_project(project_id, project)

    if not project:
        raise ValueError(f"Project {project_id} not found")

    features = project.featureRequests or []
    if not features:
        logger.info("No feature requests found for project %s; WBS will be empty", project_id)
        return {"groups": [], "total_groups": 0, "total_tasks": 0}

    deadline = project.projectInfo.deadline if project.projectInfo else None

    # Build context for LLM
    info = project.projectInfo
    context = ""
    tech_stack = ", ".join([t.name for t in (project.technologies or [])]) if project.technologies else "None specified"
    
    if info:
        desc = info.description or ""
        scope = info.scope or ""
        constraints = info.constraints or ""
        context = f"Project: {info.name}\nDescription: {desc}\nScope: {scope}\nConstraints: {constraints}\nTechnology Stack: {tech_stack}\n"

    if getattr(project, "projectIntelligence", None):
        pi = project.projectIntelligence
        context += "\n--- ARCHITECTURAL INTELLIGENCE ---\n"
        if pi.systemArchitecture:
            arch = pi.systemArchitecture
            if isinstance(arch, dict):
                high_level = arch.get("highLevel") or arch.get("description") or ""
                context += f"System Architecture Overview: {high_level}\n"
            elif isinstance(arch, str):
                try:
                    arch_dict = json.loads(arch)
                    high_level = arch_dict.get("highLevel") or arch_dict.get("description") or ""
                    context += f"System Architecture Overview: {high_level}\n"
                except Exception:
                    context += f"System Architecture: {arch[:1000]}\n"

    stored_groups: list[Any] = []
    global_task_index = 0
    
    # Pre-extract keywords for all features for dependency mapping
    feature_keywords: dict[str, set[str]] = {}
    for f in features:
        feature_keywords[f.id] = _extract_keywords(f"{f.title} {f.description or ''}")

    # --- PARALLEL WBS LLM EXTRACTION WORKERS ---
    sem = asyncio.Semaphore(4)

    async def get_feature_wbs(feature: Any) -> tuple[Any, Optional[dict[str, Any]], bool]:
        async with sem:
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    wbs_data = await ollama_service.generate_combined_wbs(
                        context=context,
                        feature=f"{feature.title}: {feature.description}"
                    )
                    if wbs_data and "milestones" in wbs_data and wbs_data["milestones"]:
                        return feature, wbs_data, False
                    
                    logger.warning(
                        "WBS structure invalid on attempt %d for feature %s, retrying...",
                        attempt + 1, feature.id
                    )
                except Exception as e:
                    logger.warning(
                        "Failed WBS attempt %d for feature %s: %s",
                        attempt + 1, feature.id, e
                    )
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
            
            logger.error("All %d retries failed for feature %s", max_retries, feature.id)
            return feature, None, True

    logger.info("Starting concurrent WBS LLM extraction for %d features...", len(features))
    wbs_results = await asyncio.gather(*(get_feature_wbs(f) for f in features))

    # --- IN-MEMORY COMPILATION AND ID MAPPING ---
    group_records = []
    task_records = []
    group_tasks_meta = []

    for group_order, (feature, wbs_data, is_failed) in enumerate(wbs_results):
        group_name = f"Feature: {feature.title}"
        group_id = str(uuid.uuid4())
        
        task_list = []
        if wbs_data:
            milestones = []
            if isinstance(wbs_data, dict):
                if "milestones" in wbs_data:
                    milestones = wbs_data["milestones"]
                elif "milestone" in wbs_data:
                    milestones = wbs_data["milestone"]
                else:
                    for val in wbs_data.values():
                        if isinstance(val, list):
                            milestones = val
                            break
            elif isinstance(wbs_data, list):
                milestones = wbs_data

            if isinstance(milestones, list):
                for m_idx, milestone in enumerate(milestones):
                    if isinstance(milestone, dict):
                        m_title = milestone.get("title", f"Milestone {m_idx+1}")
                        m_tasks = milestone.get("tasks") or milestone.get("task") or []
                        if isinstance(m_tasks, list):
                            for t in m_tasks:
                                if isinstance(t, dict):
                                    t_title = t.get("title", "Task")
                                    t_desc = t.get("description", "No description")
                                    t_hours = 4.0
                                    try:
                                        t_hours = float(t.get("estimated_hours") or t.get("hours") or 4.0)
                                    except Exception:
                                        pass
                                    t_deps = t.get("depends_on") or t.get("dependencies") or []
                                    if not isinstance(t_deps, list):
                                        t_deps = []

                                    task_list.append({
                                        "title": f"[{m_title}] {t_title}",
                                        "raw_title": t_title,
                                        "description": t_desc,
                                        "estimated_hours": t_hours,
                                        "depends_on": t_deps,
                                        "order": len(task_list) + 1
                                    })
        
        is_fallback = False
        if not task_list:
            logger.warning("LLM failed to generate tasks for feature %s, falling back to templates", feature.id)
            task_list = WBS_TASK_TEMPLATES
            is_fallback = True

        title_to_id = {}
        feature_task_records = []
        
        feature_efforts = []
        feature_durations = []
        feature_complexities = []

        for template in task_list:
            global_task_index += 1
            task_id = str(uuid.uuid4())
            task_title = template['title']
            task_description = template.get('description', f"Task for {feature.title}")
            due = _compute_due_date(deadline, template["order"], len(task_list))

            # Compute complexity and effort in-memory
            deps_list = template.get("depends_on", [])
            complexity = _compute_task_complexity(
                title=task_title,
                description=task_description,
                dependency_count=len(deps_list),
            )
            estimated_hours = template.get("estimated_hours", 4.0)
            effort, duration = _compute_effort(
                estimated_hours,
                complexity,
                productive_hours_per_day
            )

            feature_efforts.append(effort)
            feature_durations.append(duration)
            feature_complexities.append(complexity)

            t_record = {
                "id": task_id,
                "title": task_title[:255],
                "description": task_description,
                "assignee": "unassigned",
                "priority": feature.priority or "medium",
                "dueDate": due,
                "status": "pending",
                "dependencies": json.dumps([]),
                "projectId": project_id,
                "featureRequestId": feature.id,
                "groupId": group_id,
                "order": template["order"],
                "estimatedHours": float(estimated_hours),
                "complexityScore": float(complexity),
                "effortHours": float(effort),
                "durationDays": float(duration),
            }
            
            raw_t = template.get("raw_title", task_title).lower()
            title_to_id[raw_t] = task_id
            feature_task_records.append((t_record, deps_list))
            task_records.append(t_record)

        total_effort = sum(feature_efforts)
        total_duration = sum(feature_durations)
        avg_complexity = sum(feature_complexities) / len(feature_complexities) if feature_complexities else 0.0

        group_records.append({
            "id": group_id,
            "name": group_name,
            "projectId": project_id,
            "description": feature.description or "No description provided.",
            "order": group_order,
            "featureIds": json.dumps([feature.id]),
            "totalEffortHours": round(total_effort, 2),
            "totalDurationDays": round(total_duration, 2),
            "avgComplexityScore": round(avg_complexity, 2),
        })

        group_tasks_meta.append({
            "group_id": group_id,
            "feature_id": feature.id,
            "tasks": feature_task_records,
            "title_to_id": title_to_id,
            "is_fallback": is_fallback
        })

    # --- IN-MEMORY INTRA-FEATURE DEPENDENCY RESOLUTION ---
    for meta in group_tasks_meta:
        tasks = meta["tasks"]
        is_fallback = meta["is_fallback"]
        title_to_id = meta["title_to_id"]
        
        if is_fallback:
            sorted_tasks = sorted(tasks, key=lambda x: x[0]["order"])
            for i, (task_record, _) in enumerate(sorted_tasks):
                if i > 0:
                    task_record["dependencies"] = json.dumps([sorted_tasks[i - 1][0]["id"]])
        else:
            for task_record, depends_on_list in tasks:
                dep_ids = []
                for dep_title in depends_on_list:
                    matched_id = None
                    dt_lower = dep_title.lower()
                    for raw_t, tid in title_to_id.items():
                        if dt_lower in raw_t or raw_t in dt_lower:
                            matched_id = tid
                            break
                    if matched_id:
                        dep_ids.append(matched_id)
                if dep_ids:
                    task_record["dependencies"] = json.dumps(dep_ids)

    # --- IN-MEMORY CROSS-FEATURE DEPENDENCY RESOLUTION ---
    for feature in features:
        feature_tasks = [t for t in task_records if t["featureRequestId"] == feature.id]
        feature_tasks.sort(key=lambda t: t["order"])
        if not feature_tasks:
            continue
        first_task = feature_tasks[0]

        for other_feat in features:
            if other_feat.id >= feature.id:
                continue
            other_keywords = feature_keywords.get(other_feat.id, set())
            if other_keywords and feature_keywords.get(feature.id, set()) & other_keywords:
                other_tasks = [t for t in task_records if t["featureRequestId"] == other_feat.id]
                other_tasks.sort(key=lambda t: t["order"], reverse=True)
                if other_tasks:
                    dep_id = other_tasks[0]["id"]
                    existing_deps = json.loads(first_task["dependencies"])
                    if dep_id not in existing_deps and first_task["id"] != dep_id:
                        existing_deps.append(dep_id)
                        first_task["dependencies"] = json.dumps(existing_deps)

    # --- SINGLE-PASS DB BATCH INSERT ---
    logger.info("Executing batched database writes for WBS groupings and tasks...")
    if group_records:
        await db.groupingtask.create_many(data=group_records)
    if task_records:
        await db.task.create_many(data=task_records)

    # Read back once to return expected relational model structures
    stored_groups = await db.groupingtask.find_many(
        where={"projectId": project_id},
        include={"tasks": True},
        order={"order": "asc"}
    )

    logger.info(
        "WBS generated for project %s: %d groups, %d tasks",
        project_id,
        len(stored_groups),
        len(task_records),
    )

    return {
        "groups": stored_groups,
        "total_groups": len(stored_groups),
        "total_tasks": len(task_records),
    }
