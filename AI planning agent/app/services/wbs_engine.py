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

MONOLITHIC_WBS_PROMPT = """You are an Expert Project Architect and Technical Project Manager.
Given the global project context, technologies, and a list of specific feature requests, generate a complete, production-ready, Milestone-Driven WBS plan for the entire project.

CRITICAL ARCHITECTURE DIRECTIVE (MONOLITHIC DELIVERY):
- The project is a Monolithic codebase. Engineering work happens horizontally in lifecycle phases, NOT in vertical feature silos.
- Boilerplate tasks (e.g. setting up postgres database, base repository initialization, configuring monolithic structure) MUST only occur ONCE globally under "Shared Infrastructure & Core Baseline" at Milestone 1. Do NOT duplicate these tasks across features.
- All modules run in the same process and use the same database.

CRITICAL DIRECTIVES FOR TASK GRANULARITY:
- Do NOT generate generic tasks matching feature titles exactly (e.g. do NOT write a task named "Faculty Management" or "Student Registration").
- You MUST create multiple granular, technical, actionable tasks for each feature request across the different Milestones based on the provided "Database Tables and Entities Schema" and "REST API Routes and Endpoint Specifications".
- In Milestone 1, create specific database migration and table creation tasks (e.g., "Create Faculty and Department Tables", "Set up Student registration tables").
- In Milestone 2, create specific backend API endpoint and business logic tasks (e.g., "Implement Faculty profile API routes", "Build Student enrollment validation routes").
- In Milestone 3, create specific frontend integration tasks (e.g., "Develop Student Registration Form view", "Integrate Course Enrollment front-end components").
- Each milestone must contain detailed technical sub-tasks for the features, not just a single high-level feature placeholder task.

Generate tasks organized into exactly 5 logical, sequential Milestone phases:
- Milestone 1: Infrastructure & Database Setup (Shared infrastructure setup, database schema setup, base layout)
- Milestone 2: Backend API & Core Logic (Business logic, API routes, module controllers)
- Milestone 3: Frontend UI & Integration (UI views, API integration, client-side modules)
- Milestone 4: QA Verification & Testing (Testing, QA, profiling)
- Milestone 5: Deployment & Go-Live (CI/CD, production release)

For each task you generate, you MUST attribute it to the corresponding feature request ID from the provided list. If the task is cross-cutting or shared infrastructure, attribute it to "shared".

Return ONLY a valid JSON object matching this schema exactly. Do not wrap in markdown blocks, add trailing commas, or include conversational text:
{
  "milestones": [
    {
      "title": "Milestone Phase Title",
      "description": "Milestone Description detailing phase goals",
      "tasks": [
        {
          "title": "Granular, context-specific task title",
          "description": "Explicit step-by-step technical execution plan for this task",
          "estimated_hours": 4.5,
          "feature_id": "feature-request-uuid-value-here or 'shared'",
          "depends_on": ["Title of previous task this blocks on", "Can be empty [] for independent tasks"]
        }
      ]
    }
  ]
}
"""

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
        has_fr = await db.featurerequest.count(where={"projectId": project_id}) > 0
        delete_where = {"projectId": project_id}
        if has_fr:
            delete_where["featureRequestId"] = {"not": None}
        await db.task.delete_many(where=delete_where)
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
    arch_type = "Monolithic"
    
    if info:
        desc = info.description or ""
        scope = info.scope or ""
        constraints = info.constraints or ""
        arch_type = info.projectType or "Monolithic"
        context = f"Project: {info.name}\nDescription: {desc}\nScope: {scope}\nConstraints: {constraints}\nTechnology Stack: {tech_stack}\nPreferred Architecture Pattern: {arch_type}\n"

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
        # Check Redis connection
        redis_available = False
        redis_client = None
        token_id = str(uuid.uuid4())
        try:
            from app.db.redis import get_redis_client, is_redis_available
            if await is_redis_available():
                redis_client = get_redis_client()
                redis_available = True
        except Exception:
            pass

        # Concurrency Wrapper
        class ConcurrencyGuard:
            async def __aenter__(self):
                if redis_available and redis_client:
                    import time
                    acquired = False
                    while not acquired:
                        now = time.time()
                        # Clean up expired tokens (>60 seconds old)
                        await redis_client.zremrangebyscore("sem:llm:tokens", 0, now - 60)
                        count = await redis_client.zcard("sem:llm:tokens")
                        if count < 4:
                            await redis_client.zadd("sem:llm:tokens", {token_id: now})
                            acquired = True
                        else:
                            await asyncio.sleep(0.5)
                else:
                    await sem.acquire()

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if redis_available and redis_client:
                    await redis_client.zrem("sem:llm:tokens", token_id)
                else:
                    sem.release()

        async with ConcurrencyGuard():
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    feature_payload = (
                        f"Preferred Architecture Pattern: {arch_type}\n"
                        f"Feature: {feature.title}: {feature.description}\n"
                        "Note: If Monolithic Architecture, this feature is a local module inside a single codebase, not an independent network service. Treat integration tasks as in-process calls."
                    )
                    wbs_data = await ollama_service.generate_combined_wbs(
                        context=context,
                        feature=feature_payload
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

    is_monolith = arch_type.lower() in ["monolith", "monolithic"]
    if is_monolith:
        return await _generate_wbs_monolithic(
            project_id=project_id,
            project=project,
            features=features,
            wbs_results=wbs_results,
            deadline=deadline,
            productive_hours_per_day=productive_hours_per_day
        )

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


def _extract_clean_title(title: str) -> tuple[str, str]:
    import re
    match = re.match(r"^(\[[^\]]+\]\s*)(.*)$", title)
    prefix, core = ("", title)
    if match:
        prefix, core = match.group(1), match.group(2)
    
    # Strip relative task numbering prefixes (e.g. "Task 1.1:", "Task 2 -", "Task 1.2")
    core = re.sub(r"^task\s+[0-9a-zA-Z]+(?:\.[0-9a-zA-Z]+)*\s*[:\-\s]\s*", "", core, flags=re.IGNORECASE)
    
    core = core.strip()
    if core:
        core = core[0].upper() + core[1:]
        
    return prefix, core


def _generate_monolithic_fallback(features: list[Any], project_id: str) -> dict[str, Any]:
    milestones = [
        {
            "title": "Milestone 1: Core Foundation & Context Baseline",
            "description": "Initialize database schema and monolithic codebase setup",
            "tasks": [
                {"title": "Initialize Database & Monolithic Codebase", "description": "Set up PostgreSQL database and directory structure.", "estimated_hours": 8.0, "feature_id": "shared"}
            ]
        },
        {
            "title": "Milestone 2: Functional Feature Engineering & System Core",
            "description": "Implement core backend models, business logic and APIs for features",
            "tasks": []
        },
        {
            "title": "Milestone 3: Interface Integration & Frontend Synchronization",
            "description": "Develop and integrate UI screens and APIs",
            "tasks": []
        },
        {
            "title": "Milestone 4: Verification, Hardening, & Risk Mitigation",
            "description": "Testing, QA, and validation",
            "tasks": [
                {"title": "Unified QA Verification & E2E Testing", "description": "Execute comprehensive end-to-end integration testing.", "estimated_hours": 12.0, "feature_id": "shared"}
            ]
        },
        {
            "title": "Milestone 5: Deployment Go-Live & Pipeline Sign-off",
            "description": "Unified production release",
            "tasks": [
                {"title": "Monolithic Production Deployment", "description": "Configure unified CI/CD release pipeline.", "estimated_hours": 6.0, "feature_id": "shared"}
            ]
        }
    ]

    for f in features:
        milestones[1]["tasks"].append({
            "title": f"Implement Core Logic for {f.title}",
            "description": f"Build database mappings, business logic, and API endpoints for {f.title}.",
            "estimated_hours": 16.0,
            "feature_id": f.id
        })
        milestones[2]["tasks"].append({
            "title": f"Build UI components and Integrate {f.title}",
            "description": f"Develop frontend views and hook them up to the API endpoints for {f.title}.",
            "estimated_hours": 12.0,
            "feature_id": f.id
        })

    return {"milestones": milestones}


def is_generic_boilerplate(title: str, feature_title: str) -> bool:
    title_lower = title.lower()
    feat_lower = feature_title.lower()
    
    # If the title mentions specific feature keywords, it is not generic boilerplate
    # Split feature title into words, filtering out short words
    feat_words = [w for w in re.findall(r"[a-z0-9]+", feat_lower) if len(w) > 3]
    has_feature_context = any(w in title_lower for w in feat_words)
    
    if has_feature_context:
        return False
        
    # Check for boilerplate indicators
    boilerplate_words = ["initialize", "setup", "boilerplate", "ci/cd", "deployment", "deploy", "testing", "qa", "e2e", "workflow"]
    return any(w in title_lower for w in boilerplate_words)


async def _generate_wbs_monolithic(
    project_id: str,
    project: Any,
    features: list[Any],
    wbs_results: list[tuple[Any, Optional[dict[str, Any]], bool]],
    deadline: Optional[datetime],
    productive_hours_per_day: float = 6.0
) -> dict[str, Any]:
    global_milestones = [
        {
            "title": "Milestone 1: Infrastructure & Database Setup",
            "description": "Initialize database schema, core models, and monolithic codebase baseline.",
            "tasks": [
                {
                    "title": "Initialize PostgreSQL Database & Monolithic Codebase",
                    "description": "Set up the shared PostgreSQL database instance, configure migrations baseline, and initialize the monolithic repository directory structure.",
                    "estimated_hours": 8.0,
                    "feature_id": "shared",
                    "depends_on": []
                }
            ]
        },
        {
            "title": "Milestone 2: Backend API & Core Logic",
            "description": "Implement business logic, API routes, controllers, and core backend modules.",
            "tasks": []
        },
        {
            "title": "Milestone 3: Frontend UI & Integration",
            "description": "Develop UI screens, integrate API endpoints, and synchronize client-side components.",
            "tasks": []
        },
        {
            "title": "Milestone 4: QA Verification & Testing",
            "description": "QA verification, end-to-end integration testing, and bug fixing.",
            "tasks": [
                {
                    "title": "Unified QA Verification & End-to-End Testing",
                    "description": "Execute comprehensive testing suite, including unit tests, API integration tests, and UAT verification.",
                    "estimated_hours": 12.0,
                    "feature_id": "shared",
                    "depends_on": []
                }
            ]
        },
        {
            "title": "Milestone 5: Deployment & Go-Live",
            "description": "Final release preparation, CI/CD pipeline setup, and production deployment.",
            "tasks": [
                {
                    "title": "Monolithic Production Deployment",
                    "description": "Configure CI/CD release pipeline and deploy the monolithic codebase package to the target server.",
                    "estimated_hours": 6.0,
                    "feature_id": "shared",
                    "depends_on": []
                }
            ]
        }
    ]

    def get_milestone_index(title_str: str, default_idx: int) -> int:
        t_lower = title_str.lower()
        if "milestone 1" in t_lower or "infrastructure" in t_lower or "database" in t_lower or "foundation" in t_lower or "baseline" in t_lower:
            return 0
        if "milestone 2" in t_lower or "backend" in t_lower or "api" in t_lower or "core logic" in t_lower or "feature engineering" in t_lower or "system core" in t_lower:
            return 1
        if "milestone 3" in t_lower or "frontend" in t_lower or "ui" in t_lower or "integration" in t_lower or "synchronization" in t_lower:
            return 2
        if "milestone 4" in t_lower or "qa" in t_lower or "testing" in t_lower or "verification" in t_lower or "hardening" in t_lower:
            return 3
        if "milestone 5" in t_lower or "deployment" in t_lower or "go-live" in t_lower or "pipeline" in t_lower:
            return 4
        return min(max(default_idx, 0), 4)

    # Process results from concurrent LLM queries
    seen_feature_backend_tasks = {}
    for feature, wbs_data, is_failed in wbs_results:
        if feature.id not in seen_feature_backend_tasks:
            seen_feature_backend_tasks[feature.id] = set()

        # If failed, generate basic fallback tasks for this feature
        if is_failed or not wbs_data:
            # Add backend task to Milestone 2
            global_milestones[1]["tasks"].append({
                "title": f"Implement Core Logic for {feature.title}",
                "description": f"Build database mappings, business logic, and API endpoints for {feature.title}.",
                "estimated_hours": 16.0,
                "feature_id": feature.id,
                "depends_on": []
            })
            # Add frontend task to Milestone 3
            global_milestones[2]["tasks"].append({
                "title": f"Build UI components and Integrate {feature.title}",
                "description": f"Develop frontend views and hook them up to the API endpoints for {feature.title}.",
                "estimated_hours": 12.0,
                "feature_id": feature.id,
                "depends_on": []
            })
            continue

        # Extract milestones list
        milestones_list = []
        if isinstance(wbs_data, dict):
            if "milestones" in wbs_data:
                milestones_list = wbs_data["milestones"]
            elif "milestone" in wbs_data:
                milestones_list = wbs_data["milestone"]
            else:
                for val in wbs_data.values():
                     if isinstance(val, list):
                         milestones_list = val
                         break
        elif isinstance(wbs_data, list):
            milestones_list = wbs_data

        if not isinstance(milestones_list, list):
            milestones_list = []

        for default_idx, milestone in enumerate(milestones_list):
            if not isinstance(milestone, dict):
                continue
            
            m_title = milestone.get("title") or f"Milestone {default_idx + 1}"
            m_idx = get_milestone_index(m_title, default_idx)

            m_tasks = milestone.get("tasks") or milestone.get("task") or []
            if not isinstance(m_tasks, list):
                m_tasks = [m_tasks]

            for t in m_tasks:
                if not isinstance(t, dict):
                    continue
                t_title = t.get("title") or "Unnamed Task"
                t_desc = t.get("description") or "No description"
                t_hours = 4.0
                try:
                    t_hours = float(t.get("estimated_hours") or t.get("hours") or 4.0)
                except Exception:
                    pass
                t_deps = t.get("depends_on") or t.get("dependencies") or []
                if not isinstance(t_deps, list):
                    t_deps = []

                # Monolithic Pruning Rule 1: Skip any feature-specific deployment/docker/containerization tasks in Milestone 5
                t_title_lower = t_title.lower()
                if m_idx == 4 and any(w in t_title_lower for w in ["deploy", "docker", "container", "nginx", "apache", "provision", "sign off", "sign-off", "pipeline sign-off", "pipeline sign off", "release preparation"]):
                    continue

                # Monolithic Pruning Rule 2: Skip generic "setup monolithic architecture" or "integrate into monolithic architecture" in Milestone 1
                if m_idx == 0 and any(w in t_title_lower for w in ["monolithic architecture", "in-process calls", "integrate student registration", "integrate the student registration", "typescript", "project structure", "setup project", "setup environment", "initial repo", "repository structure", "project boilerplate", "git repository", "setup repo"]):
                    continue

                # Relocate backend implementation tasks that leak into Milestone 1 to Milestone 2
                if m_idx == 0 and any(w in t_title_lower for w in ["implement", "develop", "write code", "coding", "logic", "functionality", "controller", "service layer", "business logic", "route handler", "endpoints", "module"]):
                    # Unless it's database schema/table setup
                    if not any(w in t_title_lower for w in ["database", "schema", "table", "migration", "entity", "model", "index", "relationship"]):
                        m_idx = 1

                # Skip/de-duplicate generic boilerplate setup tasks
                if is_generic_boilerplate(t_title, feature.title):
                    continue

                # Backend Task De-duplication Rule (Milestones 1 & 2):
                if m_idx in (0, 1):
                    t_clean = t_title_lower
                    t_clean = re.sub(r"\b(implement|develop|create|write|setup|configure|logic|feature|module|functionality|code|component|system)\b", "", t_clean)
                    words = [w for w in re.findall(r"[a-z0-9]+", t_clean) if len(w) > 3]
                    norm_key = " ".join(sorted(list(set(words))))
                    if norm_key:
                        if norm_key in seen_feature_backend_tasks[feature.id]:
                            continue
                        seen_feature_backend_tasks[feature.id].add(norm_key)

                global_milestones[m_idx]["tasks"].append({
                    "title": t_title,
                    "description": t_desc,
                    "estimated_hours": t_hours,
                    "feature_id": feature.id,
                    "depends_on": t_deps
                })

    # Database insertions
    group_records = []
    task_records = []
    title_to_id = {}
    all_task_deps_meta = []
    global_task_index = 0

    for m_idx, milestone in enumerate(global_milestones):
        m_title = milestone["title"]
        m_desc = milestone["description"]
        
        group_id = str(uuid.uuid4())
        m_tasks = milestone["tasks"]
        group_efforts = []
        group_durations = []
        group_complexities = []
        feature_ids_in_milestone = set()

        for t_idx, t in enumerate(m_tasks):
            t_title = t["title"]
            t_desc = t["description"]
            t_hours = t["estimated_hours"]
            t_feature_id = t["feature_id"]
            
            priority = "medium"
            resolved_feature_id = None
            prefix = "[Shared Infrastructure & Core Baseline] "

            if t_feature_id and t_feature_id != "shared":
                # Find matching feature request
                matched_f = next((f for f in features if f.id == t_feature_id), None)
                if matched_f:
                    resolved_feature_id = matched_f.id
                    priority = matched_f.priority or "medium"
                    prefix = f"[Feature: {matched_f.title}] "
                    feature_ids_in_milestone.add(matched_f.id)
            
            _, clean_title = _extract_clean_title(t_title)
            final_title = f"{prefix}{clean_title}"[:255]

            deps_list = t["depends_on"]
            complexity = _compute_task_complexity(
                title=final_title,
                description=t_desc,
                dependency_count=len(deps_list)
            )
            effort, duration = _compute_effort(
                t_hours,
                complexity,
                productive_hours_per_day
            )

            group_efforts.append(effort)
            group_durations.append(duration)
            group_complexities.append(complexity)

            global_task_index += 1
            task_id = str(uuid.uuid4())
            
            title_to_id[t_title.lower().strip()] = task_id
            title_to_id[clean_title.lower().strip()] = task_id

            due = _compute_due_date(deadline, m_idx + 1, len(global_milestones))

            t_record = {
                "id": task_id,
                "title": final_title,
                "description": t_desc,
                "assignee": "unassigned",
                "priority": priority,
                "dueDate": due,
                "status": "pending",
                "dependencies": json.dumps([]),
                "projectId": project_id,
                "featureRequestId": resolved_feature_id,
                "groupId": group_id,
                "order": t_idx + 1,
                "estimatedHours": float(t_hours),
                "complexityScore": float(complexity),
                "effortHours": float(effort),
                "durationDays": float(duration),
            }
            task_records.append(t_record)
            all_task_deps_meta.append((t_record, deps_list))

        total_effort = sum(group_efforts)
        total_duration = sum(group_durations)
        avg_complexity = sum(group_complexities) / len(group_complexities) if group_complexities else 0.0

        group_records.append({
            "id": group_id,
            "name": m_title,
            "projectId": project_id,
            "description": m_desc,
            "order": m_idx,
            "featureIds": json.dumps(list(feature_ids_in_milestone)),
            "totalEffortHours": round(total_effort, 2),
            "totalDurationDays": round(total_duration, 2),
            "avgComplexityScore": round(avg_complexity, 2),
        })

    # Dependencies resolution
    for task_record, depends_on_list in all_task_deps_meta:
        dep_ids = []
        for dep_title in depends_on_list:
            matched_id = None
            dt_lower = dep_title.lower().strip()
            if dt_lower in title_to_id:
                matched_id = title_to_id[dt_lower]
            else:
                for raw_t, tid in title_to_id.items():
                    if dt_lower in raw_t or raw_t in dt_lower:
                        matched_id = tid
                        break
            if matched_id and matched_id != task_record["id"]:
                dep_ids.append(matched_id)
        if dep_ids:
            task_record["dependencies"] = json.dumps(dep_ids)

    logger.info("Executing batched database writes for Monolithic WBS groupings and tasks...")
    if group_records:
        await db.groupingtask.create_many(data=group_records)
    if task_records:
        await db.task.create_many(data=task_records)

    stored_groups = await db.groupingtask.find_many(
        where={"projectId": project_id},
        include={"tasks": True},
        order={"order": "asc"}
    )

    logger.info(
        "Monolithic WBS generated for project %s: %d groups, %d tasks",
        project_id,
        len(stored_groups),
        len(task_records),
    )

    return {
        "groups": stored_groups,
        "total_groups": len(stored_groups),
        "total_tasks": len(task_records),
    }
