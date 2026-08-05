import json
import logging
import math
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.db.prisma import db, get_cached_project, set_cached_project, clear_project_cache

logger = logging.getLogger(__name__)

SPRINT_DEFAULT_DAYS: int = 14
MIN_SPRINT_DAYS: int = 7


async def generate_schedule(
    project_id: str,
    reindex: bool = False,
    sprint_duration_days: int = 14,
    productive_hours_per_day: float = 6.0,
) -> dict[str, Any]:
    if reindex:
        clear_project_cache(project_id)

    project = get_cached_project(project_id)
    if not project or not getattr(project, "groupingTasks", None):
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

    info = project.projectInfo
    if not info or not info.startDate:
        raise ValueError("Project start date is required in ProjectInfo")

    start = info.startDate
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)

    groups = project.groupingTasks or []
    all_tasks: list[Any] = []
    for g in groups:
        all_tasks.extend(getattr(g, "tasks", None) or [])

    total_task_hours = sum((t.effortHours or getattr(t, 'estimatedHours', 0) or 4.0) for t in all_tasks)

    if not info.deadline:
        required_headcount = 2
        total_available_days = max(1, math.ceil(total_task_hours / (required_headcount * productive_hours_per_day)))
        deadline = start + timedelta(days=total_available_days)
    else:
        deadline = info.deadline
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        total_available_days = max(1, (deadline - start).days)
        required_headcount = max(2, math.ceil(total_task_hours / (total_available_days * productive_hours_per_day)))

    if reindex:
        await _clear_schedule(project_id)

    warnings: list[dict[str, str]] = []

    # --- Feasibility check ---
    total_estimated_duration = sum(g.totalDurationDays or 0 for g in groups)
    if total_estimated_duration > total_available_days:
        warnings.append({
            "category": "timeline",
            "message": (
                f"Total estimated duration ({total_estimated_duration:.1f}d) "
                f"exceeds available timeline ({total_available_days}d)"
            ),
            "severity": "critical",
        })

    # --- Schedule tasks first ---
    task_schedules = _schedule_tasks(all_tasks, start, total_available_days, warnings, productive_hours_per_day, required_headcount)

    # Calculate actual project deadline based on scheduling
    max_end_date = start
    for ts in task_schedules:
        if ts["scheduled_end"] and ts["scheduled_end"] > max_end_date:
            max_end_date = ts["scheduled_end"]

    # Align project deadline to the actual scheduling finish date
    deadline = max_end_date

    # --- Generate Milestones ---
    milestones = await _generate_milestones(project_id, start, deadline)

    # --- Generate Sprints (aligned with task dates) ---
    sprints = await _assign_sprints(project_id, groups, start, deadline, task_schedules, sprint_duration_days)

    # --- Persist task schedules ---
    async def update_task_sched(ts):
        s_start = ts["scheduled_start"]
        s_end = ts["scheduled_end"]
        if s_start and s_start.tzinfo is None:
            s_start = s_start.replace(tzinfo=timezone.utc)
        if s_end and s_end.tzinfo is None:
            s_end = s_end.replace(tzinfo=timezone.utc)
        await db.task.update(
            where={"id": ts["task_id"]},
            data={
                "scheduledStart": s_start,
                "scheduledEnd": s_end,
            },
        )
    await asyncio.gather(*(update_task_sched(ts) for ts in task_schedules))

    # --- Persist to project summary ---
    project_summary = {}
    if project.summary:
        try:
            project_summary = json.loads(project.summary)
        except (json.JSONDecodeError, TypeError):
            project_summary = {}
    project_summary["schedule"] = {
        "start_date": start.isoformat(),
        "deadline": deadline.isoformat(),
        "expected_deadline": deadline.isoformat() if not info.deadline else None,
        "total_estimated_duration": round(total_estimated_duration, 1),
        "feasible": total_estimated_duration <= total_available_days,
        "required_headcount": required_headcount,
        "milestones": [
            {"title": m.title, "due_date": m.dueDate.isoformat()}
            for m in milestones
        ],
        "sprint_count": len(sprints),
        "task_count": len(all_tasks),
    }
    await db.project.update(
        where={"id": project_id},
        data={"summary": json.dumps(project_summary)},
    )

    logger.info(
        "Schedule generated for project %s: %d milestones, %d sprints, %d tasks",
        project_id, len(milestones), len(sprints), len(all_tasks),
    )

    return {
        "start_date": start,
        "deadline": deadline,
        "milestones": milestones,
        "sprints": sprints,
        "tasks": task_schedules,
        "warnings": warnings,
        "feasible": total_estimated_duration <= total_available_days,
        "required_headcount": required_headcount,
        "expected_deadline": deadline if not info.deadline else None,
    }


async def _clear_schedule(project_id: str) -> None:
    await db.milestone.delete_many(where={"projectId": project_id})
    await db.groupingtask.update_many(
        where={"projectId": project_id},
        data={"sprintNumber": None, "sprintStart": None, "sprintEnd": None},
    )
    await db.task.update_many(
        where={"projectId": project_id},
        data={"scheduledStart": None, "scheduledEnd": None},
    )


async def _generate_milestones(
    project_id: str,
    start: datetime,
    deadline: datetime,
) -> list[Any]:
    total_secs = (deadline - start).total_seconds()
    # Generate milestones at key project phases
    phases = [
        (0.20, "Requirements & Design Complete"),
        (0.45, "Core Development Complete"),
        (0.70, "Testing & QA Complete"),
        (1.00, "Project Delivery"),
    ]

    import uuid
    records_data = []
    for pct, title in phases:
        offset = timedelta(seconds=int(total_secs * pct))
        due = start + offset
        phase_start = start + timedelta(seconds=int(total_secs * (pct - 0.15))) if pct > 0.15 else start

        records_data.append({
            "id": str(uuid.uuid4()),
            "projectId": project_id,
            "title": title,
            "description": f"Milestone: {title}",
            "startDate": phase_start,
            "endDate": due,
            "dueDate": due,
            "status": "PENDING",
            "createdBy": "system",
        })

    if records_data:
        await db.milestone.create_many(data=records_data)

    return await db.milestone.find_many(
        where={"projectId": project_id}
    )


async def _assign_sprints(
    project_id: str,
    groups: list[Any],
    start: datetime,
    deadline: datetime,
    task_schedules: list[dict[str, Any]],
    sprint_duration_days: int = 14,
) -> list[dict[str, Any]]:
    tid_to_schedule = {ts["task_id"]: ts for ts in task_schedules}
    group_update_calls = []
    active_groups: dict[str, dict[str, Any]] = {}

    max_sprint_num = 1
    task_sprints: dict[str, int] = {}
    for ts in task_schedules:
        s_start = ts["scheduled_start"]
        if s_start:
            days_from_start = (s_start - start).days
            sprint_idx = max(0, days_from_start // sprint_duration_days)
            sprint_num = sprint_idx + 1
            task_sprints[ts["task_id"]] = sprint_num
            if sprint_num > max_sprint_num:
                max_sprint_num = sprint_num

    for s_idx in range(max_sprint_num):
        sprint_num = s_idx + 1
        s_start = start + timedelta(days=s_idx * sprint_duration_days)
        s_end = s_start + timedelta(days=sprint_duration_days)

        sprint_tids = [tid for tid, s_num in task_sprints.items() if s_num == sprint_num]
        if not sprint_tids:
            continue

        group_tasks: dict[str, list[Any]] = defaultdict(list)
        for tid in sprint_tids:
            for g in groups:
                for t in (getattr(g, "tasks", None) or []):
                    if t.id == tid:
                        group_tasks[g.id].append(t)

        for gid, gtasks in group_tasks.items():
            g = next((g for g in groups if g.id == gid), None)
            if g:
                async def update_group_sprint(g_id=gid, s_num=sprint_num, start_d=s_start, end_d=s_end):
                    await db.groupingtask.update(
                        where={"id": g_id},
                        data={
                            "sprintNumber": s_num,
                            "sprintStart": start_d,
                            "sprintEnd": end_d,
                        },
                    )
                group_update_calls.append(update_group_sprint())

                key = f"{gid}_{sprint_num}"
                if key not in active_groups:
                    active_groups[key] = {
                        "sprint_number": sprint_num,
                        "group_id": gid,
                        "group_name": g.name or "",
                        "start_date": s_start,
                        "end_date": s_end,
                        "total_effort_hours": sum(
                            t.effortHours or t.estimatedHours or 0 for t in gtasks
                        ),
                        "task_count": len(gtasks),
                    }
                    
    if group_update_calls:
        await asyncio.gather(*group_update_calls)

    sprints_info = sorted(active_groups.values(), key=lambda x: (x["sprint_number"], x["group_name"]))
    return sprints_info


def _schedule_tasks(
    tasks: list[Any],
    project_start: datetime,
    total_days: int,
    warnings: list[dict[str, str]],
    productive_hours_per_day: float = 6.0,
    required_headcount: int = 2,
) -> list[dict[str, Any]]:
    if not tasks:
        return []

    import re
    def get_m_num(title: str) -> int:
        match = re.search(r'Milestone\s*(\d+)', title, re.IGNORECASE)
        return int(match.group(1)) if match else 999

    tasks = sorted(tasks, key=lambda t: get_m_num(t.title))

    task_deps: dict[str, list[str]] = {}
    for t in tasks:
        deps: list[str] = []
        if t.dependencies:
            try:
                deps = json.loads(t.dependencies)
            except (json.JSONDecodeError, TypeError):
                deps = [t.dependencies] if t.dependencies else []
        task_deps[t.id] = deps

    # Topological sort
    in_degree: dict[str, int] = {t.id: 0 for t in tasks}
    dep_to: dict[str, list[str]] = defaultdict(list)

    for t in tasks:
        for dep_id in task_deps.get(t.id, []):
            if dep_id in in_degree:
                dep_to[dep_id].append(t.id)
                in_degree[t.id] = in_degree.get(t.id, 0) + 1

    queue: deque = deque(tid for tid, deg in in_degree.items() if deg == 0)
    topo_order: list[str] = []

    while queue:
        tid = queue.popleft()
        topo_order.append(tid)
        for child in dep_to.get(tid, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(topo_order) != len(tasks):
        warnings.append({
            "category": "dependency",
            "message": f"Circular dependencies detected among {len(tasks) - len(topo_order)} tasks",
            "severity": "high",
        })
        topo_order = [t.id for t in tasks]

    tid_map = {t.id: t for t in tasks}
    scheduled_starts: dict[str, datetime] = {}
    scheduled_ends: dict[str, datetime] = {}
    daily_allocation: dict[Any, int] = {}

    for tid in topo_order:
        t = tid_map.get(tid)
        if not t:
            continue
        duration = t.durationDays or (getattr(t, 'estimatedHours', getattr(t, 'effortHours', 4))) / productive_hours_per_day
        duration = max(0.5, duration)
        duration_days = max(1, math.ceil(duration))

        deps = task_deps.get(tid, [])
        latest_end = project_start
        for dep_id in deps:
            dep_end = scheduled_ends.get(dep_id)
            if dep_end and dep_end > latest_end:
                latest_end = dep_end

        task_start_date = latest_end
        
        while True:
            can_schedule = True
            for i in range(duration_days):
                check_date = task_start_date + timedelta(days=i)
                current_active = daily_allocation.get(check_date.date(), 0)
                if current_active >= required_headcount:
                    can_schedule = False
                    break
            
            if can_schedule:
                break
            else:
                task_start_date += timedelta(days=1)
                
        for i in range(duration_days):
            check_date = task_start_date + timedelta(days=i)
            daily_allocation[check_date.date()] = daily_allocation.get(check_date.date(), 0) + 1

        task_end_date = task_start_date + timedelta(days=duration)

        scheduled_starts[tid] = task_start_date
        scheduled_ends[tid] = task_end_date

    results: list[dict[str, Any]] = []
    for tid in topo_order:
        t = tid_map.get(tid)
        if not t:
            continue
        deps_list: Optional[list[str]] = None
        if task_deps.get(tid):
            deps_list = task_deps[tid]
        results.append({
            "task_id": tid,
            "title": t.title,
            "group_id": t.groupId,
            "scheduled_start": scheduled_starts.get(tid),
            "scheduled_end": scheduled_ends.get(tid),
            "duration_days": round(t.durationDays or (t.effortHours or 4) / productive_hours_per_day, 1),
            "dependencies": deps_list,
        })

    return results
