import json
import logging
import math
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.db.prisma import db

logger = logging.getLogger(__name__)

SPRINT_DEFAULT_DAYS: int = 14
MIN_SPRINT_DAYS: int = 7


async def generate_schedule(
    project_id: str,
    reindex: bool = False,
    sprint_duration_days: int = 14,
    productive_hours_per_day: float = 6.0,
) -> dict[str, Any]:
    project = await db.project.find_unique(
        where={"id": project_id},
        include={
            "projectInfo": True,
            "groupingTasks": {"include": {"tasks": True}},
            "resourceAllocations": True,
        },
    )

    if not project:
        raise ValueError(f"Project {project_id} not found")

    info = project.projectInfo
    if not info or not info.startDate or not info.deadline:
        raise ValueError("Project start date and deadline are required in ProjectInfo")

    start = info.startDate
    deadline = info.deadline
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)

    total_available_days = max(1, (deadline - start).days)

    if reindex:
        await _clear_schedule(project_id)

    groups = project.groupingTasks or []
    all_tasks: list[Any] = []
    for g in groups:
        all_tasks.extend(getattr(g, "tasks", None) or [])

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

    total_task_hours = sum((t.effortHours or getattr(t, 'estimatedHours', 0) or 4.0) for t in all_tasks)
    required_headcount = max(2, math.ceil(total_task_hours / (total_available_days * productive_hours_per_day)))

    # --- Generate Milestones ---
    milestones = await _generate_milestones(project_id, start, deadline)

    # --- Generate Sprints ---
    sprints = await _assign_sprints(project_id, groups, start, deadline, sprint_duration_days, productive_hours_per_day, required_headcount)

    # --- Schedule tasks ---
    task_schedules = _schedule_tasks(all_tasks, start, total_available_days, warnings, productive_hours_per_day, required_headcount)

    # --- Persist task schedules ---
    for ts in task_schedules:
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
    }


async def _clear_schedule(project_id: str) -> None:
    await db.milestone.delete_many(where={"projectId": project_id})
    groups = await db.groupingtask.find_many(
        where={"projectId": project_id},
        include={"tasks": True},
    )
    for g in groups:
        await db.groupingtask.update(
            where={"id": g.id},
            data={"sprintNumber": None, "sprintStart": None, "sprintEnd": None},
        )
        for t in (g.tasks or []):
            await db.task.update(
                where={"id": t.id},
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

    records: list[Any] = []
    for pct, title in phases:
        offset = timedelta(seconds=int(total_secs * pct))
        due = start + offset
        phase_start = start + timedelta(seconds=int(total_secs * (pct - 0.15))) if pct > 0.15 else start

        record = await db.milestone.create(
            data={
                "projectId": project_id,
                "title": title,
                "description": f"Milestone: {title}",
                "startDate": phase_start,
                "endDate": due,
                "dueDate": due,
                "status": "PENDING",
                "createdBy": "system",
            },
        )
        records.append(record)

    return records


async def _assign_sprints(
    project_id: str,
    groups: list[Any],
    start: datetime,
    deadline: datetime,
    sprint_duration_days: int = 14,
    productive_hours_per_day: float = 6.0,
    required_headcount: int = 2,
) -> list[dict[str, Any]]:
    # 1. Gather all tasks
    all_sprint_tasks: list[tuple[float, Any]] = []
    for g_idx, group in enumerate(groups):
        tasks = getattr(group, "tasks", None) or []
        for task in tasks:
            effort = task.effortHours or task.estimatedHours or 4
            all_sprint_tasks.append((effort, task))

    # Sort by order
    all_sprint_tasks.sort(key=lambda x: x[1].order if hasattr(x[1], "order") else 0)

    task_deps: dict[str, list[str]] = {}
    for _, task in all_sprint_tasks:
        deps: list[str] = []
        if task.dependencies:
            try:
                deps = json.loads(task.dependencies)
            except (json.JSONDecodeError, TypeError):
                deps = [task.dependencies] if task.dependencies else []
        task_deps[task.id] = deps

    # Topological sort for sprint assignment
    in_degree: dict[str, int] = {t.id: 0 for _, t in all_sprint_tasks}
    dep_to: dict[str, list[str]] = defaultdict(list)

    for _, task in all_sprint_tasks:
        for dep_id in task_deps.get(task.id, []):
            if dep_id in in_degree:
                dep_to[dep_id].append(task.id)
                in_degree[task.id] = in_degree.get(task.id, 0) + 1

    queue: deque = deque(tid for tid, deg in in_degree.items() if deg == 0)
    topo_order: list[str] = []

    while queue:
        tid = queue.popleft()
        topo_order.append(tid)
        for child in dep_to[tid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(topo_order) != len(all_sprint_tasks):
        logger.warning("Circular dependency detected; using fallback ordering")
        topo_order = [t.id for _, t in all_sprint_tasks]

    tid_to_task = {t.id: t for _, t in all_sprint_tasks}
    tid_to_effort = {t.id: e for e, t in all_sprint_tasks}

    # Assign to sprints dynamically
    MAX_SPRINT_EFFORT = sprint_duration_days * productive_hours_per_day * required_headcount
    sprint_efforts: list[float] = [0.0]
    sprint_tasks: list[list[Any]] = [[]]
    dep_sprint_map: dict[str, int] = {}

    for tid in topo_order:
        task = tid_to_task.get(tid)
        if not task:
            continue
        effort = tid_to_effort.get(tid, 4.0)

        deps = task_deps.get(tid, [])
        latest_dep_sprint = -1
        for dep_id in deps:
            ds = dep_sprint_map.get(dep_id, -1)
            if ds > latest_dep_sprint:
                latest_dep_sprint = ds

        # Find the first available sprint after the latest dependency
        start_search = max(0, latest_dep_sprint)
        assigned_sprint = -1
        
        for s in range(start_search, len(sprint_efforts)):
            if sprint_efforts[s] + effort <= MAX_SPRINT_EFFORT:
                assigned_sprint = s
                break
                
        # If no existing sprint has capacity, append a new one
        if assigned_sprint == -1:
            # We must also respect the dependency, so we can't assign earlier than latest_dep_sprint
            # if start_search is beyond current len, we need to pad (shouldn't happen sequentially, but just in case)
            assigned_sprint = max(start_search, len(sprint_efforts))
            while len(sprint_efforts) <= assigned_sprint:
                sprint_efforts.append(0.0)
                sprint_tasks.append([])

        sprint_efforts[assigned_sprint] += effort
        dep_sprint_map[tid] = assigned_sprint
        sprint_tasks[assigned_sprint].append(task)

    num_sprints = len(sprint_efforts)

    # Update GroupingTask with sprint metadata
    sprints_info: list[dict[str, Any]] = []
    active_groups: dict[str, dict[str, Any]] = {}

    for s_idx in range(num_sprints):
        if not sprint_tasks[s_idx]:
            continue
        s_start = start + timedelta(days=s_idx * sprint_duration_days)
        s_end = s_start + timedelta(days=sprint_duration_days)

        group_tasks: dict[str, list[Any]] = defaultdict(list)
        for task in sprint_tasks[s_idx]:
            if task.groupId:
                group_tasks[task.groupId].append(task)

        for gid, gtasks in group_tasks.items():
            g = next((g for g in groups if g.id == gid), None)
            if g:
                sprint_num = s_idx + 1
                await db.groupingtask.update(
                    where={"id": gid},
                    data={
                        "sprintNumber": sprint_num,
                        "sprintStart": s_start,
                        "sprintEnd": s_end,
                    },
                )
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
        if task_end_date > project_start + timedelta(days=total_days):
            task_end_date = project_start + timedelta(days=total_days)

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
