import logging
from datetime import timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status

from app.db.prisma import db
from app.models.schemas import (
    ErrorResponse,
    MilestoneResponse,
    ScheduleRequest,
    ScheduleResponse,
    ScheduleWarning,
    ScheduledTaskInfo,
    SprintInfo,
)
from app.services.schedule_engine import generate_schedule

logger = logging.getLogger(__name__)

router = APIRouter()


def _milestone_to_response(m: Any) -> MilestoneResponse:
    return MilestoneResponse(
        id=m.id,
        project_id=m.projectId,
        title=m.title,
        description=m.description,
        start_date=m.startDate,
        end_date=m.endDate,
        due_date=m.dueDate,
        status=m.status,
        created_at=m.createdAt,
    )


@router.post(
    "/{project_id}/schedule",
    response_model=ScheduleResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def generate_schedule_endpoint(
    project_id: str,
    body: ScheduleRequest = ScheduleRequest(),
) -> ScheduleResponse:
    try:
        result = await generate_schedule(
            project_id, 
            reindex=body.reindex,
            sprint_duration_days=body.sprint_duration_days or 14,
            productive_hours_per_day=body.productive_hours_per_day or 6.0
        )
        project = await db.project.find_unique(where={"id": project_id})

        milestones = [_milestone_to_response(m) for m in result["milestones"]]
        sprints = [
            SprintInfo(
                sprint_number=s["sprint_number"],
                group_id=s["group_id"],
                group_name=s["group_name"],
                start_date=s["start_date"],
                end_date=s["end_date"],
                total_effort_hours=round(s["total_effort_hours"], 2),
                task_count=s["task_count"],
            )
            for s in result["sprints"]
        ]
        tasks = [
            ScheduledTaskInfo(
                task_id=t["task_id"],
                title=t["title"],
                group_id=t["group_id"],
                scheduled_start=t["scheduled_start"],
                scheduled_end=t["scheduled_end"],
                duration_days=t["duration_days"],
                dependencies=t["dependencies"],
            )
            for t in result["tasks"]
        ]
        warnings_list = [
            ScheduleWarning(**w) for w in result["warnings"]
        ]

        return ScheduleResponse(
            project_id=project_id,
            project_name=project.name if project else "Unknown",
            start_date=result["start_date"],
            deadline=result["deadline"],
            milestones=milestones,
            sprints=sprints,
            tasks=tasks,
            warnings=warnings_list,
            feasible=result["feasible"],
            required_headcount=result.get("required_headcount"),
            expected_deadline=result.get("expected_deadline"),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Schedule generation failed for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Schedule generation failed: {exc}",
        )


@router.get(
    "/{project_id}/schedule",
    response_model=ScheduleResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_schedule(project_id: str) -> ScheduleResponse:
    try:
        project = await db.project.find_unique(
            where={"id": project_id},
            include={
                "projectInfo": True,
                "groupingTasks": {"include": {"tasks": True}},
            },
        )
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        milestones = await db.milestone.find_many(
            where={"projectId": project_id},
            order={"dueDate": "asc"},
        )

        if not milestones:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No schedule for project {project_id}. Call POST /{project_id}/schedule first.",
            )

        info = project.projectInfo
        start = info.startDate if (info and info.startDate) else milestones[0].dueDate
        deadline = info.deadline if (info and info.deadline) else milestones[-1].dueDate

        groups = project.groupingTasks or []
        sprints_list: list[SprintInfo] = []
        for g in groups:
            if g.sprintNumber is not None:
                gtasks = getattr(g, "tasks", None) or []
                sprints_list.append(
                    SprintInfo(
                        sprint_number=g.sprintNumber,
                        group_id=g.id,
                        group_name=g.name or "",
                        start_date=g.sprintStart,
                        end_date=g.sprintEnd,
                        total_effort_hours=round(
                            sum(t.effortHours or t.estimatedHours or 0 for t in gtasks), 2
                        ),
                        task_count=len(gtasks),
                    )
                )

        all_tasks: list[ScheduledTaskInfo] = []
        for g in groups:
            for t in (g.tasks or []):
                deps: Optional[list[str]] = None
                if t.dependencies:
                    try:
                        import json
                        deps = json.loads(t.dependencies)
                    except (json.JSONDecodeError, TypeError):
                        deps = [t.dependencies]
                all_tasks.append(
                    ScheduledTaskInfo(
                        task_id=t.id,
                        title=t.title,
                        group_id=t.groupId,
                        scheduled_start=t.scheduledStart,
                        scheduled_end=t.scheduledEnd,
                        duration_days=t.durationDays or 0,
                        dependencies=deps,
                    )
                )

        total_est = sum(g.totalDurationDays or 0 for g in groups)
        available = max(1, (deadline - start).days)
        feasible = total_est <= available

        import json
        req_hc = None
        exp_dl = None
        if project.summary:
            try:
                summ = json.loads(project.summary)
                if "schedule" in summ:
                    req_hc = summ["schedule"].get("required_headcount")
                    if summ["schedule"].get("expected_deadline"):
                        from datetime import datetime
                        exp_dl = datetime.fromisoformat(summ["schedule"]["expected_deadline"])
            except Exception:
                pass

        warnings_vec: list[ScheduleWarning] = []
        if not feasible:
            warnings_vec.append(
                ScheduleWarning(
                    category="timeline",
                    message=f"Estimated duration ({total_est:.1f}d) exceeds timeline ({available}d)",
                    severity="critical",
                )
            )

        return ScheduleResponse(
            project_id=project_id,
            project_name=project.name,
            start_date=start,
            deadline=deadline,
            milestones=[_milestone_to_response(m) for m in milestones],
            sprints=sprints_list,
            tasks=all_tasks,
            warnings=warnings_vec,
            feasible=feasible,
            required_headcount=req_hc,
            expected_deadline=exp_dl,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get schedule: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get schedule: {exc}",
        )


@router.delete(
    "/{project_id}/schedule",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
async def clear_schedule(project_id: str) -> None:
    try:
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

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

        project_summary = {}
        if project.summary:
            try:
                import json
                project_summary = json.loads(project.summary)
            except (json.JSONDecodeError, TypeError):
                project_summary = {}
        project_summary.pop("schedule", None)
        import json
        await db.project.update(
            where={"id": project_id},
            data={"summary": json.dumps(project_summary) if project_summary else None},
        )

        logger.info("Cleared schedule for project %s", project_id)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to clear schedule: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to clear schedule: {exc}",
        )
