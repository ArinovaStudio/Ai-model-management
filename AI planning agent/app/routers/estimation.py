import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status

from app.db.prisma import db
from app.models.schemas import (
    ErrorResponse,
    EstimationRequest,
    EstimationResponse,
    EstimationSummary,
    GroupEstimateResponse,
    TaskEstimateResponse,
)
from app.routers.wbs import verify_wbs_approved
from app.services.estimation_engine import run_estimation

logger = logging.getLogger(__name__)

router = APIRouter()


async def _build_estimation_response(project_id: str, result: dict[str, Any]) -> EstimationResponse:
    project = await db.project.find_unique(where={"id": project_id})
    project_name = project.name if project else "Unknown"

    groups = []
    for g in result.get("groups", []):
        tasks = [
            TaskEstimateResponse(
                task_id=t["task_id"],
                task_title=t.get("task_title", ""),
                order=t.get("order", 0),
                complexity_score=t["complexity_score"],
                effort_hours=t["effort_hours"],
                duration_days=t["duration_days"],
                estimated_hours=t.get("estimated_hours", 0),
                dependencies=t.get("dependencies"),
            )
            for t in g.get("tasks", [])
        ]
        groups.append(
            GroupEstimateResponse(
                group_id=g["group_id"],
                group_name=g.get("group_name", ""),
                order=g.get("order", 0),
                total_effort_hours=g["total_effort_hours"],
                total_duration_days=g["total_duration_days"],
                avg_complexity_score=g["avg_complexity_score"],
                task_count=g.get("task_count", len(tasks)),
                tasks=tasks,
            )
        )

    summary_data = result.get("summary", {})
    summary = EstimationSummary(
        total_effort_hours=summary_data.get("total_effort_hours", 0),
        total_duration_days=summary_data.get("total_duration_days", 0),
        avg_complexity_score=summary_data.get("avg_complexity_score", 0),
        total_tasks=summary_data.get("total_tasks", 0),
        total_groups=summary_data.get("total_groups", 0),
    )

    return EstimationResponse(
        project_id=project_id,
        project_name=project_name,
        summary=summary,
        groups=groups,
    )


@router.post(
    "/{project_id}/estimate",
    response_model=EstimationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def estimate_project(
    project_id: str,
    body: EstimationRequest = EstimationRequest(),
) -> EstimationResponse:
    try:
        await verify_wbs_approved(project_id, body.is_pipeline_run)
        result = await run_estimation(
            project_id, 
            reindex=body.reindex,
            productive_hours_per_day=body.productive_hours_per_day or 6.0
        )
        return await _build_estimation_response(project_id, result)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Estimation failed for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Estimation failed: {exc}",
        )


@router.get(
    "/{project_id}/estimate",
    response_model=EstimationResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_estimation(project_id: str) -> EstimationResponse:
    try:
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        from app.db.redis import get_cached_wbs_estimates, set_cached_wbs_estimates
        cached = await get_cached_wbs_estimates(project_id)
        if cached is not None:
            return cached

        groups = await db.groupingtask.find_many(
            where={"projectId": project_id},
            include={"tasks": True},
            order={"order": "asc"},
        )

        if not groups:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No WBS data for project {project_id}. Run POST /{project_id}/estimate first.",
            )

        result_groups = []
        for g in groups:
            tasks_list = getattr(g, "tasks", None) or []
            tasks_list = sorted(tasks_list, key=lambda x: x.order or 0)
            deps_map = {}
            for t in tasks_list:
                deps: Optional[list[str]] = None
                if t.dependencies:
                    try:
                        deps = json.loads(t.dependencies)
                    except (json.JSONDecodeError, TypeError):
                        deps = [t.dependencies]
                deps_map[t.id] = deps

            tasks = [
                TaskEstimateResponse(
                    task_id=t.id,
                    task_title=t.title,
                    order=t.order or 0,
                    complexity_score=t.complexityScore or 0,
                    effort_hours=t.effortHours or 0,
                    duration_days=t.durationDays or 0,
                    estimated_hours=t.estimatedHours or 0,
                    dependencies=deps_map.get(t.id),
                )
                for t in tasks_list
            ]
            result_groups.append(
                GroupEstimateResponse(
                    group_id=g.id,
                    group_name=g.name or "",
                    order=g.order or 0,
                    total_effort_hours=g.totalEffortHours or 0,
                    total_duration_days=g.totalDurationDays or 0,
                    avg_complexity_score=g.avgComplexityScore or 0,
                    task_count=len(tasks),
                    tasks=tasks,
                )
            )

        total_effort = sum(g.totalEffortHours or 0 for g in groups)
        total_duration = sum(g.totalDurationDays or 0 for g in groups)
        all_scores = [t.complexityScore or 0 for g in groups for t in (g.tasks or [])]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

        summary = EstimationSummary(
            total_effort_hours=total_effort,
            total_duration_days=total_duration,
            avg_complexity_score=avg_score,
            total_tasks=len(all_scores),
            total_groups=len(groups),
        )

        resp = EstimationResponse(
            project_id=project_id,
            project_name=project.name,
            summary=summary,
            groups=result_groups,
        )
        await set_cached_wbs_estimates(project_id, resp)
        return resp

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get estimation for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get estimation: {exc}",
        )


@router.delete(
    "/{project_id}/estimate",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
async def clear_estimation(project_id: str) -> None:
    try:
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        groups = await db.groupingtask.find_many(
            where={"projectId": project_id},
            include={"tasks": True},
        )
        for g in groups:
            for t in (g.tasks or []):
                await db.task.update(
                    where={"id": t.id},
                    data={
                        "complexityScore": 0,
                        "effortHours": 0,
                        "durationDays": 0,
                    },
                )
            await db.groupingtask.update(
                where={"id": g.id},
                data={
                    "totalEffortHours": 0,
                    "totalDurationDays": 0,
                    "avgComplexityScore": 0,
                },
            )

        project_summary = {}
        if project.summary:
            try:
                project_summary = json.loads(project.summary)
            except (json.JSONDecodeError, TypeError):
                project_summary = {}
        project_summary.pop("effort_estimation", None)
        await db.project.update(
            where={"id": project_id},
            data={"summary": json.dumps(project_summary) if project_summary else None},
        )

        logger.info("Cleared estimation for project %s", project_id)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to clear estimation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to clear estimation: {exc}",
        )
