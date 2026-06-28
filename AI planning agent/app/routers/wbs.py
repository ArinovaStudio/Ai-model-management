import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.db.prisma import db
from app.models.schemas import (
    ErrorResponse,
    WbsGroupResponse,
    WbsRequest,
    WbsResponse,
    WbsTaskResponse,
    WbsTaskUpdate,
)
from app.services.wbs_engine import generate_wbs

logger = logging.getLogger(__name__)

router = APIRouter()


def _task_to_response(t: Any) -> WbsTaskResponse:
    deps: Optional[list[str]] = None
    if t.dependencies:
        try:
            deps = json.loads(t.dependencies)
        except (json.JSONDecodeError, TypeError):
            deps = [t.dependencies]

    feat_title: Optional[str] = None
    if t.featureRequestId:
        feat = getattr(t, "featureRequest", None)
        if feat:
            feat_title = feat.title

    return WbsTaskResponse(
        id=t.id,
        title=t.title,
        description=t.description,
        order=t.order,
        status=t.status,
        priority=t.priority,
        dependencies=deps,
        estimated_hours=t.estimatedHours,
        complexity_score=t.complexityScore,
        effort_hours=t.effortHours,
        duration_days=t.durationDays,
        feature_request_id=t.featureRequestId,
        feature_request_title=feat_title,
        group_id=t.groupId,
        created_at=t.createdAt,
        updated_at=t.updatedAt,
    )


def _group_to_response(g: Any) -> WbsGroupResponse:
    feature_ids: Optional[list[str]] = None
    if g.featureIds:
        try:
            feature_ids = json.loads(g.featureIds)
        except (json.JSONDecodeError, TypeError):
            feature_ids = [g.featureIds]

    tasks = getattr(g, "tasks", None) or []
    return WbsGroupResponse(
        id=g.id,
        name=g.name or "",
        description=g.description,
        order=g.order,
        feature_ids=feature_ids,
        tasks=[_task_to_response(t) for t in tasks],
        total_effort_hours=g.totalEffortHours,
        total_duration_days=g.totalDurationDays,
        avg_complexity_score=g.avgComplexityScore,
        created_at=g.createdAt,
        updated_at=g.updatedAt,
    )


@router.post(
    "/{project_id}/wbs",
    response_model=WbsResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def generate_wbs_endpoint(
    project_id: str,
    body: WbsRequest = WbsRequest(),
) -> WbsResponse:
    try:
        result = await generate_wbs(project_id, reindex=body.reindex)
        project = await db.project.find_unique(where={"id": project_id})

        groups = [_group_to_response(g) for g in result["groups"]]

        return WbsResponse(
            project_id=project_id,
            project_name=project.name if project else "Unknown",
            groups=groups,
            total_groups=result["total_groups"],
            total_tasks=result["total_tasks"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("WBS generation failed for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"WBS generation failed: {exc}",
        )


@router.get(
    "/{project_id}/wbs",
    response_model=WbsResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_wbs(project_id: str) -> WbsResponse:
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
            order={"order": "asc"},
        )

        if not groups:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No WBS exists for project {project_id}. Call POST /{project_id}/wbs first.",
            )

        wbs_groups = [_group_to_response(g) for g in groups]
        total_tasks = sum(len(g.tasks or []) for g in groups)

        return WbsResponse(
            project_id=project_id,
            project_name=project.name,
            groups=wbs_groups,
            total_groups=len(wbs_groups),
            total_tasks=total_tasks,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get WBS for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get WBS: {exc}",
        )


@router.get(
    "/{project_id}/wbs/groups",
    response_model=list[WbsGroupResponse],
    responses={404: {"model": ErrorResponse}},
)
async def list_wbs_groups(project_id: str) -> list[WbsGroupResponse]:
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
            order={"order": "asc"},
        )

        return [_group_to_response(g) for g in groups]

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list WBS groups: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to list WBS groups: {exc}",
        )


@router.get(
    "/{project_id}/wbs/tasks",
    response_model=list[WbsTaskResponse],
    responses={404: {"model": ErrorResponse}},
)
async def list_wbs_tasks(
    project_id: str,
    group_id: str = Query(default=None),
    status_filter: str = Query(default=None, alias="status"),
) -> list[WbsTaskResponse]:
    try:
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        where: dict[str, Any] = {"projectId": project_id, "featureRequestId": {"not": None}}
        if group_id:
            where["groupId"] = group_id
        if status_filter:
            where["status"] = status_filter

        tasks = await db.task.find_many(
            where=where,
            include={"featureRequest": True},
            order={"order": "asc"},
        )

        return [_task_to_response(t) for t in tasks]

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list WBS tasks: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to list WBS tasks: {exc}",
        )


@router.patch(
    "/{project_id}/wbs/tasks/{task_id}",
    response_model=WbsTaskResponse,
    responses={404: {"model": ErrorResponse}},
)
async def update_wbs_task(
    project_id: str,
    task_id: str,
    body: WbsTaskUpdate,
) -> WbsTaskResponse:
    try:
        existing = await db.task.find_unique(where={"id": task_id})
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found",
            )
        if existing.projectId != project_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Task does not belong to this project",
            )

        update_data: dict[str, Any] = {}
        if body.title is not None:
            update_data["title"] = body.title
        if body.description is not None:
            update_data["description"] = body.description
        if body.status is not None:
            update_data["status"] = body.status
        if body.priority is not None:
            update_data["priority"] = body.priority
        if body.assignee is not None:
            update_data["assignee"] = body.assignee
        if body.estimated_hours is not None:
            update_data["estimatedHours"] = body.estimated_hours
        if body.dependencies is not None:
            update_data["dependencies"] = json.dumps(body.dependencies)

        if not update_data:
            return _task_to_response(existing)

        updated = await db.task.update(
            where={"id": task_id},
            data=update_data,
            include={"featureRequest": True},
        )

        return _task_to_response(updated)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to update WBS task %s: %s", task_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update task: {exc}",
        )


@router.delete(
    "/{project_id}/wbs",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
async def clear_wbs(project_id: str) -> None:
    try:
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        await db.task.delete_many(
            where={"projectId": project_id, "featureRequestId": {"not": None}},
        )
        await db.groupingtask.delete_many(where={"projectId": project_id})
        logger.info("Cleared WBS for project %s", project_id)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to clear WBS: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to clear WBS: {exc}",
        )
