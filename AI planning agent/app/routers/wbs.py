import json
import logging
import asyncio
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse

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


async def verify_wbs_approved(project_id: str, is_pipeline_run: bool = False):
    if is_pipeline_run:
        return
    project = await db.project.find_unique(where={"id": project_id})
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if project.wbsStatus != "APPROVED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Action blocked: The Work Breakdown Structure (WBS) must be approved before running this stage.",
        )


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
    tasks = sorted(tasks, key=lambda x: x.order or 0)
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
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def generate_wbs_endpoint(
    project_id: str,
    body: WbsRequest = WbsRequest(),
):
    async def keep_alive_generator():
        # Run generate_wbs as a background task to allow yielding
        task = asyncio.create_task(
            generate_wbs(
                project_id, 
                reindex=body.reindex,
                sprint_duration_days=body.sprint_duration_days or 14,
                productive_hours_per_day=body.productive_hours_per_day or 6.0
            )
        )
        
        while not task.done():
            yield b" "  # Send a space character to keep the browser connection alive
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=15.0)
            except asyncio.TimeoutError:
                pass
                
        try:
            result = task.result()
        except ValueError as exc:
            yield json.dumps({"error": str(exc), "status_code": 404}).encode('utf-8')
            return
        except Exception as exc:
            logger.error("WBS generation failed for project %s: %s", project_id, exc)
            yield json.dumps({"error": f"WBS generation failed: {exc}", "status_code": 422}).encode('utf-8')
            return

        project = await db.project.update(
            where={"id": project_id},
            data={"wbsStatus": "DRAFT"}
        )
        groups = [_group_to_response(g) for g in result["groups"]]

        response_model = WbsResponse(
            project_id=project_id,
            project_name=project.name if project else "Unknown",
            groups=groups,
            total_groups=result["total_groups"],
            total_tasks=result["total_tasks"],
            wbs_status="DRAFT",
        )
        
        # Yield the actual JSON response
        yield response_model.model_dump_json().encode('utf-8')

    return StreamingResponse(keep_alive_generator(), media_type="application/json")


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
            wbs_status=project.wbsStatus,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get WBS for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get WBS: {exc}",
        )


@router.post(
    "/{project_id}/wbs/approve",
    response_model=dict[str, str],
    responses={404: {"model": ErrorResponse}},
)
async def approve_wbs(project_id: str) -> dict[str, str]:
    try:
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        # Update status in DB
        await db.project.update(
            where={"id": project_id},
            data={"wbsStatus": "APPROVED"}
        )

        # Trigger schedule and validation engines automatically to generate initial timelines
        try:
            from app.services.schedule_engine import generate_schedule
            from app.services.validation_engine import run_validation
            await generate_schedule(project_id, reindex=True)
            await run_validation(project_id)
        except Exception as schedule_err:
            logger.warning("Failed to auto-generate schedule/validation upon WBS approval: %s", schedule_err)

        return {"wbs_status": "APPROVED", "message": "WBS plan approved successfully."}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to approve WBS for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to approve WBS: {exc}",
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

        where: dict[str, Any] = {"projectId": project_id}
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

        # Recalculate effort and complexity if any planning attributes are changed
        if body.estimated_hours is not None or body.dependencies is not None or body.title is not None or body.description is not None:
            est_hours = body.estimated_hours if body.estimated_hours is not None else (existing.estimatedHours or 4.0)
            deps_val = body.dependencies if body.dependencies is not None else json.loads(existing.dependencies or "[]")
            title_val = body.title if body.title is not None else (existing.title or "")
            desc_val = body.description if body.description is not None else (existing.description or "")
            
            from app.services.estimation_engine import _compute_task_complexity, _compute_effort
            complexity = _compute_task_complexity(
                title=title_val,
                description=desc_val,
                dependency_count=len(deps_val)
            )
            # Default productive hours per day is 6.0
            effort, duration = _compute_effort(
                float(est_hours),
                complexity,
                6.0
            )
            update_data["complexityScore"] = float(complexity)
            update_data["effortHours"] = float(effort)
            update_data["durationDays"] = float(duration)

        if not update_data:
            return _task_to_response(existing)

        updated = await db.task.update(
            where={"id": task_id},
            data=update_data,
            include={"featureRequest": True},
        )

        # Sync downstream schedule
        try:
            from app.services.schedule_engine import generate_schedule
            from app.services.validation_engine import run_validation
            await generate_schedule(project_id, reindex=True)
            await run_validation(project_id)
        except Exception as schedule_err:
            logger.warning("Failed to auto-reschedule after manual WBS edit: %s", schedule_err)

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

        has_fr = await db.featurerequest.count(where={"projectId": project_id}) > 0
        delete_where = {"projectId": project_id}
        if has_fr:
            delete_where["featureRequestId"] = {"not": None}
        await db.task.delete_many(where=delete_where)
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
