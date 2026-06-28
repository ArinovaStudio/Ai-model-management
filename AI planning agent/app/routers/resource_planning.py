import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.db.prisma import db
from app.models.schemas import (
    ErrorResponse,
    ResourceAllocationResponse,
    ResourcePlanRequest,
    ResourcePlanResponse,
)
from app.services.resource_engine import run_resource_planning

logger = logging.getLogger(__name__)

router = APIRouter()


def _alloc_to_response(a: Any) -> ResourceAllocationResponse:
    return ResourceAllocationResponse(
        id=a.id,
        project_id=a.projectId,
        role=a.role,
        required_count=a.requiredCount,
        allocated_hours=a.allocatedHours,
        workload_percentage=a.workloadPercentage,
        recommended_count=a.recommendedCount,
        created_at=a.createdAt,
        updated_at=a.updatedAt,
    )


@router.post(
    "/{project_id}/resource-plan",
    response_model=ResourcePlanResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def generate_resource_plan(
    project_id: str,
    body: ResourcePlanRequest = ResourcePlanRequest(),
) -> ResourcePlanResponse:
    try:
        result = await run_resource_planning(project_id, reindex=body.reindex)
        project = await db.project.find_unique(where={"id": project_id})

        allocations = [_alloc_to_response(a) for a in result["allocations"]]

        return ResourcePlanResponse(
            project_id=project_id,
            project_name=project.name if project else "Unknown",
            allocations=allocations,
            total_team_size=result["total_team_size"],
            total_effort_hours=result["total_effort_hours"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Resource planning failed for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Resource planning failed: {exc}",
        )


@router.get(
    "/{project_id}/resource-plan",
    response_model=ResourcePlanResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_resource_plan(project_id: str) -> ResourcePlanResponse:
    try:
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        allocations = await db.resourceallocation.find_many(
            where={"projectId": project_id},
            order={"allocatedHours": "desc"},
        )

        if not allocations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No resource plan for project {project_id}. Call POST /{project_id}/resource-plan first.",
            )

        alloc_responses = [_alloc_to_response(a) for a in allocations]
        total_team_size = sum(a.requiredCount for a in allocations)
        total_effort = sum(a.allocatedHours for a in allocations)

        return ResourcePlanResponse(
            project_id=project_id,
            project_name=project.name,
            allocations=alloc_responses,
            total_team_size=total_team_size,
            total_effort_hours=total_effort,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get resource plan: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get resource plan: {exc}",
        )


@router.delete(
    "/{project_id}/resource-plan",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
async def clear_resource_plan(project_id: str) -> None:
    try:
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        await db.resourceallocation.delete_many(where={"projectId": project_id})

        project_summary = {}
        if project.summary:
            try:
                import json
                project_summary = json.loads(project.summary)
            except (json.JSONDecodeError, TypeError):
                project_summary = {}
        project_summary.pop("resource_plan", None)
        import json
        await db.project.update(
            where={"id": project_id},
            data={"summary": json.dumps(project_summary) if project_summary else None},
        )

        logger.info("Cleared resource plan for project %s", project_id)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to clear resource plan: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to clear resource plan: {exc}",
        )
