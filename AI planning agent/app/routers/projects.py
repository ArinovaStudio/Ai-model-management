import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.db.prisma import db, clear_project_cache
from app.models.schemas import (
    ErrorResponse,
    ProjectCreate,
    ProjectInfoUpdate,
    ProjectResponse,
)
from app.services.context_engine import generate_context

logger = logging.getLogger(__name__)

router = APIRouter()


def _project_to_response(project: Any) -> ProjectResponse:
    summary_data: dict[str, Any] | None = None
    if project.summary:
        try:
            summary_data = json.loads(project.summary)
        except (json.JSONDecodeError, TypeError):
            summary_data = {"raw": project.summary}

    project_info = None
    if project.projectInfo:
        info = project.projectInfo
        meta = None
        if info.metadata:
            try:
                meta = json.loads(info.metadata)
            except (json.JSONDecodeError, TypeError):
                meta = {"raw": info.metadata}

        project_info = {
            "id": info.id,
            "project_id": info.projectId,
            "name": info.name,
            "description": info.description,
            "start_date": info.startDate,
            "deadline": info.deadline,
            "scope": info.scope,
            "constraints": info.constraints,
            "metadata": meta,
            "created_at": info.createdAt,
            "updated_at": info.updatedAt,
        }

    plan_docs = getattr(project, "planDocs", None) or []
    docs = getattr(project, "docs", None) or []
    feature_requests = getattr(project, "featureRequests", None) or []
    technologies = getattr(project, "technologies", None) or []
    risk_blockages = getattr(project, "riskBlockages", None) or []
    resource_allocations = getattr(project, "resourceAllocations", None) or []
    grouping_tasks = getattr(project, "groupingTasks", None) or []

    wbs_tasks_count = 0
    total_effort = 0.0
    total_duration = 0.0
    all_scores: list[float] = []
    if grouping_tasks:
        for g in grouping_tasks:
            gtasks = getattr(g, "tasks", None) or []
            wbs_tasks_count += len(gtasks)
            total_effort += g.totalEffortHours or 0
            total_duration += g.totalDurationDays or 0
            for t in gtasks:
                if t.complexityScore is not None:
                    all_scores.append(float(t.complexityScore))
    avg_complexity = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return ProjectResponse(
        id=project.id,
        name=project.name,
        summary=summary_data,
        status=str(project.status) if project.status is not None else "unknown",
        project_info=project_info,
        plan_docs_count=len(plan_docs),
        docs_count=len(docs),
        feature_requests_count=len(feature_requests),
        technologies_count=len(technologies),
        risk_blockages_count=len(risk_blockages),
        grouping_tasks_count=len(grouping_tasks),
        wbs_tasks_count=wbs_tasks_count,
        resource_allocations_count=len(resource_allocations),
        total_effort_hours=round(total_effort, 2),
        total_duration_days=round(total_duration, 2),
        avg_complexity_score=round(avg_complexity, 2),
        wbs_status=project.wbsStatus,
        created_at=project.createdAt,
        updated_at=project.createdAt,
    )


@router.post(
    "",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def create_project(body: ProjectCreate) -> ProjectResponse:
    try:
        metadata_json = json.dumps(body.metadata) if body.metadata else None

        project = await db.project.create(
            data={
                "name": body.name,
                "status": 1,
                "projectInfo": {
                    "create": {
                        "name": body.name,
                        "description": body.description,
                        "startDate": body.start_date,
                        "deadline": body.deadline,
                        "metadata": metadata_json,
                        "budget": 0.0,
                        "projectType": "AI Planning",
                    }
                },
            },
            include={
                "projectInfo": True,
                "planDocs": True,
                "docs": True,
                "featureRequests": True,
                "technologies": True,
                "riskBlockages": True,
                "resourceAllocations": True,
                "groupingTasks": {"include": {"tasks": True}},
            },
        )

        logger.info("Created project %s with name '%s'", project.id, project.name)
        return _project_to_response(project)

    except Exception as exc:
        logger.error("Failed to create project: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create project: {exc}",
        )


@router.get(
    "",
    response_model=list[ProjectResponse],
    responses={400: {"model": ErrorResponse}},
)
async def list_projects() -> list[ProjectResponse]:
    try:
        projects = await db.project.find_many(
            include={
                "projectInfo": True,
                "groupingTasks": {"include": {"tasks": True}},
            },
            order={"createdAt": "desc"},
        )
        return [_project_to_response(p) for p in projects]

    except Exception as exc:
        logger.error("Failed to list projects: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to list projects: {exc}",
        )


@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_project(project_id: str) -> ProjectResponse:
    try:
        project = await db.project.find_unique(
            where={"id": project_id},
            include={
                "projectInfo": True,
                "planDocs": True,
                "docs": True,
                "featureRequests": True,
                "technologies": True,
                "riskBlockages": True,
                "resourceAllocations": True,
                "groupingTasks": {"include": {"tasks": True}},
            },
        )

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        return _project_to_response(project)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get project: {exc}",
        )


@router.put(
    "/{project_id}/info",
    response_model=ProjectResponse,
    responses={404: {"model": ErrorResponse}},
)
async def update_projectInfo(
    project_id: str,
    body: ProjectInfoUpdate,
) -> ProjectResponse:
    try:
        existing = await db.project.find_unique(
            where={"id": project_id},
            include={"projectInfo": True},
        )
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        update_data: dict[str, Any] = {}
        if body.name is not None:
            update_data["name"] = body.name
        if body.description is not None:
            update_data["description"] = body.description
        if body.start_date is not None:
            update_data["startDate"] = body.start_date
        if body.deadline is not None:
            update_data["deadline"] = body.deadline
        if body.scope is not None:
            update_data["scope"] = body.scope
        if body.constraints is not None:
            update_data["constraints"] = body.constraints
        if body.project_type is not None:
            update_data["projectType"] = body.project_type
        if body.metadata is not None:
            update_data["metadata"] = json.dumps(body.metadata)

        if update_data:
            if existing.projectInfo:
                await db.projectinfo.update(
                    where={"id": existing.projectInfo.id},
                    data=update_data,
                )
            else:
                update_data["project_id"] = project_id
                await db.projectinfo.create(data=update_data)
            clear_project_cache(project_id)

        project = await db.project.find_unique(
            where={"id": project_id},
            include={
                "projectInfo": True,
                "planDocs": True,
                "docs": True,
                "featureRequests": True,
                "technologies": True,
                "riskBlockages": True,
                "resourceAllocations": True,
                "groupingTasks": {"include": {"tasks": True}},
            },
        )

        return _project_to_response(project)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to update project info %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update project info: {exc}",
        )


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
async def delete_project(project_id: str) -> None:
    try:
        existing = await db.project.find_unique(where={"id": project_id})
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        await db.project.delete(where={"id": project_id})
        clear_project_cache(project_id)
        logger.info("Deleted project %s", project_id)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to delete project: {exc}",
        )
