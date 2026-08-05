import logging

from fastapi import APIRouter, HTTPException, status

from app.db.prisma import db
from app.models.schemas import ContextResponse, ErrorResponse
from app.services.context_engine import (
    ProjectNotFoundError,
    generate_context,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/{project_id}/context",
    response_model=ContextResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def generate_project_context(project_id: str) -> ContextResponse:
    try:
        summary = await generate_context(project_id)
        project = await db.project.find_unique(where={"id": project_id})

        return ContextResponse(
            project_id=project_id,
            project_name=project.name if project else "Unknown",
            summary=summary,
        )

    except ProjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Context generation failed for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Context generation failed: {exc}",
        )


@router.get(
    "/{project_id}/context",
    response_model=ContextResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_project_context(project_id: str) -> ContextResponse:
    try:
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        if not project.summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No context generated for project {project_id}. Call POST /{project_id}/context first.",
            )

        import json

        summary = json.loads(project.summary)

        return ContextResponse(
            project_id=project_id,
            project_name=project.name,
            summary=summary,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get context for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get context: {exc}",
        )
