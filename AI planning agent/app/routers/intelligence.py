import logging

from fastapi import APIRouter, HTTPException, status

from app.db.prisma import db, clear_project_cache
from app.services.intelligence_engine import generate_project_intelligence

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/projects/{project_id}/intelligence")


@router.post("/generate")
async def generate_intelligence(project_id: str):
    try:
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )
        # Clear stale cache before generation
        clear_project_cache(project_id)
        result = await generate_project_intelligence(project_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Intelligence generation failed for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Intelligence generation failed: {exc}",
        )


@router.get("/")
async def get_intelligence(project_id: str):
    try:
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )
        record = await db.projectintelligence.find_unique(
            where={"projectId": project_id}
        )
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No intelligence record found for project {project_id}",
            )
        return record
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get intelligence for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get intelligence: {exc}",
        )
