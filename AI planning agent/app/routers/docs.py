import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.db.prisma import db
from app.models.schemas import (
    DocCreate,
    DocResponse,
    ErrorResponse,
    PlanDocCreate,
    PlanDocResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/{project_id}/plan-docs",
    response_model=PlanDocResponse,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}},
)
async def create_plan_doc(project_id: str, body: PlanDocCreate) -> PlanDocResponse:
    try:
        existing = await db.project.find_unique(where={"id": project_id})
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        metadata_json = json.dumps(body.metadata) if body.metadata else None

        doc = await db.plandoc.create(
            data={
                "projectId": project_id,
                "title": body.title,
                "content": body.content,
                "metadata": metadata_json,
            },
        )

        logger.info(
            "Created PlanDoc %s for project %s",
            doc.id,
            project_id,
        )
        return PlanDocResponse(
            id=doc.id,
            project_id=doc.projectId,
            title=doc.title,
            content=doc.content,
            metadata=json.loads(doc.metadata) if doc.metadata else None,
            created_at=doc.createdAt,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to create PlanDoc: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create PlanDoc: {exc}",
        )


@router.get(
    "/{project_id}/plan-docs",
    response_model=list[PlanDocResponse],
    responses={404: {"model": ErrorResponse}},
)
async def list_plan_docs(project_id: str) -> list[PlanDocResponse]:
    try:
        existing = await db.project.find_unique(where={"id": project_id})
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        docs = await db.plandoc.find_many(
            where={"projectId": project_id},
            order={"createdAt": "desc"},
        )

        return [
            PlanDocResponse(
                id=doc.id,
                project_id=doc.projectId,
                title=doc.title,
                content=doc.content,
                metadata=json.loads(doc.metadata) if doc.metadata else None,
                created_at=doc.createdAt,
            )
            for doc in docs
        ]

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list PlanDocs: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to list PlanDocs: {exc}",
        )


@router.post(
    "/{project_id}/docs",
    response_model=DocResponse,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}},
)
async def create_doc(project_id: str, body: DocCreate) -> DocResponse:
    try:
        existing = await db.project.find_unique(where={"id": project_id})
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        metadata_json = json.dumps(body.metadata) if body.metadata else None

        doc = await db.doc.create(
            data={
                "projectId": project_id,
                "title": body.title,
                "content": body.content,
                "docType": body.doc_type,
                "metadata": metadata_json,
            },
        )

        logger.info(
            "Created Doc %s (type=%s) for project %s",
            doc.id,
            doc.docType,
            project_id,
        )
        return DocResponse(
            id=doc.id,
            project_id=doc.projectId,
            title=doc.title,
            content=doc.content,
            doc_type=doc.docType,
            metadata=json.loads(doc.metadata) if doc.metadata else None,
            created_at=doc.createdAt,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to create Doc: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create Doc: {exc}",
        )


@router.get(
    "/{project_id}/docs",
    response_model=list[DocResponse],
    responses={404: {"model": ErrorResponse}},
)
async def list_docs(project_id: str) -> list[DocResponse]:
    try:
        existing = await db.project.find_unique(where={"id": project_id})
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        docs = await db.doc.find_many(
            where={"projectId": project_id},
            order={"createdAt": "desc"},
        )

        return [
            DocResponse(
                id=doc.id,
                project_id=doc.projectId,
                title=doc.title,
                content=doc.content,
                doc_type=doc.docType,
                metadata=json.loads(doc.metadata) if doc.metadata else None,
                created_at=doc.createdAt,
            )
            for doc in docs
        ]

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list Docs: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to list Docs: {exc}",
        )
