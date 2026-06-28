import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.db.prisma import db
from app.models.schemas import (
    ConfidenceReport,
    ErrorResponse,
    PlanningWarningInfo,
    RiskAssessmentInfo,
    ValidationRequest,
    ValidationResponse,
)
from app.services.validation_engine import run_validation

logger = logging.getLogger(__name__)

router = APIRouter()


def _warning_from_dict(w: dict[str, str]) -> PlanningWarningInfo:
    return PlanningWarningInfo(
        category=w.get("category", "general"),
        message=w.get("message", ""),
        severity=w.get("severity", "medium"),
    )


def _risk_from_dict(r: dict[str, Any]) -> RiskAssessmentInfo:
    return RiskAssessmentInfo(
        risk_id=r.get("risk_id", ""),
        title=r.get("title", ""),
        type=r.get("type", "risk"),
        severity=r.get("severity", "medium"),
        status=r.get("status", "open"),
        confidence_impact=r.get("confidence_impact", 0),
        warning=r.get("warning"),
    )


@router.post(
    "/{project_id}/validate",
    response_model=ValidationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def validate_project(
    project_id: str,
    body: ValidationRequest = ValidationRequest(),
) -> ValidationResponse:
    try:
        result = await run_validation(project_id, reindex=body.reindex)
        project = await db.project.find_unique(where={"id": project_id})

        confidence = ConfidenceReport(**result["confidence"])
        warnings = [_warning_from_dict(w) for w in result["warnings"]]
        risk_assessments = [_risk_from_dict(r) for r in result["risk_assessments"]]

        return ValidationResponse(
            project_id=project_id,
            project_name=project.name if project else "Unknown",
            confidence=confidence,
            warnings=warnings,
            risk_assessments=risk_assessments,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Validation failed for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation failed: {exc}",
        )


@router.get(
    "/{project_id}/validate",
    response_model=ValidationResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_validation(project_id: str) -> ValidationResponse:
    try:
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        risk_blockages = await db.riskblockage.find_many(
            where={"projectId": project_id},
            order={"createdAt": "desc"},
        )

        if not risk_blockages:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No validation data for project {project_id}. Call POST /{project_id}/validate first.",
            )

        # Build confidence from stored project summary if available
        project_summary = {}
        if project.summary:
            try:
                import json
                project_summary = json.loads(project.summary)
            except (json.JSONDecodeError, TypeError):
                pass

        val_data = project_summary.get("validation", {})
        comp = val_data.get("components", {})
        confidence = ConfidenceReport(
            score=val_data.get("confidence_score", 0),
            requirement_completeness=comp.get("requirement_completeness", 0),
            document_coverage=comp.get("document_coverage", 0),
            dependency_clarity=comp.get("dependency_clarity", 0),
            timeline_feasibility=comp.get("timeline_feasibility", 0),
            technology_certainty=comp.get("technology_certainty", 0),
        )

        risk_assessments = [
            RiskAssessmentInfo(
                risk_id=r.id,
                title=r.title,
                type=r.type,
                severity=r.severity,
                status=r.status,
                confidence_impact=r.confidenceScore or 0,
                warning=r.planningWarnings,
            )
            for r in risk_blockages
        ]

        return ValidationResponse(
            project_id=project_id,
            project_name=project.name,
            confidence=confidence,
            warnings=[],
            risk_assessments=risk_assessments,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get validation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get validation: {exc}",
        )


@router.delete(
    "/{project_id}/validate",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
async def clear_validation(project_id: str) -> None:
    try:
        project = await db.project.find_unique(where={"id": project_id})
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        await db.riskblockage.update_many(
            where={"projectId": project_id},
            data={"confidenceScore": 0, "planningWarnings": None},
        )

        project_summary = {}
        if project.summary:
            try:
                import json
                project_summary = json.loads(project.summary)
            except (json.JSONDecodeError, TypeError):
                project_summary = {}
        project_summary.pop("validation", None)
        import json
        await db.project.update(
            where={"id": project_id},
            data={"summary": json.dumps(project_summary) if project_summary else None},
        )

        logger.info("Cleared validation for project %s", project_id)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to clear validation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to clear validation: {exc}",
        )
