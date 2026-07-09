import json
import logging
import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.db.prisma import db
from app.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ClarificationInput,
    ErrorResponse,
    FeatureRequestResponse,
    GapAnalysisReport,
    GapAnalysisRequest,
    GapAnalysisResponse,
    ProjectTechnologyResponse,
    RiskBlockageResponse,
    SemanticSearchResponse,
    SemanticSearchResult,
)
from app.services.analysis_engine import (
    ProjectNotFoundError as AnalysisProjectNotFoundError,
    analyze_project,
    clarify_requirement,
)
from app.services.embedding import embedding_service
from app.services.ollama_service import ollama_service
from app.services.gap_engine import (
    ProjectNotFoundError as GapProjectNotFoundError,
    perform_gap_analysis,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _feature_to_response(f: Any) -> FeatureRequestResponse:
    meta = None
    if f.metadata:
        try:
            meta = json.loads(f.metadata)
        except (json.JSONDecodeError, TypeError):
            meta = {"raw": f.metadata}

    return FeatureRequestResponse(
        id=f.id,
        project_id=f.projectId,
        title=f.title,
        description=f.description,
        priority=f.priority,
        status=f.status,
        category=f.category,
        source_doc_id=f.sourceDocId,
        source_doc_type=f.sourceDocType,
        is_ambiguous=f.isAmbiguous,
        clarification_prompt=f.clarificationPrompt,
        metadata=meta,
        created_at=f.createdAt,
        updated_at=f.updatedAt,
    )


def _tech_to_response(t: Any) -> ProjectTechnologyResponse:
    deps = None
    if t.dependencies:
        try:
            deps = json.loads(t.dependencies)
        except (json.JSONDecodeError, TypeError):
            deps = [t.dependencies]

    meta = None
    if t.metadata:
        try:
            meta = json.loads(t.metadata)
        except (json.JSONDecodeError, TypeError):
            meta = {"raw": t.metadata}

    return ProjectTechnologyResponse(
        id=t.id,
        project_id=t.projectId,
        name=t.name,
        category=t.category,
        version=t.version,
        is_detected=t.isDetected,
        dependencies=deps,
        metadata=meta,
        created_at=t.createdAt,
        updated_at=t.updatedAt,
    )


def _risk_to_response(r: Any) -> RiskBlockageResponse:
    meta = None
    if r.metadata:
        try:
            meta = json.loads(r.metadata)
        except (json.JSONDecodeError, TypeError):
            meta = {"raw": r.metadata}

    return RiskBlockageResponse(
        id=r.id,
        project_id=r.projectId,
        type=r.type,
        category=r.category,
        title=r.title,
        description=r.description,
        severity=r.severity,
        status=r.status,
        source_feature_id=r.sourceFeatureId,
        source_doc_id=r.sourceDocId,
        report_type=r.reportType,
        metadata=meta,
        confidence_score=r.confidenceScore,
        planning_warnings=r.planningWarnings,
        created_at=r.createdAt,
        updated_at=r.updatedAt,
    )


# --- Phase 2: Analysis endpoints ---


@router.post(
    "/{project_id}/analyze",
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def analyze_project_endpoint(
    project_id: str,
    body: AnalysisRequest = AnalysisRequest(),
):
    async def keep_alive_generator():
        # Run analyze_project as a background task
        task = asyncio.create_task(
            analyze_project(project_id, reindex=body.reindex)
        )
        
        while not task.done():
            yield b" "  # Keep-alive space
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=15.0)
            except asyncio.TimeoutError:
                pass

        try:
            result = task.result()
        except AnalysisProjectNotFoundError as exc:
            yield json.dumps({"error": str(exc), "status_code": 404}).encode('utf-8')
            return
        except Exception as exc:
            logger.error("Analysis failed for project %s: %s", project_id, exc)
            yield json.dumps({"error": f"Analysis failed: {exc}", "status_code": 422}).encode('utf-8')
            return

        project = await db.project.find_unique(where={"id": project_id})
        features_response = [_feature_to_response(f) for f in result["feature_requests"]]
        techs_response = [_tech_to_response(t) for t in result["technologies"]]
        ambiguous_response = [_feature_to_response(f) for f in result["ambiguous_requirements"]]

        response_model = AnalysisResponse(
            project_id=project_id,
            project_name=project.name if project else "Unknown",
            feature_requests=features_response,
            technologies=techs_response,
            ambiguous_requirements=ambiguous_response,
            clarification_prompts=result["clarification_prompts"],
        )
        yield response_model.model_dump_json().encode('utf-8')

    return StreamingResponse(keep_alive_generator(), media_type="application/json")


@router.get(
    "/{project_id}/features",
    response_model=list[FeatureRequestResponse],
)
async def list_features(project_id: str) -> list[FeatureRequestResponse]:
    try:
        existing = await db.project.find_unique(where={"id": project_id})
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        features = await db.featurerequest.find_many(
            where={"projectId": project_id},
            order={"createdAt": "desc"},
        )

        return [_feature_to_response(f) for f in features]

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list features for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to list features: {exc}",
        )


@router.get(
    "/{project_id}/technologies",
    response_model=list[ProjectTechnologyResponse],
)
async def list_technologies(project_id: str) -> list[ProjectTechnologyResponse]:
    try:
        existing = await db.project.find_unique(where={"id": project_id})
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        techs = await db.projecttechnology.find_many(
            where={"projectId": project_id},
            order={"createdAt": "desc"},
        )

        return [_tech_to_response(t) for t in techs]

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list technologies for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to list technologies: {exc}",
        )


@router.post(
    "/{project_id}/clarify",
    response_model=FeatureRequestResponse,
    responses={404: {"model": ErrorResponse}},
)
async def clarify_feature_endpoint(
    project_id: str,
    body: ClarificationInput,
) -> FeatureRequestResponse:
    try:
        feat = await db.featurerequest.find_unique(where={"id": body.feature_id})
        if not feat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"FeatureRequest {body.feature_id} not found",
            )
        if feat.projectId != project_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="FeatureRequest does not belong to this project",
            )

        updated = await clarify_requirement(body.feature_id, body.clarification)
        return _feature_to_response(updated)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Clarification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Clarification failed: {exc}",
        )


@router.get(
    "/{project_id}/ambiguous",
    response_model=list[FeatureRequestResponse],
)
async def list_ambiguous_features(project_id: str) -> list[FeatureRequestResponse]:
    try:
        existing = await db.project.find_unique(where={"id": project_id})
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        features = await db.featurerequest.find_many(
            where={"projectId": project_id, "isAmbiguous": True},
            order={"createdAt": "desc"},
        )

        return [_feature_to_response(f) for f in features]

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list ambiguous features: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to list ambiguous features: {exc}",
        )


@router.get(
    "/{project_id}/search",
    response_model=SemanticSearchResponse,
)
async def semantic_search_endpoint(
    project_id: str,
    query: str = Query(..., min_length=1),
    k: int = Query(default=5, ge=1, le=50),
    doc_type: str = Query(default=None),
) -> SemanticSearchResponse:
    try:
        from app.db.prisma import get_cached_project, set_cached_project
        project = get_cached_project(project_id)
        if not project:
            project = await db.project.find_unique(
                where={"id": project_id},
                include={
                    "projectInfo": True,
                    "technologies": True,
                    "projectIntelligence": True,
                }
            )
            if project:
                set_cached_project(project_id, project)

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        results = await embedding_service.semantic_search(
            project_id=project_id,
            query=query,
            k=k,
            doc_type=doc_type or None,
        )

        # Build tech stack details
        tech_list = []
        if project.technologies:
            for t in project.technologies:
                tech_list.append(f"- {t.name} ({t.category or 'general'})")
        tech_stack_str = "\n".join(tech_list) if tech_list else "No explicit tech stack detected."

        # Build architecture details
        arch_details = "No system architecture defined yet."
        db_details = "No database ERD defined yet."
        if project.projectIntelligence:
            intel = project.projectIntelligence
            if intel.systemArchitecture:
                try:
                    system_arch = json.loads(str(intel.systemArchitecture)) if isinstance(intel.systemArchitecture, str) else intel.systemArchitecture
                    arch_details = system_arch.get("highLevel", "No high level description.")
                except Exception:
                    arch_details = str(intel.systemArchitecture)
            if intel.databaseDesign:
                try:
                    db_design = json.loads(str(intel.databaseDesign)) if isinstance(intel.databaseDesign, str) else intel.databaseDesign
                    db_details = db_design.get("normalization", "No DB normalization info.")
                except Exception:
                    db_details = str(intel.databaseDesign)

        # Build project basic context
        proj_info_str = "No description provided."
        if project.projectInfo:
            proj_info_str = f"Description: {project.projectInfo.description or ''}\nScope: {project.projectInfo.scope or ''}"

        # Combine all structural context
        structural_context = (
            f"=== Project Configuration ===\n"
            f"Project Name: {project.name}\n"
            f"{proj_info_str}\n\n"
            f"=== Active Technology Stack ===\n"
            f"{tech_stack_str}\n\n"
            f"=== System Architecture Design ===\n"
            f"{arch_details}\n\n"
            f"=== Database Normalization & ERD Schema ===\n"
            f"{db_details}\n"
        )

        context_str = ""
        if results:
            context_str = "\n\n".join(
                f"Document: {r['title']}\nSnippet: {r['content_snippet']}"
                for r in results
            )
        else:
            context_str = "No specific document context matches found."

        rag_system_prompt = (
            "You are an Expert Project Architect and Brainstorming Assistant.\n"
            "Your goal is to answer the user's question with high technical accuracy. You are provided with two layers of context:\n"
            "1) Document Context: Excerpts from project documents (if available).\n"
            "2) Project Blueprint: The generated tech stack, system architecture, database design, and project scope.\n\n"
            "CRITICAL DIRECTIVES:\n"
            "- If the user's question asks about something NOT explicitly mentioned in the documents (like deployment stack, libraries, or security mechanisms), "
            "you MUST use the Project Blueprint as your grounding base, and project/recommend solutions using your general engineering expertise.\n"
            "- Ensure your recommendations are 100% compatible with the Active Technology Stack and System Architecture design.\n"
            "- Do NOT invent client constraints, but feel free to recommend best-practice technical implementation details.\n"
            "- Keep the response technical, direct, and conversational. Do not reference prompt structures or repeat the prompt back to the user."
        )

        answer = await ollama_service.chat_rag(
            system_prompt=rag_system_prompt,
            user_query=(
                f"=== Project Blueprint ===\n{structural_context}\n\n"
                f"=== Document Context ===\n{context_str}\n\n"
                f"Question: {query}"
            )
        )

        return SemanticSearchResponse(
            query=query,
            results=[SemanticSearchResult(**r) for r in results],
            answer=answer,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Semantic search failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Semantic search failed: {exc}",
        )


@router.delete(
    "/{project_id}/features",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def clear_analysis(project_id: str) -> None:
    try:
        existing = await db.project.find_unique(where={"id": project_id})
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        await db.featurerequest.delete_many(where={"projectId": project_id})
        await db.projecttechnology.delete_many(where={"projectId": project_id})
        logger.info("Cleared analysis data for project %s", project_id)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to clear analysis: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to clear analysis: {exc}",
        )


# --- Phase 3: Gap Analysis endpoints ---


@router.post(
    "/{project_id}/gap-analysis",
    response_model=GapAnalysisResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def run_gap_analysis(
    project_id: str,
    body: GapAnalysisRequest = GapAnalysisRequest(),
) -> GapAnalysisResponse:
    try:
        result = await perform_gap_analysis(project_id, reindex=body.reindex)
        project = await db.project.find_unique(where={"id": project_id})

        assumptions = [_risk_to_response(r) for r in result["assumptions"]]
        clarifications = [_risk_to_response(r) for r in result["clarifications"]]
        risks = [_risk_to_response(r) for r in result["risks"]]

        return GapAnalysisResponse(
            project_id=project_id,
            project_name=project.name if project else "Unknown",
            reports=GapAnalysisReport(
                assumptions=assumptions,
                clarifications=clarifications,
                risks=risks,
            ),
            summary=result["summary"],
        )

    except GapProjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Gap analysis failed for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Gap analysis failed: {exc}",
        )


@router.get(
    "/{project_id}/gap-analysis",
    response_model=GapAnalysisResponse,
)
async def get_gap_analysis(project_id: str) -> GapAnalysisResponse:
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
                detail=f"No gap analysis exists for project {project_id}. Call POST /{project_id}/gap-analysis first.",
            )

        assumptions = [_risk_to_response(r) for r in risk_blockages if r.reportType == "assumption"]
        clarifications = [_risk_to_response(r) for r in risk_blockages if r.reportType == "clarification"]
        risks = [_risk_to_response(r) for r in risk_blockages if r.reportType == "risk"]

        summary_parts = [
            f"{len(assumptions)} assumptions",
            f"{len(clarifications)} clarifications",
            f"{len(risks)} risks/blockages",
        ]
        summary = " | ".join(summary_parts)

        return GapAnalysisResponse(
            project_id=project_id,
            project_name=project.name,
            reports=GapAnalysisReport(
                assumptions=assumptions,
                clarifications=clarifications,
                risks=risks,
            ),
            summary=summary,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get gap analysis for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get gap analysis: {exc}",
        )


@router.get(
    "/{project_id}/risks",
    response_model=list[RiskBlockageResponse],
)
async def list_risk_blockages(
    project_id: str,
    report_type: str = Query(default=None),
    severity: str = Query(default=None),
) -> list[RiskBlockageResponse]:
    try:
        existing = await db.project.find_unique(where={"id": project_id})
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        where: dict[str, Any] = {"projectId": project_id}
        if report_type:
            where["reportType"] = report_type
        if severity:
            where["severity"] = severity

        records = await db.riskblockage.find_many(
            where=where,
            order={"severity": "asc"},
        )

        return [_risk_to_response(r) for r in records]

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list risk blockages: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to list risk blockages: {exc}",
        )


@router.patch(
    "/{project_id}/risks/{risk_id}",
    response_model=RiskBlockageResponse,
)
async def update_risk_status(
    project_id: str,
    risk_id: str,
    status: str = Query(..., pattern="^(open|resolved|mitigated)$"),
) -> RiskBlockageResponse:
    try:
        existing = await db.riskblockage.find_unique(where={"id": risk_id})
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"RiskBlockage {risk_id} not found",
            )
        if existing.projectId != project_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="RiskBlockage does not belong to this project",
            )

        updated = await db.riskblockage.update(
            where={"id": risk_id},
            data={"status": status},
        )
        return _risk_to_response(updated)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to update risk status: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update risk status: {exc}",
        )


@router.delete(
    "/{project_id}/gap-analysis",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def clear_gap_analysis(project_id: str) -> None:
    try:
        existing = await db.project.find_unique(where={"id": project_id})
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        await db.riskblockage.delete_many(where={"projectId": project_id})
        logger.info("Cleared gap analysis for project %s", project_id)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to clear gap analysis: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to clear gap analysis: {exc}",
        )
