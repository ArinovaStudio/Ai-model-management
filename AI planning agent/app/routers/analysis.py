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
    ChatRequest,
    ChatResponse,
    ChatMessageResponse,
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


def _clean_user_message(content: str) -> str:
    if "User Question:" in content:
        return content.split("User Question:")[-1].strip()
    elif "Question:" in content:
        return content.split("Question:")[-1].strip()
    return content


@router.get(
    "/{project_id}/chat",
    response_model=list[ChatMessageResponse],
)
async def get_chat_history(project_id: str) -> list[ChatMessageResponse]:
    try:
        messages = await db.chatmessage.find_many(
            where={"projectId": project_id},
            order={"createdAt": "asc"},
        )
        return [
            ChatMessageResponse(
                id=m.id,
                project_id=m.projectId,
                role=m.role,
                content=_clean_user_message(m.content) if m.role == "user" else m.content,
                created_at=m.createdAt,
            )
            for m in messages
        ]
    except Exception as exc:
        logger.error("Failed to retrieve chat history: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to retrieve chat history: {exc}",
        )


@router.post(
    "/{project_id}/chat",
    response_model=ChatResponse,
)
async def chat_endpoint(project_id: str, body: ChatRequest) -> ChatResponse:
    try:
        # 1. Fetch project config and conversational history concurrently (before saving new message)
        project_task = db.project.find_unique(
            where={"id": project_id},
            include={
                "projectInfo": True,
                "technologies": True,
                "projectIntelligence": True,
            }
        )
        history_task = db.chatmessage.find_many(
            where={"projectId": project_id},
            order={"createdAt": "desc"},
            take=15,
        )

        project, past_messages = await asyncio.gather(
            project_task, history_task
        )

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found",
            )

        # 2. Sort past messages back to chronological order (since new message is not saved yet, this is our clean history)
        past_messages.reverse()
        history = [
            {"role": m.role, "content": m.content}
            for m in past_messages
        ]

        # 3. Resolve pronouns/references in the query based on recent chat history (Query Expansion)
        resolved_query = body.message
        if len(history) >= 2:
            last_exchange = history[-2:]
            expansion_prompt = (
                "Given the following brief chat history, rewrite the latest user question to resolve any pronouns or relative list references "
                "(like 'it', 'that phase', 'this database', 'the first item you mentioned', 'the 5th point') to their concrete names from the chat history "
                "so that it is a standalone search query. Do not answer the question, just output the rewritten question. If it is already a standalone query, return it unchanged.\n\n"
                f"Past User: {last_exchange[0]['content'] if isinstance(last_exchange[0], dict) and 'content' in last_exchange[0] else getattr(last_exchange[0], 'content', '')}\n"
                f"Past Assistant: {last_exchange[1]['content'] if isinstance(last_exchange[1], dict) and 'content' in last_exchange[1] else getattr(last_exchange[1], 'content', '')}\n"
                f"Latest User Question: {body.message}\n\n"
                "Rewritten Question:"
            )
            try:
                rewritten = await ollama_service.generate_document_style(
                    system_prompt="You are a query rewriting assistant. You only output the rewritten question.",
                    context=expansion_prompt
                )
                if rewritten and "Failed to generate" not in rewritten:
                    resolved_query = rewritten.strip().strip('"').strip("'")
                    logger.info("Resolved query: %s -> %s", body.message, resolved_query)
            except Exception as e:
                logger.warning("Failed to rewrite query: %s", e)

        # 4. Perform semantic search using the expanded/resolved query
        results = await embedding_service.semantic_search(
            project_id=project_id,
            query=resolved_query,
            k=3,
        )

        # 5. Long-term memory compilation (Summarization)
        summary_str = ""
        if len(past_messages) > 10:
            try:
                summary_prompt = (
                    "Summarize the key topics discussed, technical decisions made, and user constraints in the following chat conversation history. "
                    "Keep it concise (1-2 paragraphs):\n\n"
                    + "\n".join(f"{m['role']}: {m['content']}" for m in history[:-3])
                )
                summary_res = await ollama_service.generate_document_style(
                    system_prompt="You are a conversation summarizer assistant. Be concise.",
                    context=summary_prompt
                )
                if summary_res and "Failed to generate" not in summary_res:
                    summary_str = summary_res.strip()
                    history = history[-3:]  # Retain only the last 3 messages to avoid context bloat
            except Exception as e:
                logger.warning("Failed to compile conversation summary: %s", e)

        # 6. Build tech stack details
        tech_list = []
        if project.technologies:
            for t in project.technologies:
                tech_list.append(f"- {t.name} ({t.category or 'general'})")
        tech_stack_str = "\n".join(tech_list) if tech_list else "No explicit tech stack detected."

        # 7. Build architecture & DB design details
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

        # 8. Build project description/scope
        proj_info_str = "No description provided."
        if project.projectInfo:
            proj_info_str = f"Description: {project.projectInfo.description or ''}\nScope: {project.projectInfo.scope or ''}"

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

        # Keep the static project blueprint inside the system prompt to maximize Ollama prompt caching hits
        rag_system_prompt = (
            "You are an Expert Project Architect and Brainstorming Assistant.\n"
            "Your goal is to answer the user's question with high technical accuracy. You are provided with two layers of context:\n"
            "1) Document Context: Excerpts from project documents (if available).\n"
            "2) Project Blueprint: The generated tech stack, system architecture, database design, and project scope.\n\n"
            "CRITICAL DIRECTIVES:\n"
            "- Answer the user's latest question directly. If they ask follow-up questions, reply to statements, or ask for comparisons (like comparing to other technologies like Azure, AWS, etc.), use the chat history and your general engineering expertise to explain why the blueprint chose its design over alternatives.\n"
            "- Conversational Context & Reference Resolution: If the user asks you to explain or expand on your own previous recommendations, strategies, or lists from the chat history (e.g. 'the first item you mentioned', 'the 5th point', etc.), prioritize the conversational context, locate that specific item in the Assistant's previous message, and expand on it logically. Do not confuse it with lists in the Project Blueprint.\n"
            "- Primary Grounding: Use the Project Blueprint and Technology Stack as your primary reference design.\n"
            "- Ensure your recommendations are compatible with the Active Technology Stack and System Architecture design.\n"
            "- Keep the response technical, direct, and conversational. Do not reference prompt structures or repeat the prompt back to the user.\n\n"
            f"=== Project Blueprint ===\n{structural_context}"
        )

        if summary_str:
            rag_system_prompt += f"\n\n=== CONVERSATION SUMMARY SO FAR ===\n{summary_str}"

        # 9. Defensive Guessing Threshold Check
        max_similarity = max(r["similarity_score"] for r in results) if results else 0
        if max_similarity < 0.35:
            rag_system_prompt += (
                "\n\nWARNING: The retrieved document context does not have a high similarity score to the user's question, meaning the answer is likely not in the project documents. "
                "If you cannot find the answer in the Project Blueprint either, explicitly state that you couldn't find this information in the project documents, "
                "provide a general best-practice engineering suggestion, and politely ask the user to clarify or upload additional specs."
            )

        # Refine the prompt template to guide smaller local models to separate RAG context from the user query
        prompt_with_rag = (
            f"=== RETRIEVED DOCUMENT CONTEXT ===\n{context_str}\n\n"
            f"=== INSTRUCTIONS ===\n"
            f"- Answer the User Question below directly.\n"
            f"- If the retrieved document context above does not contain the answer, ignore it and answer using the Project Blueprint and your general engineering expertise.\n"
            f"- Do not hallucinate or make up recommendations that contradict the Technology Stack.\n\n"
            f"User Question: {body.message}"
        )

        # 10. Save clean user message in DB
        await db.chatmessage.create(
            data={
                "projectId": project_id,
                "role": "user",
                "content": body.message,
            }
        )
        
        # Format the final user message inside history
        formatted_history = list(history)
        formatted_history.append({"role": "user", "content": prompt_with_rag})

        # 11. Query conversational LLM
        answer = await ollama_service.chat_conversation(
            system_prompt=rag_system_prompt,
            history=formatted_history,
        )

        # 12. Save assistant response in DB
        await db.chatmessage.create(
            data={
                "projectId": project_id,
                "role": "assistant",
                "content": answer,
            }
        )

        return ChatResponse(
            query=body.message,
            answer=answer,
            results=[SemanticSearchResult(**r) for r in results],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chat conversation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chat conversation failed: {exc}",
        )

