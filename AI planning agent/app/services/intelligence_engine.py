import json
import logging

import asyncio
from app.db.prisma import db, get_cached_project, set_cached_project
from prisma import Json
from app.services.ollama_service import (
    API_AND_SECURITY_PROMPT,
    DATABASE_DESIGN_PROMPT,
    FEATURE_SPEC_PROMPT,
    PRODUCT_STRATEGY_PROMPT,
    SYSTEM_ARCHITECTURE_PROMPT,
    ollama_service,
)

logger = logging.getLogger(__name__)


async def generate_project_intelligence(project_id: str) -> dict | None:
    project = get_cached_project(project_id)
    if not project or not getattr(project, "planDocs", None):
        project = await db.project.find_unique(
            where={"id": project_id},
            include={
                "featureRequests": True,
                "projectInfo": True,
                "technologies": True,
                "projectIntelligence": True,
                "groupingTasks": {"include": {"tasks": True}},
                "resourceAllocations": True,
                "planDocs": True,
                "docs": True,
            },
        )
        if project:
            set_cached_project(project_id, project)

    if not project:
        logger.warning("Project %s not found", project_id)
        return None

    project_info = project.projectInfo
    plan_docs = project.planDocs or []
    docs = project.docs or []

    parts = [f"Project: {project.name}"]
    if project_info:
        if project_info.description:
            parts.append(f"Description: {project_info.description}")
        if project_info.scope:
            parts.append(f"Scope: {project_info.scope}")
        if project_info.constraints:
            parts.append(f"Constraints: {project_info.constraints}")
        if project_info.projectType:
            parts.append(f"Preferred Architecture Pattern: {project_info.projectType}")
        if project_info.startDate and project_info.deadline:
            duration = (project_info.deadline - project_info.startDate).days
            parts.append(
                f"Timeline: {project_info.startDate.isoformat()} to "
                f"{project_info.deadline.isoformat()} ({duration} days)"
            )

    if plan_docs:
        parts.append(f"\nPlan Documents ({len(plan_docs)}):")
        for d in plan_docs[:5]:
            parts.append(f"  - {d.title}: {d.content[:500]}")
    if docs:
        parts.append(f"\nSupporting Documents ({len(docs)}):")
        for d in docs[:5]:
            parts.append(f"  - {d.title}: {d.content[:500]}")

    base_context = "\n".join(parts)

    strategy = await ollama_service.generate_intelligence_layer(PRODUCT_STRATEGY_PROMPT, base_context, temperature=0.4)

    context_2 = base_context + "\n" + json.dumps(strategy)
    features = await ollama_service.generate_intelligence_layer(FEATURE_SPEC_PROMPT, context_2, temperature=0.3)

    context_3 = context_2 + "\n" + json.dumps(features)
    
    # Run the three independent architecture/security layers sequentially to prevent Ollama queue timeout overload
    architecture = await ollama_service.generate_intelligence_layer(SYSTEM_ARCHITECTURE_PROMPT, context_3, temperature=0.4)
    database = await ollama_service.generate_intelligence_layer(DATABASE_DESIGN_PROMPT, context_3)
    api_security = await ollama_service.generate_intelligence_layer(API_AND_SECURITY_PROMPT, context_3)

    record = await db.projectintelligence.upsert(
        where={"projectId": project_id},
        data={
            "create": {
                "project": {
                    "connect": {"id": project_id}
                },
                "productStrategy": Json(strategy),
                "featureSpecs": Json(features),
                "systemArchitecture": Json(architecture),
                "databaseDesign": Json(database),
                "apiDesign": Json(api_security),
                "security": Json(api_security),
            },
            "update": {
                "productStrategy": Json(strategy),
                "featureSpecs": Json(features),
                "systemArchitecture": Json(architecture),
                "databaseDesign": Json(database),
                "apiDesign": Json(api_security),
                "security": Json(api_security),
            },
        },
    )

    context_docs = context_3 + "\n" + json.dumps(architecture) + "\n" + json.dumps(database)

    # Immediately upsert placeholder/generating state in database so UI knows they are compiling
    doc_titles = [
        "Core Technical Blueprint",
        "Project Constraints",
        "Success Metrics Framework",
        "Assumptions Register",
        "Decision Brief"
    ]
    for title in doc_titles:
        existing_doc = await db.plandoc.find_first(
            where={"projectId": project_id, "title": title}
        )
        if not existing_doc:
            await db.plandoc.create(
                data={
                    "projectId": project_id,
                    "title": title,
                    "content": "Generating...",
                }
            )
        else:
            await db.plandoc.update(
                where={"id": existing_doc.id},
                data={"content": "Generating..."}
            )

    # Trigger document generation sequentially in a background task to prevent HTTP keep-alive timeouts
    def run_bg_generation_and_indexing(proj_id: str, ctx_docs: str):
        async def _bg_task():
            try:
                from app.services.ollama_service import (
                    CORE_TECHNICAL_BLUEPRINT_PROMPT,
                    PROJECT_CONSTRAINTS_PROMPT,
                    SUCCESS_METRICS_PROMPT,
                    ASSUMPTIONS_REGISTER_PROMPT,
                    DECISION_BRIEF_PROMPT
                )
                # Generate the 5 strategic documents sequentially to prevent Ollama overload
                blueprint = await ollama_service.generate_document_style(CORE_TECHNICAL_BLUEPRINT_PROMPT, ctx_docs)
                constraints = await ollama_service.generate_document_style(PROJECT_CONSTRAINTS_PROMPT, ctx_docs)
                metrics = await ollama_service.generate_document_style(SUCCESS_METRICS_PROMPT, ctx_docs)
                assumptions = await ollama_service.generate_document_style(ASSUMPTIONS_REGISTER_PROMPT, ctx_docs)
                decision = await ollama_service.generate_document_style(DECISION_BRIEF_PROMPT, ctx_docs)
                
                doc_contents = [blueprint, constraints, metrics, assumptions, decision]
                for title, content in zip(doc_titles, doc_contents):
                    existing = await db.plandoc.find_first(
                        where={"projectId": proj_id, "title": title}
                    )
                    if existing:
                        await db.plandoc.update(
                            where={"id": existing.id},
                            data={"content": content}
                        )
                
                from app.services.analysis_engine import _ensure_doc_embeddings
                await _ensure_doc_embeddings(proj_id)
            except Exception as e:
                logger.error("Background sequential doc generation/indexing failed: %s", e)
        
        import asyncio
        asyncio.create_task(_bg_task())

    run_bg_generation_and_indexing(project_id, context_docs)

    return record
