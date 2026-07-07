import json
import logging

from app.db.prisma import db
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
    project = await db.project.find_unique(
        where={"id": project_id},
        include={
            "projectInfo": True,
            "planDocs": True,
            "docs": True,
        },
    )

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

    return record
