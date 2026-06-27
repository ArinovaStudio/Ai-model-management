import json
import logging
from typing import Any, Optional

from app.db.prisma import db
from app.services.embedding import embedding_service
from app.services.extraction import (
    build_summary_text,
    extract_constraints,
    extract_scope,
    extract_timeline,
)

logger = logging.getLogger(__name__)


class ProjectNotFoundError(Exception):
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(f"Project {project_id} not found")


class ContextGenerationError(Exception):
    pass


async def _ensure_embeddings(
    records: list[Any],
    record_type: str,
) -> None:
    texts_to_embed: list[str] = []
    records_to_update: list[Any] = []

    for record in records:
        if not record.embedding:
            texts_to_embed.append(record.content)
            records_to_update.append(record)

    if not texts_to_embed:
        return

    embeddings = await embedding_service.generate_embeddings_batch(texts_to_embed)

    update_table = db.plandoc if record_type == "plandoc" else db.doc

    for record, emb in zip(records_to_update, embeddings):
        if emb is not None:
            serialized = embedding_service.serialize_embedding(emb)
            await update_table.update(
                where={"id": record.id},
                data={"embedding": serialized},
            )


async def generate_context(project_id: str) -> dict[str, Any]:
    project = await db.project.find_unique(
        where={"id": project_id},
        include={
            "projectInfo": True,
            "planDocs": True,
            "docs": True,
        },
    )

    if not project:
        raise ProjectNotFoundError(project_id)

    project_info = project.projectInfo
    plan_docs = project.planDocs or []
    docs = project.docs or []

    all_texts: list[str] = []
    if project_info:
        if project_info.description:
            all_texts.append(project_info.description)
        if project_info.scope:
            all_texts.append(project_info.scope)
        if project_info.constraints:
            all_texts.append(project_info.constraints)

    for doc in plan_docs:
        all_texts.append(doc.content)
    for doc in docs:
        all_texts.append(doc.content)

    try:
        await _ensure_embeddings(plan_docs, "plandoc")
        await _ensure_embeddings(docs, "doc")
    except Exception as exc:
        logger.warning("Embedding generation failed (non-fatal): %s", exc)

    timeline: dict[str, Any] = {}
    scope_items: list[str] = []
    constraints: list[str] = []

    if project_info:
        timeline = extract_timeline(
            start_date=project_info.startDate,
            deadline=project_info.deadline,
            texts=all_texts,
        )

    scope_items = extract_scope(all_texts)
    constraints = extract_constraints(all_texts)

    summary_text = build_summary_text(
        project_name=project.name,
        description=project_info.description if project_info else None,
        timeline=timeline,
        scope_items=scope_items,
        constraints=constraints,
        plan_docs_count=len(plan_docs),
        docs_count=len(docs),
    )

    summary: dict[str, Any] = {
        "project_summary": summary_text,
        "timeline": timeline,
        "scope": {
            "summary": scope_items[0] if scope_items else "No scope information extracted from available documents.",
            "items": scope_items,
        },
        "constraints": {
            "summary": constraints[0] if constraints else "No constraints extracted from available documents.",
            "items": constraints,
        },
        "documentation": {
            "plan_docs_count": len(plan_docs),
            "docs_count": len(docs),
            "total_docs": len(plan_docs) + len(docs),
            "has_embeddings": any(
                doc.embedding is not None
                for doc in list(plan_docs) + list(docs)
            ),
        },
    }

    await db.project.update(
        where={"id": project_id},
        data={"summary": json.dumps(summary)},
    )

    logger.info("Context generated successfully for project %s", project_id)
    return summary
