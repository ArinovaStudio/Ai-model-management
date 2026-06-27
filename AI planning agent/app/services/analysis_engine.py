import json
import logging
from typing import Any, Optional

from app.db.prisma import db
from app.services.embedding import embedding_service
from app.services.ollama_service import ollama_service
from app.services.tech_stack import detect_technologies_from_all

logger = logging.getLogger(__name__)


class ProjectNotFoundError(Exception):
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(f"Project {project_id} not found")


def _extract_keyword_requirements(texts: list[str]) -> list[dict[str, Any]]:
    import re
    from app.services.extraction import _split_sentences

    patterns = [
        r"(?:the\s+(?:system|application|user|platform))\s+(?:shall|must|should|will|can|needs?\s+to)\s+(.+)",
        r"(?:requirement|feature|functionality)\s*:?\s*(.+)",
        r"(?:user\s+(?:shall|must|should|will|can|needs?\s+to))\s+(.+)",
        r"(?:ability\s+to)\s+(.+)",
        r"(?:support\s+for)\s+(.+)",
    ]

    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for text in texts:
        for sentence in _split_sentences(text):
            for pattern in patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    key = sentence.lower().strip()
                    if key not in seen:
                        seen.add(key)
                        desc = match.group(0).strip()
                        title = (
                            match.group(1).strip().split(".")[0][:80]
                            if match.lastindex and match.group(1)
                            else desc[:80]
                        )
                        results.append({
                            "title": title,
                            "description": desc,
                            "category": "functional",
                            "priority": "medium",
                            "related_technologies": [],
                            "ambiguous": False,
                            "clarificationPrompt": None,
                        })
                    break

    return results


async def _ensure_doc_embeddings(project_id: str) -> None:
    plan_docs = await db.plandoc.find_many(where={"projectId": project_id})
    docs = await db.doc.find_many(where={"projectId": project_id})

    for record_list, table in [(plan_docs, db.plandoc), (docs, db.doc)]:
        texts: list[str] = []
        records: list[Any] = []
        for r in record_list:
            if not r.embedding:
                texts.append(r.content)
                records.append(r)

        if not texts:
            continue

        embeddings = await embedding_service.generate_embeddings_batch(texts)
        for record, emb in zip(records, embeddings):
            if emb is not None:
                await table.update(
                    where={"id": record.id},
                    data={"embedding": embedding_service.serialize_embedding(emb)},
                )


async def analyze_project(
    project_id: str,
    reindex: bool = False,
) -> dict[str, Any]:
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

    if reindex:
        await db.featurerequest.delete_many(where={"projectId": project_id})
        await db.projecttechnology.delete_many(where={"projectId": project_id})

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
        await _ensure_doc_embeddings(project_id)
    except Exception as exc:
        logger.warning("Embedding refresh failed (non-fatal): %s", exc)

    requirements: list[dict[str, Any]] = []
    for text in all_texts:
        extracted = await ollama_service.extract_requirements(text)
        if extracted:
            requirements.extend(extracted)

    if not requirements:
        logger.info("Ollama returned no results; falling back to keyword extraction")
        requirements = _extract_keyword_requirements(all_texts)

    feature_records: list[Any] = []
    ambiguous_records: list[Any] = []
    clarification_prompts: list[str] = []

    for req in requirements:
        title = (req.get("title") or "").strip()
        description = (req.get("description") or "").strip()
        if not title or not description:
            continue

        is_ambiguous = bool(req.get("ambiguous", False))
        clarification_prompt = req.get("clarificationPrompt")

        req_embedding = None
        try:
            emb = await embedding_service.generate_embedding(f"{title}: {description}")
            if emb:
                req_embedding = embedding_service.serialize_embedding(emb)
        except Exception as exc:
            logger.warning("Failed to generate embedding for requirement: %s", exc)

        record = await db.featurerequest.create(
            data={
                "clientId": "system",
                "projectId": project_id,
                "title": title,
                "description": description,
                "priority": (req.get("priority") or "medium"),
                "status": "needs_clarification" if is_ambiguous else "proposed",
                "category": req.get("category", "functional"),
                "embedding": req_embedding,
                "isAmbiguous": is_ambiguous,
                "clarificationPrompt": clarification_prompt,
                "metadata": json.dumps({
                    "related_technologies": req.get("related_technologies", []),
                    "extraction_method": "ollama" if requirements else "keyword",
                }),
            }
        )

        feature_records.append(record)
        if is_ambiguous:
            ambiguous_records.append(record)
            if clarification_prompt:
                clarification_prompts.append(clarification_prompt)

    technologies = detect_technologies_from_all(
        project_description=project_info.description if project_info else None,
        plan_doc_texts=[d.content for d in plan_docs],
        doc_texts=[d.content for d in docs],
    )

    existing_techs = await db.projecttechnology.find_many(
        where={"projectId": project_id},
    )
    existing_names = {t.name.lower() for t in existing_techs}

    tech_records: list[Any] = []
    for tech in technologies:
        if tech["name"].lower() in existing_names:
            continue
        record = await db.projecttechnology.create(
            data={
                "projectId": project_id,
                "name": tech["name"],
                "category": tech["category"],
                "version": tech.get("version"),
                "isDetected": tech.get("is_detected", True),
                "dependencies": json.dumps(tech.get("dependencies", [])),
            }
        )
        tech_records.append(record)
        existing_names.add(tech["name"].lower())

    return {
        "feature_requests": feature_records,
        "technologies": tech_records + existing_techs,
        "ambiguous_requirements": ambiguous_records,
        "clarification_prompts": clarification_prompts,
    }


async def clarify_requirement(
    feature_id: str,
    clarification: str,
) -> dict[str, Any]:
    existing = await db.featurerequest.find_unique(where={"id": feature_id})
    if not existing:
        raise ValueError(f"FeatureRequest {feature_id} not found")
    if not existing.isAmbiguous:
        raise ValueError(f"FeatureRequest {feature_id} is not marked as ambiguous")

    original = {
        "title": existing.title,
        "description": existing.description,
        "category": existing.category,
        "priority": existing.priority,
    }

    clarified = await ollama_service.clarify_requirement(original, clarification)

    title = (clarified.get("title") or "").strip() or existing.title
    description = (clarified.get("description") or "").strip() or clarification

    emb = await embedding_service.generate_embedding(f"{title}: {description}")
    emb_serialized = embedding_service.serialize_embedding(emb) if emb else None

    updated = await db.featurerequest.update(
        where={"id": feature_id},
        data={
            "title": title,
            "description": description,
            "category": clarified.get("category", existing.category),
            "priority": clarified.get("priority", existing.priority),
            "isAmbiguous": False,
            "clarificationPrompt": None,
            "status": "clarified",
            "embedding": emb_serialized,
            "metadata": json.dumps({
                "original_title": existing.title,
                "original_description": existing.description,
                "clarification": clarification,
            }),
        },
    )
    return updated
