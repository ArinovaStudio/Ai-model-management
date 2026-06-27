import json
import logging
import math
from typing import Any, Optional

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self.client: Optional[AsyncOpenAI] = None
        if settings.OPENAI_API_KEY:
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate_embedding(self, text: str) -> Optional[list[float]]:
        if not self.client:
            logger.warning("OpenAI client not configured; skipping embedding generation")
            return None
        try:
            response = await self.client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=text,
            )
            return response.data[0].embedding
        except Exception as exc:
            logger.error("Failed to generate embedding: %s", exc)
            return None

    async def generate_embeddings_batch(
        self, texts: list[str]
    ) -> list[Optional[list[float]]]:
        if not self.client:
            logger.warning("OpenAI client not configured; skipping batch embedding generation")
            return [None] * len(texts)
        try:
            response = await self.client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=texts,
            )
            results: list[Optional[list[float]]] = [None] * len(texts)
            for data in response.data:
                results[data.index] = data.embedding
            return results
        except Exception as exc:
            logger.error("Failed to generate batch embeddings: %s", exc)
            return [None] * len(texts)

    def serialize_embedding(self, embedding: list[float]) -> str:
        return json.dumps(embedding)

    def deserialize_embedding(self, data: str) -> list[float]:
        return json.loads(data)

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def semantic_search(
        self,
        project_id: str,
        query: str,
        k: int = 5,
        doc_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        query_emb = await self.generate_embedding(query)
        if not query_emb:
            logger.warning("Cannot perform semantic search: no query embedding")
            return []

        from app.db.prisma import db

        docs: list[Any] = []
        if doc_type in (None, "plandoc"):
            docs.extend(
                await db.plandoc.find_many(
                    where={"project_id": project_id, "embedding": {"not": None}},
                )
            )
        if doc_type in (None, "doc"):
            docs.extend(
                await db.doc.find_many(
                    where={"project_id": project_id, "embedding": {"not": None}},
                )
            )

        scored: list[tuple[float, Any, str]] = []
        for doc in docs:
            try:
                doc_emb = self.deserialize_embedding(doc.embedding)
                score = self.cosine_similarity(query_emb, doc_emb)
                doc_type_label = "plandoc" if hasattr(doc, "project_id") and not hasattr(doc, "doc_type") else "doc"
                if hasattr(doc, "doc_type") and doc.doc_type:
                    doc_type_label = doc.doc_type
                scored.append((score, doc, doc_type_label))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning("Failed to parse embedding for doc %s: %s", doc.id, exc)

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, doc, dtype in scored[:k]:
            content = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
            results.append({
                "doc_id": doc.id,
                "doc_type": dtype,
                "title": doc.title,
                "content_snippet": content,
                "similarity_score": round(score, 4),
            })

        return results


embedding_service = EmbeddingService()
