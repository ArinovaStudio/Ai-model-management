import json
import logging
import math
import asyncio
from typing import Any, Optional

from ollama import AsyncClient

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self.client = AsyncClient(host=settings.OLLAMA_BASE_URL)
        self.model = settings.EMBEDDING_MODEL

    async def generate_embedding(self, text: str) -> Optional[list[float]]:
        try:
            response = await self.client.embeddings(
                model=self.model,
                prompt=text,
            )
            return response.get("embedding")
        except Exception:
            if not self.model.endswith(":latest"):
                try:
                    response = await self.client.embeddings(
                        model=f"{self.model}:latest",
                        prompt=text,
                    )
                    return response.get("embedding")
                except Exception:
                    pass
            try:
                remote_client = AsyncClient(host=settings.OLLAMA_BASE_URL)
                response = await remote_client.embeddings(
                    model=self.model,
                    prompt=text,
                )
                return response.get("embedding")
            except Exception as remote_exc:
                logger.error("Failed to generate embedding: %s", remote_exc)
                return None

    async def generate_embeddings_batch(
        self, texts: list[str]
    ) -> list[Optional[list[float]]]:
        # Generate all embeddings concurrently
        return list(await asyncio.gather(*(self.generate_embedding(text) for text in texts)))

    def chunk_text(self, text: str, chunk_size: int = 700, overlap: int = 150) -> list[str]:
        if not text:
            return []
        
        # Split text into paragraphs/blocks first
        blocks = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0
        
        for block in blocks:
            # Check if block is a markdown table
            lines = block.strip().split("\n")
            is_table = len(lines) >= 2 and any("|" in l for l in lines) and any("---" in l for l in lines)
            
            if is_table:
                # If we have accumulated text, save it first
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                # Tables are preserved intact as atomic chunks
                chunks.append(block)
            else:
                block_len = len(block)
                if current_len + block_len > chunk_size:
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                    current_chunk = [block]
                    current_len = block_len
                else:
                    current_chunk.append(block)
                    current_len += block_len + 2 # account for separator
                    
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        # Fallback if no chunks were created (e.g. text didn't have double newlines)
        if not chunks:
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start += chunk_size - overlap
                
        return chunks

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

        # Safely check doc_type. When endpoints are called directly in testing, doc_type can default to Query() object
        doc_type_str = None
        if isinstance(doc_type, str):
            doc_type_str = doc_type.lower()
        elif doc_type is not None and not hasattr(doc_type, "default"):
            doc_type_str = str(doc_type).lower()

        docs: list[Any] = []
        if doc_type_str in (None, "plandoc"):
            docs.extend(
                await db.plandoc.find_many(
                    where={"projectId": project_id, "embedding": {"not": None}},
                )
            )
        if doc_type_str in (None, "doc"):
            docs.extend(
                await db.doc.find_many(
                    where={"projectId": project_id, "embedding": {"not": None}},
                )
            )

        scored: list[tuple[float, Any, str, str]] = []
        for doc in docs:
            # Check if document has chunks stored inside metadata JSON
            chunks = []
            if doc.metadata:
                try:
                    meta_dict = json.loads(doc.metadata)
                    if isinstance(meta_dict, dict) and "chunks" in meta_dict:
                        chunks = meta_dict["chunks"]
                except Exception:
                    pass

            doc_type_label = "plandoc" if hasattr(doc, "projectId") and not hasattr(doc, "docType") else "doc"
            if hasattr(doc, "docType") and doc.docType:
                doc_type_label = doc.docType

            if chunks:
                for chunk in chunks:
                    try:
                        chunk_emb = chunk.get("embedding")
                        if chunk_emb:
                            score = self.cosine_similarity(query_emb, chunk_emb)
                            scored.append((score, doc, doc_type_label, chunk.get("text", "")))
                    except Exception:
                        pass
            else:
                # Fallback to document-level embedding
                try:
                    doc_emb = self.deserialize_embedding(doc.embedding)
                    score = self.cosine_similarity(query_emb, doc_emb)
                    scored.append((score, doc, doc_type_label, doc.content))
                except Exception as exc:
                    logger.warning("Failed to parse embedding for doc %s: %s", doc.id, exc)

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, doc, dtype, text in scored[:k]:
            snippet = text[:300] + "..." if len(text) > 300 else text
            results.append({
                "doc_id": doc.id,
                "doc_type": dtype,
                "title": doc.title,
                "content_snippet": snippet,
                "similarity_score": round(score, 4),
            })

        return results


embedding_service = EmbeddingService()
