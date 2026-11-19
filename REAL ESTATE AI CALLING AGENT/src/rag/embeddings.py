"""Local embedding engine using fastembed ONNX runtime."""

from typing import List
from fastembed import TextEmbedding
from src.core.logging import logger

_EMBED_MODEL = None
_DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384 dimensions


def get_embedding_model() -> TextEmbedding:
    """Returns singleton TextEmbedding model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        logger.info(f"Loading local embedding model: {_DEFAULT_MODEL_NAME}...")
        _EMBED_MODEL = TextEmbedding(model_name=_DEFAULT_MODEL_NAME)
    return _EMBED_MODEL


def embed_text(text: str) -> List[float]:
    """Generates a 384-dimensional vector embedding for a single string."""
    model = get_embedding_model()
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch generates embeddings for multiple strings."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]
