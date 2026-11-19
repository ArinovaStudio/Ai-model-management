"""add hnsw vector index

Revision ID: c3a4f891b72e
Revises: 91a0f86261ed
Create Date: 2026-10-26 13:00:00.000000

"""
from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c3a4f891b72e'
down_revision: Union[str, Sequence[str], None] = '91a0f86261ed'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema by adding HNSW graph index on vector embeddings."""
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_embedding_hnsw "
        "ON knowledge_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64);"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_knowledge_chunks_embedding_hnsw;")
