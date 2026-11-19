"""Document Ingestion & Vector Indexer.

Parses .pdf, .md, and .json documents, chunks content, generates local
384-dimensional vector embeddings, and stores them in PostgreSQL with strict
access labels ('customer_shareable' vs 'internal_restricted').
"""

import json
from pathlib import Path
from typing import Any
from pypdf import PdfReader
from sqlalchemy.orm import Session
from src.db.session import SyncSessionLocal
from src.db.models import KnowledgeChunk, AccessLabel
from src.rag.embeddings import embed_texts
from src.core.logging import logger


class DocumentIndexer:
    """Ingests real estate documents and generates vector indexes."""

    def __init__(
        self,
        chunk_size: int = 1500,  # ~350-400 words/tokens
        chunk_overlap: int = 200,
        session_factory=SyncSessionLocal,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.session_factory = session_factory

    def parse_file(self, file_path: Path) -> tuple[str, str, dict[str, Any]]:
        """Extracts text, title, and metadata from supported file formats."""
        suffix = file_path.suffix.lower()
        title = file_path.stem.replace("_", " ").title()
        metadata: dict[str, Any] = {"file_name": file_path.name, "file_path": str(file_path)}

        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(pages)
            metadata["page_count"] = len(reader.pages)

        elif suffix == ".md":
            text = file_path.read_text(encoding="utf-8")

        elif suffix == ".json":
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                text = data.get("content") or json.dumps(data, indent=2)
                title = data.get("title", title)
                metadata.update(data.get("metadata", {}))
                if "property_id" in data:
                    metadata["property_id"] = data["property_id"]
            else:
                text = json.dumps(data, indent=2)

        else:
            raise ValueError(f"Unsupported file format '{suffix}'. Only .pdf, .md, and .json are supported.")

        return text.strip(), title, metadata

    def chunk_text(self, text: str) -> list[str]:
        """Splits text recursively using paragraphs, sentences, and character limits."""
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # If we're not at the very end, try breaking cleanly on a paragraph or sentence
            if end < text_len:
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break != -1 and paragraph_break > start + (self.chunk_size // 2):
                    end = paragraph_break + 2
                else:
                    sentence_break = text.rfind(". ", start, end)
                    if sentence_break != -1 and sentence_break > start + (self.chunk_size // 2):
                        end = sentence_break + 2

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Advance by step taking overlap into account
            step = end - start
            if step <= self.chunk_overlap:
                start += step
            else:
                start += step - self.chunk_overlap

        return chunks

    def ingest_file(
        self,
        file_path: Path,
        access_label: AccessLabel = AccessLabel.CUSTOMER_SHAREABLE,
        property_id: int | None = None,
        session: Session | None = None,
    ) -> int:
        """Ingests a single file, chunks it, generates embeddings, and saves to database."""
        logger.info(f"Parsing document: {file_path.name} (Access: {access_label.value})...")
        text, title, metadata = self.parse_file(file_path)

        # Allow file metadata to override property_id if present
        if property_id is None and "property_id" in metadata:
            try:
                property_id = int(metadata["property_id"])
            except (ValueError, TypeError):
                pass

        if property_id is None:
            try:
                import re
                from src.db.models import PropertyListing

                def _match_prop(s: Session) -> int | None:
                    listings = s.query(PropertyListing).all()
                    doc_ident = f"{file_path.stem} {title}".lower()
                    for prop in listings:
                        p_title = prop.title.lower()
                        if p_title in doc_ident or (prop.slug and prop.slug.replace("-", " ") in doc_ident):
                            return prop.id
                        significant_words = [
                            w for w in re.split(r"\W+", p_title)
                            if len(w) > 3 and w not in {"plot", "plots", "flat", "flats", "heights", "villas", "villa", "suites", "park", "apartments"}
                        ]
                        if significant_words and all(w in doc_ident for w in significant_words):
                            return prop.id
                    return None

                if session:
                    property_id = _match_prop(session)
                else:
                    with self.session_factory() as s_ctx:
                        property_id = _match_prop(s_ctx)
            except Exception as e:
                logger.debug(f"Could not auto-link property_id for {file_path.name}: {e}")

        chunks = self.chunk_text(text)
        if not chunks:
            logger.warning(f"No text extracted from {file_path.name}.")
            return 0

        logger.info(f"Generated {len(chunks)} text chunks for '{title}'. Computing embeddings...")
        embeddings = embed_texts(chunks)

        knowledge_records = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            record = KnowledgeChunk(
                property_id=property_id,
                document_title=f"{title} (Part {i + 1})",
                chunk_content=chunk,
                embedding=emb,
                access_label=access_label,
                metadata_payload=chunk_metadata,
            )
            knowledge_records.append(record)

        def _persist(sess: Session) -> int:
            try:
                from sqlalchemy import or_
                sess.query(KnowledgeChunk).filter(
                    or_(
                        KnowledgeChunk.document_title.like(f"{title} (Part %)"),
                        KnowledgeChunk.metadata_payload["file_name"].astext == file_path.name,
                    )
                ).delete(synchronize_session=False)
            except Exception as e:
                logger.debug(f"Idempotent purge notice for '{title}': {e}")

            sess.add_all(knowledge_records)
            sess.commit()
            return len(knowledge_records)

        if session:
            count = _persist(session)
        else:
            with self.session_factory() as sess:
                count = _persist(sess)

        logger.info(f"Successfully ingested and indexed {count} chunks for '{title}'.")
        return count

    def ingest_directory(
        self,
        directory: Path,
        access_label: AccessLabel = AccessLabel.CUSTOMER_SHAREABLE,
        session: Session | None = None,
    ) -> int:
        """Ingests all supported files in a directory."""
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Directory '{directory}' does not exist.")

        total_chunks = 0
        extensions = [".pdf", ".md", ".json"]
        files = [p for p in directory.rglob("*") if p.suffix.lower() in extensions]

        logger.info(f"Found {len(files)} documents in '{directory}' for ingestion.")
        for file_path in files:
            # Automatic heuristic: if filename or any parent folder contains 'internal', 'restricted', or 'confidential', tag as INTERNAL_RESTRICTED
            file_access = access_label
            file_path_str = str(file_path).lower()
            if any(term in file_path_str for term in ("internal", "restricted", "confidential")):
                file_access = AccessLabel.INTERNAL_RESTRICTED

            try:
                chunks_added = self.ingest_file(file_path, access_label=file_access, session=session)
                total_chunks += chunks_added
            except Exception as e:
                logger.error(f"Failed to ingest file '{file_path.name}': {e}")

        logger.info(f"Directory ingestion complete: {total_chunks} total chunks indexed.")
        return total_chunks


indexer = DocumentIndexer()
