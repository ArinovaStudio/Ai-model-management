"""CLI tool for ingesting real estate knowledge documents.

Usage:
    python -m scripts.ingest_documents --dir ./data/docs --access-level customer_shareable
    python -m scripts.ingest_documents --dir ./data/docs/internal --access-level internal_restricted
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.models.enums import AccessLabel
from src.rag.indexer import indexer
from src.core.logging import logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest real estate documents into pgvector knowledge base.")
    parser.add_argument(
        "--dir",
        type=str,
        default="./data/docs",
        help="Directory containing .pdf, .md, or .json documents to ingest.",
    )
    parser.add_argument(
        "--access-level",
        type=str,
        default="customer_shareable",
        choices=["customer_shareable", "internal_restricted"],
        help="Access classification for the ingested documents.",
    )

    args = parser.parse_args()
    target_dir = Path(args.dir)

    if not target_dir.exists():
        logger.error(f"Target directory '{target_dir}' does not exist.")
        return 1

    access_label = (
        AccessLabel.CUSTOMER_SHAREABLE
        if args.access_level == "customer_shareable"
        else AccessLabel.INTERNAL_RESTRICTED
    )

    logger.info(f"Starting ingestion from: {target_dir.resolve()}")
    logger.info(f"Assigned Access Label : {access_label.value}")

    try:
        count = indexer.ingest_directory(target_dir, access_label=access_label)
        logger.info(f"Ingestion finished successfully. Total indexed chunks: {count}")
        return 0
    except Exception as e:
        logger.error(f"Ingestion failed with exception: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
