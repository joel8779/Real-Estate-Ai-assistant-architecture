"""Indexing Pipeline for DarGlobal & Wasalt Real Estate Knowledge Base.

Loads normalized entities from persistent storage, chunks them preserving context,
generates normalized vector embeddings via all-MiniLM-L6-v2, and populates
ChromaDB and BM25 index structures.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from backend.app.config import get_settings
from backend.app.core.logging import logger
from backend.app.ingestion.storage import StorageManager
from backend.app.rag.chunker import DocumentChunker
from backend.app.rag.embedder import Embedder
from backend.app.rag.keyword_search import BM25SearchEngine
from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.schemas.document import NormalizedEntity, VectorChunk


class IndexingPipeline:
    """End-to-end indexing pipeline from processed documents to ChromaDB and BM25."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        embedder: Optional[Embedder] = None,
        bm25_engine: Optional[BM25SearchEngine] = None,
        target_chunk_size: int = 400,
        chunk_overlap: int = 50,
    ):
        self.chunker = DocumentChunker(target_chunk_size=target_chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or Embedder.get_instance()
        self.bm25_engine = bm25_engine or BM25SearchEngine()

    def run(
        self,
        entities: Optional[List[NormalizedEntity]] = None,
        processed_file: Optional[Path] = None,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Execute the indexing workflow.

        Args:
            entities: Optional pre-loaded list of NormalizedEntity.
            processed_file: Path to records.json if entities not provided.
            batch_size: Batch size for generating embeddings.

        Returns:
            Dict containing indexing statistics and summary report.
        """
        start_time = time.time()
        settings = get_settings()

        # 1. Load entities
        if entities is None:
            storage = StorageManager(processed_dir=processed_file.parent if processed_file else Path("data/processed"))
            entities = storage.load_all_normalized()

        logger.info(f"Loaded {len(entities)} normalized entities for indexing.")
        if not entities:
            logger.warning("No normalized records found to index.")
            return {
                "status": "empty",
                "indexed_entities": 0,
                "total_chunks": 0,
                "elapsed_seconds": 0.0,
            }

        # 2. Chunk documents
        logger.info("Chunking entities...")
        chunks: List[VectorChunk] = self.chunker.chunk_documents(entities)
        logger.info(f"Generated {len(chunks)} chunks from {len(entities)} entities.")

        # 3. Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(chunks)} chunks using all-MiniLM-L6-v2...")
        texts = [chunk.text_content for chunk in chunks]
        embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb_batch = self.embedder.embed_documents(batch)
            embeddings.extend(emb_batch)
            logger.debug(f"Embedded batch {i // batch_size + 1} ({len(embeddings)}/{len(texts)})")

        # 4. Populate ChromaDB
        logger.info(f"Writing {len(chunks)} chunks to ChromaDB vector store...")
        indexed_count = self.vector_store.index_chunks(chunks=chunks, embeddings=embeddings)

        # 5. Build BM25 index
        logger.info("Building BM25 keyword index...")
        self.bm25_engine.build_index(chunks)

        elapsed = time.time() - start_time

        # 6. Gather statistics
        source_breakdown: Dict[str, int] = {}
        category_breakdown: Dict[str, int] = {}
        provenance_breakdown: Dict[str, int] = {}

        for chunk in chunks:
            source_breakdown[chunk.source] = source_breakdown.get(chunk.source, 0) + 1
            cat_name = chunk.category.value
            category_breakdown[cat_name] = category_breakdown.get(cat_name, 0) + 1
            prov = str(chunk.provenance)
            provenance_breakdown[prov] = provenance_breakdown.get(prov, 0) + 1

        report = {
            "status": "success",
            "indexed_entities": len(entities),
            "total_chunks": len(chunks),
            "chroma_total_count": self.vector_store.count(),
            "source_breakdown": source_breakdown,
            "category_breakdown": category_breakdown,
            "provenance_breakdown": provenance_breakdown,
            "elapsed_seconds": round(elapsed, 2),
            "embedding_model": self.embedder.model_name,
            "chroma_dir": self.vector_store.persist_dir,
        }

        # Save indexing report to data/processed/indexing_report.json
        report_path = Path("data/processed/indexing_report.json")
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Indexing report saved to '{report_path}'.")
        except Exception as e:
            logger.warning(f"Could not write indexing report: {e}")

        logger.info(
            f"Indexing completed successfully: {len(chunks)} chunks indexed in {elapsed:.2f}s."
        )
        return report


def main():
    """CLI Entrypoint for running the indexing pipeline."""
    parser = argparse.ArgumentParser(
        description="Index normalized DarGlobal and Wasalt records into ChromaDB & BM25."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/processed/records.json"),
        help="Path to normalized records.json file (default: data/processed/records.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for generating embeddings (default: 32)",
    )
    args = parser.parse_args()

    pipeline = IndexingPipeline()
    report = pipeline.run(processed_file=args.file, batch_size=args.batch_size)
    print("\n--- Indexing Summary Report ---")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
