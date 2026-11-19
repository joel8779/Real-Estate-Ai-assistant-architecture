"""Lexical Keyword Retrieval using BM25.

Provides fast, transparent keyword search across chunk titles, text content,
and property attributes to complement dense vector retrieval.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from rank_bm25 import BM25Okapi

from backend.app.schemas.document import VectorChunk
from backend.app.core.logging import logger


def tokenize_text(text: str) -> List[str]:
    """Tokenize and normalize text into lowercase alphanumeric terms."""
    if not text:
        return []
    # Tokenize words, numbers, and property identifiers
    tokens = re.findall(r"\b[a-zA-Z0-9_\u0600-\u06FF-]+\b", text.lower())
    return tokens


class BM25SearchEngine:
    """Lightweight in-memory BM25 index over VectorChunks."""

    def __init__(self):
        self.chunks: List[VectorChunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def build_index(self, chunks: List[VectorChunk]) -> None:
        """Construct BM25 index from chunks corpus."""
        self.chunks = list(chunks)
        self.corpus_tokens = []

        for chunk in self.chunks:
            # Combine title, text content, and key metadata for lexical matching
            combined = f"{chunk.title} {chunk.category.value} {chunk.country or ''} {chunk.city or ''} {chunk.property_type or ''} {chunk.text_content}"
            tokens = tokenize_text(combined)
            self.corpus_tokens.append(tokens)

        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
            # Ensure every term has a positive IDF floor so matches in small corpora or frequent terms score > 0
            for term, val in self.bm25.idf.items():
                if val <= 0.0:
                    self.bm25.idf[term] = 0.25
            logger.info(f"BM25 index built with {len(self.chunks)} chunks.")
        else:
            self.bm25 = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> List[Tuple[VectorChunk, float]]:
        """Search BM25 index with optional source and category filters.

        Returns:
            List of (VectorChunk, score) tuples ranked in descending score order.
        """
        if not self.bm25 or not self.chunks or not query.strip():
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Pair chunks with their BM25 score
        scored_chunks: List[Tuple[VectorChunk, float]] = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]

            # Filter by source
            if source_filter and source_filter.lower() != "all":
                if chunk.source.lower() != source_filter.lower():
                    continue

            # Filter by category
            if category_filter:
                if chunk.category.value.lower() != category_filter.lower():
                    continue

            if score > 0.0:
                scored_chunks.append((chunk, float(score)))

        # Sort descending by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
