"""Hybrid Retrieval Engine combining Dense Semantic Vectors and Lexical BM25.

Merges semantic and keyword rankings using Reciprocal Rank Fusion (RRF)
with support for source and category metadata filtering.
"""

import asyncio
import time
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from backend.app.schemas.document import ProvenanceType
from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.rag.keyword_search import BM25SearchEngine
from backend.app.rag.embedder import Embedder
from backend.app.core.logging import logger


class RetrievedItem(BaseModel):
    """Ranked knowledge chunk resulting from hybrid retrieval."""

    chunk_id: str
    entity_id: str
    title: str
    source: str
    category: str
    url: str
    provenance: ProvenanceType
    text_content: str
    rrf_score: float = Field(default=0.0, description="Reciprocal Rank Fusion score")
    semantic_rank: Optional[int] = None
    semantic_similarity: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HybridRetriever:
    """Orchestrates hybrid semantic and keyword retrieval over indexed real estate chunks."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_engine: BM25SearchEngine,
        embedder: Optional[Embedder] = None,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_engine = bm25_engine
        self.embedder = embedder or Embedder.get_instance()
        self.rrf_k = rrf_k

    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        method: Literal["hybrid", "semantic", "keyword"] = "hybrid",
    ) -> Tuple[List[RetrievedItem], Dict[str, float]]:
        """Asynchronously perform concurrent hybrid retrieval with granular stage timings."""
        t_start = time.perf_counter()
        timings: Dict[str, float] = {
            "embedding_ms": 0.0,
            "vector_search_ms": 0.0,
            "bm25_search_ms": 0.0,
            "rrf_fusion_ms": 0.0,
            "total_retrieval_ms": 0.0,
        }

        cleaned_query = query.strip()
        if not cleaned_query:
            return [], timings

        async def _run_semantic():
            t_emb = time.perf_counter()
            query_vector = await asyncio.to_thread(self.embedder.embed_query, cleaned_query)
            emb_ms = (time.perf_counter() - t_emb) * 1000.0

            t_vec = time.perf_counter()
            sem_results = await asyncio.to_thread(
                self.vector_store.search,
                query_vector=query_vector,
                top_k=top_k * 2,
                source_filter=source_filter,
                category_filter=category_filter,
            )
            vec_ms = (time.perf_counter() - t_vec) * 1000.0
            return sem_results, emb_ms, vec_ms

        async def _run_bm25():
            t_bm = time.perf_counter()
            kw_results = await asyncio.to_thread(
                self.bm25_engine.search,
                query=cleaned_query,
                top_k=top_k * 2,
                source_filter=source_filter,
                category_filter=category_filter,
            )
            bm_ms = (time.perf_counter() - t_bm) * 1000.0
            return kw_results, bm_ms

        semantic_results: List[Dict[str, Any]] = []
        keyword_results = []

        if method == "hybrid":
            # Run semantic search and BM25 search concurrently in parallel threads
            (semantic_results, emb_ms, vec_ms), (keyword_results, bm_ms) = await asyncio.gather(
                _run_semantic(),
                _run_bm25(),
            )
            timings["embedding_ms"] = round(emb_ms, 2)
            timings["vector_search_ms"] = round(vec_ms, 2)
            timings["bm25_search_ms"] = round(bm_ms, 2)
        elif method == "semantic":
            semantic_results, emb_ms, vec_ms = await _run_semantic()
            timings["embedding_ms"] = round(emb_ms, 2)
            timings["vector_search_ms"] = round(vec_ms, 2)
        elif method == "keyword":
            keyword_results, bm_ms = await _run_bm25()
            timings["bm25_search_ms"] = round(bm_ms, 2)

        # Keyword-only ranking
        if method == "keyword":
            output = []
            for rank, (chunk, score) in enumerate(keyword_results[:top_k], 1):
                output.append(
                    RetrievedItem(
                        chunk_id=chunk.chunk_id,
                        entity_id=chunk.entity_id,
                        title=chunk.title,
                        source=chunk.source,
                        category=chunk.category.value,
                        url=chunk.url,
                        provenance=chunk.provenance,
                        text_content=chunk.text_content,
                        rrf_score=score,
                        bm25_rank=rank,
                        bm25_score=score,
                    )
                )
            timings["total_retrieval_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
            return output, timings

        # Semantic-only ranking
        if method == "semantic":
            output = []
            for rank, item in enumerate(semantic_results[:top_k], 1):
                output.append(
                    RetrievedItem(
                        chunk_id=item["chunk_id"],
                        entity_id=item["entity_id"],
                        title=item["title"],
                        source=item["source"],
                        category=item["category"],
                        url=item["url"],
                        provenance=item["provenance"],
                        text_content=item["text_content"],
                        rrf_score=item["similarity"],
                        semantic_rank=rank,
                        semantic_similarity=item["similarity"],
                        metadata=item.get("metadata", {}),
                    )
                )
            timings["total_retrieval_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
            return output, timings

        # Stage 6: Hybrid Reciprocal Rank Fusion (RRF)
        t_rrf = time.perf_counter()
        fused_items: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(semantic_results, 1):
            c_id = item["chunk_id"]
            rrf_val = 1.0 / (self.rrf_k + rank)
            fused_items[c_id] = {
                "chunk_id": c_id,
                "entity_id": item["entity_id"],
                "title": item["title"],
                "source": item["source"],
                "category": item["category"],
                "url": item["url"],
                "provenance": item["provenance"],
                "text_content": item["text_content"],
                "rrf_score": rrf_val,
                "semantic_rank": rank,
                "semantic_similarity": item["similarity"],
                "bm25_rank": None,
                "bm25_score": None,
                "metadata": item.get("metadata", {}),
            }

        for rank, (chunk, b_score) in enumerate(keyword_results, 1):
            c_id = chunk.chunk_id
            rrf_val = 1.0 / (self.rrf_k + rank)
            if c_id in fused_items:
                fused_items[c_id]["rrf_score"] += rrf_val
                fused_items[c_id]["bm25_rank"] = rank
                fused_items[c_id]["bm25_score"] = b_score
            else:
                fused_items[c_id] = {
                    "chunk_id": c_id,
                    "entity_id": chunk.entity_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "category": chunk.category.value,
                    "url": chunk.url,
                    "provenance": chunk.provenance,
                    "text_content": chunk.text_content,
                    "rrf_score": rrf_val,
                    "semantic_rank": None,
                    "semantic_similarity": None,
                    "bm25_rank": rank,
                    "bm25_score": b_score,
                    "metadata": {},
                }

        ranked_list = list(fused_items.values())
        ranked_list.sort(key=lambda x: x["rrf_score"], reverse=True)
        timings["rrf_fusion_ms"] = round((time.perf_counter() - t_rrf) * 1000.0, 2)
        timings["total_retrieval_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)

        return [RetrievedItem(**item) for item in ranked_list[:top_k]], timings

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        method: Literal["hybrid", "semantic", "keyword"] = "hybrid",
    ) -> List[RetrievedItem]:
        """Perform retrieval using the specified method (hybrid, semantic, or keyword)."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        # 1. Semantic Retrieval
        semantic_results: List[Dict[str, Any]] = []
        if method in ("hybrid", "semantic"):
            try:
                query_vector = self.embedder.embed_query(cleaned_query)
                semantic_results = self.vector_store.search(
                    query_vector=query_vector,
                    top_k=top_k * 2,  # Overfetch for fusion
                    source_filter=source_filter,
                    category_filter=category_filter,
                )
            except Exception as e:
                logger.error(f"Semantic search error: {e}")

        # 2. Keyword BM25 Retrieval
        keyword_results = []
        if method in ("hybrid", "keyword"):
            try:
                keyword_results = self.bm25_engine.search(
                    query=cleaned_query,
                    top_k=top_k * 2,
                    source_filter=source_filter,
                    category_filter=category_filter,
                )
            except Exception as e:
                logger.error(f"Keyword search error: {e}")

        # If keyword only
        if method == "keyword":
            output = []
            for rank, (chunk, score) in enumerate(keyword_results[:top_k], 1):
                output.append(
                    RetrievedItem(
                        chunk_id=chunk.chunk_id,
                        entity_id=chunk.entity_id,
                        title=chunk.title,
                        source=chunk.source,
                        category=chunk.category.value,
                        url=chunk.url,
                        provenance=chunk.provenance,
                        text_content=chunk.text_content,
                        rrf_score=score,
                        bm25_rank=rank,
                        bm25_score=score,
                    )
                )
            return output

        # If semantic only
        if method == "semantic":
            output = []
            for rank, item in enumerate(semantic_results[:top_k], 1):
                output.append(
                    RetrievedItem(
                        chunk_id=item["chunk_id"],
                        entity_id=item["entity_id"],
                        title=item["title"],
                        source=item["source"],
                        category=item["category"],
                        url=item["url"],
                        provenance=item["provenance"],
                        text_content=item["text_content"],
                        rrf_score=item["similarity"],
                        semantic_rank=rank,
                        semantic_similarity=item["similarity"],
                        metadata=item.get("metadata", {}),
                    )
                )
            return output

        # 3. Hybrid Reciprocal Rank Fusion (RRF)
        fused_items: Dict[str, Dict[str, Any]] = {}

        # Add semantic ranks
        for rank, item in enumerate(semantic_results, 1):
            c_id = item["chunk_id"]
            rrf_val = 1.0 / (self.rrf_k + rank)
            fused_items[c_id] = {
                "chunk_id": c_id,
                "entity_id": item["entity_id"],
                "title": item["title"],
                "source": item["source"],
                "category": item["category"],
                "url": item["url"],
                "provenance": item["provenance"],
                "text_content": item["text_content"],
                "rrf_score": rrf_val,
                "semantic_rank": rank,
                "semantic_similarity": item["similarity"],
                "bm25_rank": None,
                "bm25_score": None,
                "metadata": item.get("metadata", {}),
            }

        # Add keyword BM25 ranks
        for rank, (chunk, b_score) in enumerate(keyword_results, 1):
            c_id = chunk.chunk_id
            rrf_val = 1.0 / (self.rrf_k + rank)
            if c_id in fused_items:
                fused_items[c_id]["rrf_score"] += rrf_val
                fused_items[c_id]["bm25_rank"] = rank
                fused_items[c_id]["bm25_score"] = b_score
            else:
                fused_items[c_id] = {
                    "chunk_id": c_id,
                    "entity_id": chunk.entity_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "category": chunk.category.value,
                    "url": chunk.url,
                    "provenance": chunk.provenance,
                    "text_content": chunk.text_content,
                    "rrf_score": rrf_val,
                    "semantic_rank": None,
                    "semantic_similarity": None,
                    "bm25_rank": rank,
                    "bm25_score": b_score,
                    "metadata": {},
                }

        # Sort descending by RRF score
        ranked_list = list(fused_items.values())
        ranked_list.sort(key=lambda x: x["rrf_score"], reverse=True)

        return [RetrievedItem(**item) for item in ranked_list[:top_k]]
