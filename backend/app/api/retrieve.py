"""Hybrid Retrieval API Endpoint Router.

Provides standalone retrieval and ranking across ChromaDB and BM25 indices,
supporting hybrid RRF fusion, pure semantic vector search, pure BM25 search,
and metadata filtering by source and category.
"""

import time
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, Field

from backend.app.core.logging import logger
from backend.app.core.security import limiter, sanitize_input_text
from backend.app.rag.embedder import Embedder
from backend.app.rag.keyword_search import BM25SearchEngine
from backend.app.rag.retriever import HybridRetriever, RetrievedItem
from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.schemas.document import ProvenanceType

router = APIRouter(prefix="/api/v1", tags=["Retrieval"])

# Global singleton retriever holder for reuse across requests
_retriever_instance: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Dependency provider for HybridRetriever with persistent ChromaDB and synced BM25."""
    global _retriever_instance
    if _retriever_instance is None:
        logger.info("Initializing HybridRetriever service dependency...")
        vector_store = ChromaVectorStore()
        bm25_engine = BM25SearchEngine()

        # Sync BM25 with chunks already present in ChromaDB
        existing_chunks = vector_store.get_all_chunks()
        if existing_chunks:
            bm25_engine.build_index(existing_chunks)
            logger.info(f"Initialized BM25 index with {len(existing_chunks)} existing ChromaDB chunks.")
        else:
            logger.warning("No existing chunks found in ChromaDB for BM25 initialization.")

        embedder = Embedder.get_instance()
        _retriever_instance = HybridRetriever(
            vector_store=vector_store,
            bm25_engine=bm25_engine,
            embedder=embedder,
            rrf_k=60,
        )
    return _retriever_instance


class RetrieveRequest(BaseModel):
    """Retrieval request payload."""

    query: str = Field(..., min_length=1, max_length=500, description="User search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of chunks to return")
    source_filter: Optional[Literal["all", "darglobal", "wasalt"]] = Field(
        default="all", description="Source domain filter"
    )
    category_filter: Optional[str] = Field(
        default=None, description="Content category filter (e.g. project, company_info, listing)"
    )
    method: Literal["hybrid", "semantic", "keyword"] = Field(
        default="hybrid", description="Retrieval search method"
    )
    similarity_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum cosine similarity threshold for semantic matches"
    )


class RetrievedItemResponse(BaseModel):
    """Ranked retrieved item response."""

    chunk_id: str
    entity_id: str
    title: str
    source: str
    category: str
    url: str
    provenance: ProvenanceType
    text_content: str
    score: float
    semantic_rank: Optional[int] = None
    semantic_similarity: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    """Retrieval endpoint response payload."""

    query: str
    total_found: int
    method: str
    latency_ms: float
    items: List[RetrievedItemResponse]


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve ranked context chunks",
)
@limiter.limit("120/minute")
async def retrieve_endpoint(
    request: Request,
    payload: RetrieveRequest,
    retriever: HybridRetriever = Depends(get_retriever),
) -> RetrieveResponse:
    """Retrieve ranked context chunks using hybrid, semantic, or keyword search."""
    start_time = time.perf_counter()
    sanitized_query = sanitize_input_text(payload.query)

    items = retriever.retrieve(
        query=sanitized_query,
        top_k=payload.top_k,
        source_filter=payload.source_filter if payload.source_filter != "all" else None,
        category_filter=payload.category_filter,
        method=payload.method,
    )

    # Optional similarity threshold filtering on semantic matches
    if payload.similarity_threshold is not None:
        filtered_items = []
        for it in items:
            if it.semantic_similarity is not None:
                if it.semantic_similarity >= payload.similarity_threshold:
                    filtered_items.append(it)
            else:
                filtered_items.append(it)
        items = filtered_items

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(
        f"Retrieval executed: query='{sanitized_query[:30]}...', method={payload.method}, "
        f"returned={len(items)}, latency={latency_ms}ms"
    )

    response_items = [
        RetrievedItemResponse(
            chunk_id=item.chunk_id,
            entity_id=item.entity_id,
            title=item.title,
            source=item.source,
            category=item.category,
            url=item.url,
            provenance=item.provenance,
            text_content=item.text_content,
            score=item.rrf_score,
            semantic_rank=item.semantic_rank,
            semantic_similarity=item.semantic_similarity,
            bm25_rank=item.bm25_rank,
            bm25_score=item.bm25_score,
            metadata=item.metadata,
        )
        for item in items
    ]

    return RetrieveResponse(
        query=sanitized_query,
        total_found=len(response_items),
        method=payload.method,
        latency_ms=latency_ms,
        items=response_items,
    )
