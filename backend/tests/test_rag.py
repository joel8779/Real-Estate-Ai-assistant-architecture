"""Unit and Integration Tests for Milestone 3: Hybrid Retrieval & ChromaDB Vector Store.

Tests chunking, embedding generation, BM25 keyword search, ChromaDB persistence,
hybrid reciprocal rank fusion (RRF), and the /api/v1/retrieve API endpoint.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.document import ContentCategory, NormalizedEntity, VectorChunk
from backend.app.rag.chunker import DocumentChunker
from backend.app.rag.embedder import Embedder
from backend.app.rag.keyword_search import BM25SearchEngine, tokenize_text
from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.rag.retriever import HybridRetriever, RetrievedItem
from backend.app.api.retrieve import get_retriever


@pytest.fixture
def sample_entity() -> NormalizedEntity:
    """Fixture providing a sample luxury villa NormalizedEntity."""
    return NormalizedEntity(
        id="doc_test_villa_001",
        source="darglobal",
        category=ContentCategory.PROJECT,
        url="https://darglobal.co.uk/projects/test-villa",
        title="Test Villa by Pagani in Dubai",
        headline="Luxury Waterfront Villa",
        description=(
            "Ultra-luxury private villa overlooking the Dubai Water Canal.\n\n"
            "Features 5 master bedrooms, private elevator, infinity pool, "
            "and bespoke Italian interior architecture designed by Pagani.\n\n"
            "Freehold investment zone with 100% foreign ownership and Golden Visa eligibility."
        ),
        developer="DarGlobal",
        country="United Arab Emirates",
        city="Dubai",
        property_type="Luxury Villa",
        price=15000000.0,
        price_currency="AED",
        bedrooms=5,
        area_sqm=850.0,
        content_hash="mock_hash_villa_001",
        provenance="demonstration_unverified",
    )


@pytest.fixture
def sample_short_entity() -> NormalizedEntity:
    """Fixture providing a short NormalizedEntity that should remain in a single chunk."""
    return NormalizedEntity(
        id="doc_test_short_002",
        source="wasalt",
        category=ContentCategory.LISTING,
        url="https://wasalt.sa/en/property/test-short",
        title="Cozy Apartment in Riyadh",
        description="2 bedroom apartment in Al-Malqa, Riyadh. Price: 600,000 SAR.",
        country="Saudi Arabia",
        city="Riyadh",
        property_type="Apartment",
        price=600000.0,
        price_currency="SAR",
        bedrooms=2,
        area_sqm=120.0,
        content_hash="mock_hash_short_002",
        provenance="demonstration_unverified",
    )


# ============================================================================
# 1. Chunker Tests
# ============================================================================

def test_chunker_short_document_stays_intact(sample_short_entity: NormalizedEntity):
    chunker = DocumentChunker(target_chunk_size=400, chunk_overlap=50)
    chunks = chunker.chunk_entity(sample_short_entity)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "doc_test_short_002_chunk_0"
    assert chunk.entity_id == "doc_test_short_002"
    assert "Cozy Apartment in Riyadh" in chunk.text_content
    assert "600,000 SAR" in chunk.text_content
    assert chunk.provenance == "demonstration_unverified"


def test_chunker_long_document_splits_with_context_header(sample_entity: NormalizedEntity):
    # Using small chunk size to force multiple splits
    chunker = DocumentChunker(target_chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk_entity(sample_entity)

    assert len(chunks) > 1
    for idx, c in enumerate(chunks):
        assert c.chunk_id == f"doc_test_villa_001_chunk_{idx}"
        # Context header must be present in every chunk
        assert "[Title: Test Villa by Pagani in Dubai | Source: DARGLOBAL" in c.text_content
        assert "Developer: DarGlobal" in c.text_content
        assert "15,000,000 AED" in c.text_content
        assert c.provenance == "demonstration_unverified"
        assert c.bedrooms == 5
        assert c.price == 15000000.0


def test_chunker_batch_helper(sample_entity: NormalizedEntity, sample_short_entity: NormalizedEntity):
    chunker = DocumentChunker(target_chunk_size=500)
    chunks = chunker.chunk_documents([sample_entity, sample_short_entity])
    assert len(chunks) >= 2
    entity_ids = {c.entity_id for c in chunks}
    assert "doc_test_villa_001" in entity_ids
    assert "doc_test_short_002" in entity_ids


# ============================================================================
# 2. Embedder Tests
# ============================================================================

def test_embedder_dimensions_and_normalization():
    embedder = Embedder.get_instance()
    vec = embedder.embed_query("Luxury waterfront villa in Dubai")

    assert isinstance(vec, list)
    assert len(vec) == 384
    # Check L2 normalization: sum(v^2) should be approximately 1.0
    l2_norm = sum(x * x for x in vec) ** 0.5
    assert abs(l2_norm - 1.0) < 1e-4


def test_embedder_batch_documents():
    embedder = Embedder.get_instance()
    docs = [
        "Trump International Golf Club Oman in AIDA",
        "Villa for sale in Al-Mahdiyah Riyadh on Wasalt",
    ]
    vectors = embedder.embed_documents(docs)

    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def test_embedder_empty_batch():
    embedder = Embedder.get_instance()
    assert embedder.embed_documents([]) == []


# ============================================================================
# 3. BM25 Search Engine Tests
# ============================================================================

def test_bm25_tokenization():
    tokens = tokenize_text("DarGlobal & Wasalt: 5-bedroom villa in Riyadh 1,600,000 SAR!")
    assert "darglobal" in tokens
    assert "wasalt" in tokens
    assert "villa" in tokens
    assert "riyadh" in tokens
    assert "5-bedroom" in tokens or ("5" in tokens and "bedroom" in tokens)


def test_bm25_search_ranking():
    engine = BM25SearchEngine()
    chunk1 = VectorChunk(
        chunk_id="c1",
        entity_id="e1",
        source="darglobal",
        category=ContentCategory.PROJECT,
        title="Tierra Viva by Lamborghini",
        url="https://darglobal.co.uk/projects/tierra-viva",
        text_content="Super sports car luxury villas in Benahavís Spain by Automobili Lamborghini.",
        provenance="demonstration_unverified",
    )
    chunk2 = VectorChunk(
        chunk_id="c2",
        entity_id="e2",
        source="wasalt",
        category=ContentCategory.LISTING,
        title="Apartment in Al-Narjis Riyadh",
        url="https://wasalt.sa/en/property/narjis",
        text_content="Modern 3-bedroom apartment in Al-Narjis North Riyadh with smart home.",
        provenance="demonstration_unverified",
    )
    engine.build_index([chunk1, chunk2])

    results = engine.search("Lamborghini villas in Spain", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "c1"
    assert score > 0.0

    # Source filter check
    results_wasalt = engine.search("Lamborghini", top_k=2, source_filter="wasalt")
    assert len(results_wasalt) == 0


# ============================================================================
# 4. ChromaDB & Hybrid Retriever Integration Tests
# ============================================================================

def test_vector_store_persistence_and_search(tmp_path: Path):
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma_test", collection_name="test_col")
    embedder = Embedder.get_instance()

    chunk = VectorChunk(
        chunk_id="test_c1",
        entity_id="test_e1",
        source="darglobal",
        category=ContentCategory.PROJECT,
        title="DaVinci Tower by Pagani",
        url="https://darglobal.co.uk/projects/davinci",
        text_content="Luxury residential tower on the Dubai Water Canal designed with Pagani Automobili.",
        provenance="demonstration_unverified",
        city="Dubai",
        country="United Arab Emirates",
    )

    vec = embedder.embed_query(chunk.text_content)
    store.index_chunks(chunks=[chunk], embeddings=[vec])
    assert store.count() == 1

    # Search with query
    query_vec = embedder.embed_query("Pagani residential tower on Dubai Canal")
    search_results = store.search(query_vector=query_vec, top_k=2)

    assert len(search_results) == 1
    assert search_results[0]["chunk_id"] == "test_c1"
    assert search_results[0]["similarity"] > 0.5

    # Incremental update check: re-indexing test_e1 should not duplicate
    store.index_chunks(chunks=[chunk], embeddings=[vec])
    assert store.count() == 1


def test_hybrid_retriever_fusion(tmp_path: Path):
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma_hybrid_test", collection_name="hybrid_col")
    embedder = Embedder.get_instance()
    bm25 = BM25SearchEngine()

    chunks = [
        VectorChunk(
            chunk_id="c_trump",
            entity_id="e_trump",
            source="darglobal",
            category=ContentCategory.PROJECT,
            title="AIDA Oman Trump Golf Course",
            url="https://darglobal.co.uk/projects/aida",
            text_content="Championship 18-hole golf course developed with Trump International in Muscat Oman.",
            provenance="demonstration_unverified",
        ),
        VectorChunk(
            chunk_id="c_villa_riyadh",
            entity_id="e_villa_riyadh",
            source="wasalt",
            category=ContentCategory.LISTING,
            title="Villa in Al-Mahdiyah Riyadh",
            url="https://wasalt.sa/en/property/villa",
            text_content="333 SQM villa for sale in Al-Mahdiyah Riyadh with 5 bedrooms and 20m street width.",
            provenance="demonstration_unverified",
        ),
    ]

    embeddings = embedder.embed_documents([c.text_content for c in chunks])
    store.index_chunks(chunks=chunks, embeddings=embeddings)
    bm25.build_index(chunks)

    retriever = HybridRetriever(vector_store=store, bm25_engine=bm25, embedder=embedder)

    # Hybrid search
    results = retriever.retrieve("Trump golf course in Oman", top_k=2, method="hybrid")
    assert len(results) > 0
    assert results[0].chunk_id == "c_trump"
    assert results[0].rrf_score > 0.0
    assert results[0].semantic_rank == 1
    assert results[0].bm25_rank == 1


# ============================================================================
# 5. Retrieval API Endpoint Tests (/api/v1/retrieve)
# ============================================================================

def test_api_retrieve_success():
    client = TestClient(app)
    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "AIDA Trump Golf Club Muscat Oman",
            "top_k": 3,
            "source_filter": "all",
            "method": "hybrid",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "AIDA Trump Golf Club Muscat Oman"
    assert data["total_found"] > 0
    assert data["method"] == "hybrid"
    assert data["latency_ms"] >= 0.0

    first_item = data["items"][0]
    assert "Trump" in first_item["text_content"] or "AIDA" in first_item["title"]
    assert first_item["provenance"] == "demonstration_unverified"
    assert first_item["url"].startswith("http")


def test_api_retrieve_with_source_filter():
    client = TestClient(app)
    # Search with source_filter='wasalt'
    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "apartment or villa for sale",
            "top_k": 3,
            "source_filter": "wasalt",
            "method": "hybrid",
        },
    )
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        assert item["source"] == "wasalt"


def test_api_retrieve_empty_query_rejected():
    client = TestClient(app)
    response = client.post(
        "/api/v1/retrieve",
        json={"query": ""},
    )
    assert response.status_code == 422


def test_api_retrieve_oversized_query_rejected():
    client = TestClient(app)
    response = client.post(
        "/api/v1/retrieve",
        json={"query": "A" * 600},
    )
    assert response.status_code == 422
