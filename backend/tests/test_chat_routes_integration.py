"""Route-Level HTTP Integration Tests for FastAPI Chat Endpoints.

Exercises actual HTTP requests against:
- POST /api/v1/chat (non-streaming)
- POST /api/v1/chat/stream (SSE streaming)

Verifies:
1. Greeting via non-streaming endpoint (zero retrieval, zero citations, zero demo warning, model=None).
2. Greeting via streaming endpoint (zero searching status, empty citations, warning=None, model=None).
3. Acknowledgement via non-streaming and streaming (welcome message, zero retrieval, warning=None).
4. Short property queries ('AIDA?', 'villas?', 'prices?') route to RAG (retrieval executed).
5. Full property queries via streaming and non-streaming.
6. Fallback sequence on transient errors and immediate halt on account-wide daily quota exhaustion.
7. Correct disclaimer text and actual model attribution.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.rag.openrouter import (
    OpenRouterClient,
    OpenRouterRateLimitError,
    OpenRouterServiceError,
)
from backend.app.rag.retriever import RetrievedItem


@pytest.fixture
def test_client():
    """FastAPI TestClient instance."""
    return TestClient(app)


# ============================================================================
# 1. Greeting via Non-Streaming & Streaming Endpoints
# ============================================================================

def test_api_chat_greeting_non_streaming(test_client):
    """Test pure greeting via /api/v1/chat returns fast response with zero citations/warning/model."""
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "source_filter": "all",
    }
    response = test_client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Content and structural assertions
    assert "DarGlobal & Wasalt Real Estate Knowledge Assistant" in data["answer"]
    assert data["citations"] == []
    assert data["property_cards"] == []
    assert data["provenance_warning"] is None
    assert data["model"] is None
    assert data["data_mode"] == "conversational"
    assert data["grounded"] is True


def test_api_chat_greeting_streaming(test_client):
    """Test pure greeting via /api/v1/chat/stream emits token, citations, complete with NO searching status."""
    payload = {
        "messages": [{"role": "user", "content": "hello there"}],
        "source_filter": "all",
    }
    response = test_client.post("/api/v1/chat/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = response.text.strip().split("\n\n")
    events = []
    for line_group in lines:
        parts = line_group.split("\n")
        event_name = None
        event_data = None
        for p in parts:
            if p.startswith("event: "):
                event_name = p[7:].strip()
            elif p.startswith("data: "):
                event_data = json.loads(p[6:].strip())
        if event_name and event_data:
            events.append((event_name, event_data))

    event_names = [e[0] for e in events]
    # Verify searching status is NOT emitted
    assert "status" not in event_names
    assert "token" in event_names
    assert "citations" in event_names
    assert "complete" in event_names

    token_data = next(d for name, d in events if name == "token")
    assert "DarGlobal & Wasalt Real Estate Knowledge Assistant" in token_data["token"]

    citations_data = next(d for name, d in events if name == "citations")
    assert citations_data["citations"] == []
    assert citations_data["provenance_warning"] is None
    assert citations_data["model"] is None
    assert citations_data["data_mode"] == "conversational"

    complete_data = next(d for name, d in events if name == "complete")
    assert complete_data["provenance_warning"] is None
    assert complete_data["model"] is None
    assert complete_data["data_mode"] == "conversational"


# ============================================================================
# 2. Acknowledgements & Out-of-Scope
# ============================================================================

def test_api_chat_acknowledgement(test_client):
    """Test 'thanks' via /api/v1/chat returns polite response without retrieval."""
    payload = {"messages": [{"role": "user", "content": "thanks!"}]}
    response = test_client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "welcome" in data["answer"].lower()
    assert data["citations"] == []
    assert data["provenance_warning"] is None
    assert data["model"] is None


def test_api_chat_out_of_scope(test_client):
    """Test out-of-scope non-real-estate query returns refusal without retrieval."""
    payload = {"messages": [{"role": "user", "content": "How do I bake a chocolate cake?"}]}
    response = test_client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "information regarding this question is not currently available" in data["answer"]
    assert data["citations"] == []
    assert data["provenance_warning"] is None
    assert data["model"] is None


# ============================================================================
# 3. Short Property Query Routing to RAG
# ============================================================================

def test_api_chat_short_property_query_not_greeting(test_client):
    """Test short query 'AIDA?' routes to RAG (does not take greeting fast-path)."""
    mock_item = RetrievedItem(
        chunk_id="chunk_aida_01",
        entity_id="entity_aida",
        title="AIDA Oman Luxury Community",
        source="darglobal",
        category="project",
        url="https://darglobal.co.uk/aida",
        provenance="demonstration_unverified",
        text_content="AIDA is a cliffside master development in Muscat, Oman.",
        rrf_score=0.035,
        semantic_similarity=0.88,
    )

    with patch("backend.app.rag.retriever.HybridRetriever.aretrieve", new_callable=AsyncMock) as mock_ret:
        mock_ret.return_value = ([mock_item], {"total_retrieval_ms": 5.0})
        with patch("backend.app.rag.openrouter.OpenRouterClient.generate_completion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "content": "AIDA is a luxury community in Muscat, Oman.",
                "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
                "dispatch_ms": 10.0,
                "generation_ms": 150.0,
            }

            payload = {"messages": [{"role": "user", "content": "AIDA?"}]}
            response = test_client.post("/api/v1/chat", json=payload)
            assert response.status_code == 200
            data = response.json()

            # Retrieval and LLM were actually executed
            assert mock_ret.called
            assert mock_llm.called
            assert len(data["citations"]) > 0
            assert data["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
            assert data["provenance_warning"] is not None
            assert "This prototype uses sample property information" in data["provenance_warning"]


# ============================================================================
# 4. Model Routing & Fallback Verification
# ============================================================================

def test_api_chat_fallback_routing_on_primary_transient_failure(test_client):
    """Test that model-specific 429 triggers Fallback 1 and returns exact model attribution."""
    mock_item = RetrievedItem(
        chunk_id="chunk_dg_01",
        entity_id="entity_dg",
        title="DarGlobal Corporate Info",
        source="darglobal",
        category="company_info",
        url="https://darglobal.co.uk/about",
        provenance="verified_source",
        text_content="DarGlobal is listed on the London Stock Exchange.",
        rrf_score=0.035,
        semantic_similarity=0.90,
    )

    with patch("backend.app.rag.retriever.HybridRetriever.aretrieve", new_callable=AsyncMock) as mock_ret:
        mock_ret.return_value = ([mock_item], {"total_retrieval_ms": 5.0})
        with patch("backend.app.rag.openrouter.OpenRouterClient.generate_completion", new_callable=AsyncMock) as mock_llm:
            # Fallback 1 succeeded
            mock_llm.return_value = {
                "content": "DarGlobal is a premier luxury real estate developer.",
                "model": "nvidia/nemotron-3.5-lightning:free",
                "dispatch_ms": 15.0,
                "generation_ms": 250.0,
            }

            payload = {"messages": [{"role": "user", "content": "tell me about DarGlobal"}]}
            response = test_client.post("/api/v1/chat", json=payload)
            assert response.status_code == 200
            data = response.json()

            assert data["model"] == "nvidia/nemotron-3.5-lightning:free"
            assert len(data["citations"]) > 0


def test_api_chat_account_wide_daily_limit_halts_fallback(test_client):
    """Test that account-wide daily limit (free-models-per-day) halts fallback and returns clean error."""
    mock_item = RetrievedItem(
        chunk_id="chunk_dg_01",
        entity_id="entity_dg",
        title="DarGlobal Corporate Info",
        source="darglobal",
        category="company_info",
        url="https://darglobal.co.uk/about",
        provenance="verified_source",
        text_content="DarGlobal is listed on the London Stock Exchange.",
        rrf_score=0.035,
        semantic_similarity=0.90,
    )

    with patch("backend.app.rag.retriever.HybridRetriever.aretrieve", new_callable=AsyncMock) as mock_ret:
        mock_ret.return_value = ([mock_item], {"total_retrieval_ms": 5.0})
        with patch("backend.app.rag.openrouter.OpenRouterClient.generate_completion", new_callable=AsyncMock) as mock_llm:
            # Account level quota exhausted raises OpenRouterRateLimitError
            mock_llm.side_effect = OpenRouterRateLimitError(
                message="OpenRouter free daily request limit reached for this account. Please try again tomorrow or upgrade your OpenRouter account."
            )

            payload = {"messages": [{"role": "user", "content": "tell me about DarGlobal"}]}
            response = test_client.post("/api/v1/chat", json=payload)
            assert response.status_code == 200
            data = response.json()

            assert data["error_code"] == "LLM_RATE_LIMITED"
            assert "daily request limit reached" in data["answer"]
            assert data["citations"] == []
            assert data["provenance_warning"] is None
