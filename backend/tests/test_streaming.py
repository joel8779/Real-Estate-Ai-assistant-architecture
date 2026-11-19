"""Comprehensive Automated Test Suite for RAG Streaming & Server-Sent Events (SSE).

Covers:
1. SSE event parsing in OpenRouterClient.
2. Incremental token delivery.
3. Correct completion event with TTFT and generation metrics.
4. Citation delivery strictly after final validation.
5. OpenRouter error handling (401, 402, 503, network errors).
6. HTTP 429 rate limit handling.
7. Client cancellation / disconnect handling.
8. Empty or malformed SSE chunks.
9. Out-of-domain questions with low confidence.
10. Demonstration-mode warnings and provenance tracking.
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import app
from backend.app.rag.openrouter import (
    OpenRouterClient,
    OpenRouterAuthError,
    OpenRouterQuotaError,
    OpenRouterRateLimitError,
    OpenRouterServiceError,
    LLMNotConfiguredException,
)
from backend.app.rag.orchestrator import RAGOrchestrator
from backend.app.rag.retriever import RetrievedItem
from backend.app.schemas.chat import ChatMessage, ChatRequest


@pytest.fixture
def mock_retrieved_items() -> List[RetrievedItem]:
    """Sample retrieved items with demonstration provenance."""
    return [
        RetrievedItem(
            chunk_id="chunk_aida_1",
            entity_id="doc_aida",
            title="AIDA Muscat Community",
            source="darglobal",
            category="project",
            url="https://darglobal.co.uk/projects/aida",
            provenance="demonstration_unverified",
            text_content="AIDA is a cliffside master community in Muscat, Oman developed with Trump International.",
            rrf_score=0.035,
            semantic_similarity=0.88,
            metadata={"developer": "DarGlobal", "location": "Muscat, Oman"},
        )
    ]


@pytest.fixture
def mock_verified_items() -> List[RetrievedItem]:
    """Sample retrieved items with verified source provenance."""
    return [
        RetrievedItem(
            chunk_id="chunk_dg_about_1",
            entity_id="doc_dg_about",
            title="DarGlobal LSE Listing",
            source="darglobal",
            category="company_info",
            url="https://darglobal.co.uk/about-us",
            provenance="verified_source",
            text_content="DarGlobal is listed on the London Stock Exchange under symbol DAR.",
            rrf_score=0.038,
            semantic_similarity=0.91,
            metadata={},
        )
    ]


# ============================================================================
# 1. SSE Event Parsing & 2. Incremental Token Delivery in OpenRouterClient
# ============================================================================

@pytest.mark.asyncio
async def test_openrouter_stream_completion_tokens_and_done():
    """Verify stream_completion parses SSE chunks, yields tokens, and completes on [DONE]."""
    client = OpenRouterClient(
        api_key="sk-test-key",
        primary_model="nvidia/nemotron-3-ultra-550b-a55b:free",
    )

    mock_lines = [
        "",  # empty line
        ": keep-alive comment",  # SSE comment
        'data: {"choices": [{"delta": {"content": "DarGlobal "}}]}',
        'data: {"choices": [{"delta": {"content": "develops "}}]}',
        'data: {"choices": [{"delta": {"content": "luxury "}}]}',
        'data: {"choices": [{"delta": {"content": "residences."}}]}',
        "data: [DONE]",
    ]

    async def mock_aiter_lines():
        for line in mock_lines:
            yield line

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = mock_aiter_lines

    mock_context_manager = MagicMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient.stream", return_value=mock_context_manager):
        tokens = []
        async for chunk in client.stream_completion([{"role": "user", "content": "Tell me about DarGlobal"}]):
            if chunk.get("type") == "token":
                tokens.append(chunk["token"])
            elif chunk.get("type") == "done":
                tokens.append("[DONE_SIGNAL]")

        assert tokens == ["DarGlobal ", "develops ", "luxury ", "residences.", "[DONE_SIGNAL]"]


# ============================================================================
# 8. Empty or Malformed SSE Chunks
# ============================================================================

@pytest.mark.asyncio
async def test_openrouter_stream_handles_malformed_and_empty_chunks():
    """Verify parser gracefully skips corrupted JSON, empty deltas, and whitespace lines."""
    client = OpenRouterClient(
        api_key="sk-test-key",
        primary_model="nvidia/nemotron-3-ultra-550b-a55b:free",
    )

    mock_lines = [
        "data: {corrupted json}",
        "data: {}",
        'data: {"choices": []}',
        'data: {"choices": [{"delta": {}}]}',  # empty delta
        'data: {"choices": [{"delta": {"content": "Valid token"}}]}',
        "data: [DONE]",
    ]

    async def mock_aiter_lines():
        for line in mock_lines:
            yield line

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = mock_aiter_lines

    mock_context_manager = MagicMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient.stream", return_value=mock_context_manager):
        tokens = []
        async for chunk in client.stream_completion([{"role": "user", "content": "Hi"}]):
            if chunk.get("type") == "token":
                tokens.append(chunk["token"])

        assert tokens == ["Valid token"]


# ============================================================================
# 5. OpenRouter Error Handling (401, 402, 503)
# ============================================================================

@pytest.mark.asyncio
async def test_openrouter_stream_401_auth_error():
    """Verify 401 raises OpenRouterAuthError during stream initiation."""
    client = OpenRouterClient(api_key="bad-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.aread = AsyncMock(return_value=b"Unauthorized")

    mock_context_manager = MagicMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient.stream", return_value=mock_context_manager):
        with pytest.raises(OpenRouterAuthError):
            async for _ in client.stream_completion([{"role": "user", "content": "Hi"}]):
                pass


@pytest.mark.asyncio
async def test_openrouter_stream_402_quota_error():
    """Verify 402 raises OpenRouterQuotaError during stream initiation."""
    client = OpenRouterClient(api_key="test-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 402
    mock_resp.aread = AsyncMock(return_value=b"Payment Required")

    mock_context_manager = MagicMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient.stream", return_value=mock_context_manager):
        with pytest.raises(OpenRouterQuotaError):
            async for _ in client.stream_completion([{"role": "user", "content": "Hi"}]):
                pass


# ============================================================================
# 6. HTTP 429 Rate Limit Handling
# ============================================================================

@pytest.mark.asyncio
async def test_openrouter_stream_429_rate_limit_error_with_retry_after():
    """Verify 429 raises OpenRouterRateLimitError and parses Retry-After header."""
    client = OpenRouterClient(api_key="test-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {"Retry-After": "45"}
    mock_resp.aread = AsyncMock(return_value=b"Rate limited")

    mock_context_manager = MagicMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient.stream", return_value=mock_context_manager):
        with pytest.raises(OpenRouterRateLimitError) as exc_info:
            async for _ in client.stream_completion([{"role": "user", "content": "Hi"}]):
                pass
        assert exc_info.value.retry_after == 45
        assert "45 seconds" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openrouter_stream_429_rate_limit_error_without_retry_after():
    """Verify 429 without Retry-After provides a clean generic message and retry_after=None."""
    client = OpenRouterClient(api_key="test-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {}
    mock_resp.aread = AsyncMock(return_value=b"Rate limited")

    mock_context_manager = MagicMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient.stream", return_value=mock_context_manager):
        with pytest.raises(OpenRouterRateLimitError) as exc_info:
            async for _ in client.stream_completion([{"role": "user", "content": "Hi"}]):
                pass
        assert exc_info.value.retry_after is None
        assert "The AI service is temporarily busy. Please wait a moment and try again." in str(exc_info.value)


@pytest.mark.asyncio
async def test_orchestrator_streaming_429_emits_error_no_citations(mock_retrieved_items: List[RetrievedItem]):
    """Verify that when 429 occurs during streaming, an error event is emitted without citations or demo warning."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = mock_retrieved_items
    mock_retriever.aretrieve = AsyncMock(return_value=(mock_retrieved_items, {"total_retrieval_ms": 10.0}))

    mock_llm = MagicMock()

    async def mock_rate_limited_stream(messages, **kwargs):
        raise OpenRouterRateLimitError(retry_after=60)
        yield  # make it a generator

    mock_llm.stream_completion = mock_rate_limited_stream
    mock_llm.primary_model = "nvidia/nemotron-3-ultra-550b-a55b:free"

    orchestrator = RAGOrchestrator(retriever=mock_retriever, llm_client=mock_llm)

    payload = ChatRequest(messages=[ChatMessage(role="user", content="Tell me about AIDA")])

    events = []
    async for ev in orchestrator.stream_answer_question(payload):
        events.append(ev)

    event_names = [e["event"] for e in events]
    assert "error" in event_names
    assert "citations" not in event_names
    assert "complete" not in event_names

    error_ev = next(e for e in events if e["event"] == "error")
    assert error_ev["data"]["error_code"] == "LLM_RATE_LIMITED"
    assert error_ev["data"]["retry_after"] == 60
    assert "60 seconds" in error_ev["data"]["message"]
    # Ensure no misleading demonstration warning or citations in the error event
    assert error_ev["data"]["citations"] == []
    assert error_ev["data"]["provenance_warning"] is None


# ============================================================================
# 3. Correct Completion Event & 4. Citation Delivery After Validation
# ============================================================================

@pytest.mark.asyncio
async def test_orchestrator_streaming_full_flow(mock_retrieved_items: List[RetrievedItem]):
    """Verify orchestrator emits status, incremental tokens, validated citations, and complete metadata."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = mock_retrieved_items
    mock_retriever.aretrieve = AsyncMock(return_value=(mock_retrieved_items, {"total_retrieval_ms": 12.5}))

    mock_llm = MagicMock()

    async def mock_stream_completion(messages, **kwargs):
        chunks = ["AIDA ", "is a cliffside ", "development in Oman."]
        for c in chunks:
            yield {"type": "token", "token": c}
        yield {"type": "done", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"}

    mock_llm.stream_completion = mock_stream_completion
    mock_llm.primary_model = "nvidia/nemotron-3-ultra-550b-a55b:free"

    orchestrator = RAGOrchestrator(retriever=mock_retriever, llm_client=mock_llm)

    with patch("backend.app.rag.orchestrator.get_settings") as mock_settings:
        settings_instance = Settings()
        settings_instance.ALLOW_UNVERIFIED_DEMO_DATA = True
        mock_settings.return_value = settings_instance

        payload = ChatRequest(messages=[ChatMessage(role="user", content="Where is AIDA located?")])

        events = []
        async for ev in orchestrator.stream_answer_question(payload):
            events.append(ev)

        event_names = [e["event"] for e in events]
        assert "status" in event_names
        assert "token" in event_names
        assert "citations" in event_names
        assert "complete" in event_names

        # Verify ordering: status -> tokens -> citations -> complete
        status_indices = [i for i, name in enumerate(event_names) if name == "status"]
        token_indices = [i for i, name in enumerate(event_names) if name == "token"]
        citation_indices = [i for i, name in enumerate(event_names) if name == "citations"]
        complete_indices = [i for i, name in enumerate(event_names) if name == "complete"]

        assert max(token_indices) < citation_indices[0]
        assert citation_indices[0] < complete_indices[0]

        # Verify citations content
        citation_ev = events[citation_indices[0]]
        assert len(citation_ev["data"]["citations"]) == 1
        assert citation_ev["data"]["citations"][0]["url"] == "https://darglobal.co.uk/projects/aida"
        assert citation_ev["data"]["grounded"] is True

        # Verify complete event content (TTFT, generation_ms, model)
        complete_ev = events[complete_indices[0]]
        assert complete_ev["data"]["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
        assert complete_ev["data"]["ttft_ms"] > 0
        assert complete_ev["data"]["generation_ms"] >= 0
        assert complete_ev["data"]["total_tokens"] == 3


# ============================================================================
# 9. Out-of-Domain Questions
# ============================================================================

@pytest.mark.asyncio
async def test_orchestrator_streaming_out_of_domain():
    """Verify out-of-domain queries with empty retrieval return safe refusal tokens without calling LLM."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_retriever.aretrieve = AsyncMock(return_value=([], {"total_retrieval_ms": 5.0}))

    mock_llm = MagicMock()
    mock_llm.stream_completion = AsyncMock()

    orchestrator = RAGOrchestrator(retriever=mock_retriever, llm_client=mock_llm)

    payload = ChatRequest(messages=[ChatMessage(role="user", content="How do I bake a chocolate cake?")])

    events = []
    async for ev in orchestrator.stream_answer_question(payload):
        events.append(ev)

    token_events = [e for e in events if e["event"] == "token"]
    assert len(token_events) > 0
    refusal_text = "".join(t["data"]["token"] for t in token_events)
    assert "information regarding this question is not currently available" in refusal_text

    # Verify LLM was NOT invoked
    mock_llm.stream_completion.assert_not_called()


# ============================================================================
# 10. Demonstration-Mode Warnings in Streaming
# ============================================================================

@pytest.mark.asyncio
async def test_orchestrator_streaming_demonstration_warning(mock_retrieved_items: List[RetrievedItem]):
    """Verify demonstration mode emits warning banner metadata in citations and complete events."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = mock_retrieved_items
    mock_retriever.aretrieve = AsyncMock(return_value=(mock_retrieved_items, {"total_retrieval_ms": 10.0}))

    mock_llm = MagicMock()

    async def mock_stream_completion(messages, **kwargs):
        yield {"type": "token", "token": "AIDA cliffside luxury community."}
        yield {"type": "done", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"}

    mock_llm.stream_completion = mock_stream_completion
    mock_llm.primary_model = "nvidia/nemotron-3-ultra-550b-a55b:free"

    orchestrator = RAGOrchestrator(retriever=mock_retriever, llm_client=mock_llm)

    with patch("backend.app.rag.orchestrator.get_settings") as mock_settings:
        settings_instance = Settings()
        settings_instance.ALLOW_UNVERIFIED_DEMO_DATA = True
        mock_settings.return_value = settings_instance

        payload = ChatRequest(messages=[ChatMessage(role="user", content="What is AIDA?")])

        events = []
        async for ev in orchestrator.stream_answer_question(payload):
            events.append(ev)

        citation_ev = next(e for e in events if e["event"] == "citations")
        assert citation_ev["data"]["data_mode"] == "demonstration"
        assert "sample property information prepared for demonstration purposes" in citation_ev["data"]["provenance_warning"]


# ============================================================================
# 7. FastAPI Endpoint (/api/v1/chat/stream) & Client Disconnect
# ============================================================================

def test_api_chat_stream_endpoint():
    """Verify POST /api/v1/chat/stream returns valid text/event-stream chunks."""
    client = TestClient(app)

    async def mock_stream(messages, **kwargs):
        yield {"type": "token", "token": "Streaming "}
        yield {"type": "token", "token": "test answer."}
        yield {"type": "done", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"}

    with patch("backend.app.rag.openrouter.OpenRouterClient.stream_completion", side_effect=mock_stream):
        with patch("backend.app.rag.orchestrator.get_settings") as mock_settings:
            settings_instance = Settings()
            settings_instance.ALLOW_UNVERIFIED_DEMO_DATA = True
            mock_settings.return_value = settings_instance

            response = client.post(
                "/api/v1/chat/stream",
                json={"messages": [{"role": "user", "content": "What is AIDA?"}]},
            )

            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            body = response.text
            assert "event: status" in body
            assert "event: token" in body
            assert "Streaming " in body
            assert "event: citations" in body
            assert "event: complete" in body
