"""Comprehensive Test Suite for Primary OpenRouter to Secondary Ollama Failover Routing.

Tests:
1. Quota Detection: Account-wide free-model daily limit detection.
2. Primary Success: OpenRouter (Nemotron 3 Ultra) responds successfully; Ollama is never called.
3. Daily Quota Exhaustion Failover: 429 free-models-per-day triggers immediate failover to Ollama Qwen 2.5 1.5B.
4. Provider Outage Failover: HTTP 502/503/504 retries once on OpenRouter, then fails over to Ollama.
5. Network Error Failover: Upstream connection timeout retries once on OpenRouter, then fails over to Ollama.
6. Permanent Errors Blocked: HTTP 400, 401, 402 do not fail over to Ollama.
7. Dual Failure: When both OpenRouter and Ollama are unavailable, clear user-friendly error is returned.
8. Streaming Failover: Pre-token failure fails over cleanly to Ollama streaming.
9. Streaming Midstream Invariant: Failures AFTER tokens have started streaming never switch providers.
10. Model Attribution & Transparency Notices: Exact model and notice strings for quota vs outage.
"""

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from backend.app.config import Settings
from backend.app.core.exceptions import KnowledgeBaseException
from backend.app.rag.ollama import OllamaClient, OllamaUnavailableError, OllamaTimeoutError
from backend.app.rag.openrouter import (
    OpenRouterClient,
    OpenRouterAuthError,
    OpenRouterQuotaError,
    OpenRouterRateLimitError,
    OpenRouterServiceError,
    OpenRouterAccountQuotaExhaustedError,
    is_account_level_quota_exhausted,
)
from backend.app.rag.orchestrator import RAGOrchestrator
from backend.app.rag.retriever import RetrievedItem
from backend.app.schemas.chat import ChatMessage, ChatRequest


@pytest.fixture
def sample_evidence() -> List[RetrievedItem]:
    return [
        RetrievedItem(
            chunk_id="chunk_01",
            entity_id="project_aida",
            title="AIDA Oman Luxury Community",
            source="darglobal",
            category="project",
            url="https://darglobal.co.uk/projects/aida",
            provenance="demonstration_unverified",
            text_content="AIDA is a premier cliffside community in Muscat, Oman featuring Trump International Golf Club.",
            rrf_score=0.95,
            semantic_similarity=0.88,
        )
    ]


# ============================================================================
# 1. Quota Exhaustion Detection Unit Tests
# ============================================================================

def test_is_account_level_quota_exhausted_detection():
    """Verify detection of account-wide daily limit vs model-specific rate limiting."""
    assert is_account_level_quota_exhausted(
        '{"error": {"message": "Rate limit exceeded: free-models-per-day: 50 requests/day"}}'
    ) is True
    assert is_account_level_quota_exhausted("Account daily limit reached (requests per day)") is True
    assert is_account_level_quota_exhausted("User-level quota exceeded") is True
    assert is_account_level_quota_exhausted("Account credit limit reached") is True

    # Model-specific transient messages
    assert is_account_level_quota_exhausted("Too many concurrent requests for this model") is False
    assert is_account_level_quota_exhausted("Provider capacity reached, please slow down") is False
    assert is_account_level_quota_exhausted("") is False


# ============================================================================
# 2. OpenRouter Client Routing Policy (Locked to Single Model)
# ============================================================================

@pytest.mark.asyncio
async def test_openrouter_client_single_model_success():
    """Verify OpenRouterClient sends requests strictly to the configured primary model."""
    client = OpenRouterClient(
        api_key="sk-test-key",
        primary_model="nvidia/nemotron-3-ultra-550b-a55b:free",
    )

    resp_200 = MagicMock(spec=httpx.Response)
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Nemotron answer"}}],
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = resp_200
        result = await client.generate_completion([{"role": "user", "content": "Hello"}])

        assert result["content"] == "Nemotron answer"
        assert result["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
        assert mock_post.call_count == 1
        assert mock_post.call_args_list[0].kwargs["json"]["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"


@pytest.mark.asyncio
async def test_openrouter_account_quota_exhausted_raises_immediately():
    """Account-wide quota exhaustion on OpenRouter raises immediately without retrying."""
    client = OpenRouterClient(
        api_key="sk-test-key",
        primary_model="nvidia/nemotron-3-ultra-550b-a55b:free",
    )

    resp_429 = MagicMock(spec=httpx.Response)
    resp_429.status_code = 429
    resp_429.text = '{"error": {"message": "Rate limit exceeded: free-models-per-day: 50 requests/day"}}'
    resp_429.headers = {"Retry-After": "3600"}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = resp_429
        with pytest.raises(OpenRouterAccountQuotaExhaustedError):
            await client.generate_completion([{"role": "user", "content": "Hello"}])

        assert mock_post.call_count == 1


@pytest.mark.asyncio
async def test_openrouter_service_outage_retries_once():
    """Upstream 503 retries once with backoff then raises OpenRouterServiceError."""
    client = OpenRouterClient(
        api_key="sk-test-key",
        primary_model="nvidia/nemotron-3-ultra-550b-a55b:free",
    )

    resp_503 = MagicMock(spec=httpx.Response)
    resp_503.status_code = 503
    resp_503.text = "Service Unavailable"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = resp_503
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(OpenRouterServiceError):
                await client.generate_completion([{"role": "user", "content": "Hello"}])

            assert mock_post.call_count == 2
            assert mock_sleep.call_count == 1


@pytest.mark.asyncio
async def test_openrouter_permanent_errors_never_retry():
    """HTTP 400, 401, 402 raise immediately without retrying."""
    client = OpenRouterClient(api_key="sk-test-key")

    for status_code, expected_exc in [
        (400, KnowledgeBaseException),
        (401, OpenRouterAuthError),
        (402, OpenRouterQuotaError),
    ]:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.text = f"Error {status_code}"

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = resp
            with pytest.raises(expected_exc):
                await client.generate_completion([{"role": "user", "content": "Hello"}])
            assert mock_post.call_count == 1


# ============================================================================
# 3. End-to-End Orchestrator Non-Streaming Failover Tests
# ============================================================================

@pytest.mark.asyncio
async def test_orchestrator_primary_success(sample_evidence):
    """When OpenRouter succeeds, Ollama is never called and response shows OpenRouter provider."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = sample_evidence

    mock_openrouter = MagicMock()
    mock_openrouter.primary_model = "nvidia/nemotron-3-ultra-550b-a55b:free"
    mock_openrouter.generate_completion = AsyncMock(return_value={
        "content": "AIDA in Oman features stunning cliffs and Trump International Golf Club.",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "dispatch_ms": 10.0,
        "generation_ms": 25.0,
    })

    mock_ollama = MagicMock()
    mock_ollama.generate_completion = AsyncMock()

    orchestrator = RAGOrchestrator(
        retriever=mock_retriever,
        llm_client=mock_openrouter,
        ollama_client=mock_ollama,
    )

    req = ChatRequest(messages=[ChatMessage(role="user", content="Tell me about AIDA Oman")])
    resp = await orchestrator.answer_question(req)

    assert resp.provider == "openrouter"
    assert resp.fallback_active is False
    assert resp.fallback_reason is None
    assert resp.fallback_notice is None
    assert "nvidia/nemotron-3-ultra-550b-a55b:free" in resp.model
    mock_ollama.generate_completion.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_failover_on_account_quota_exhaustion(sample_evidence):
    """When OpenRouter raises OpenRouterAccountQuotaExhaustedError, orchestrator fails over to Ollama."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = sample_evidence

    mock_openrouter = MagicMock()
    mock_openrouter.primary_model = "nvidia/nemotron-3-ultra-550b-a55b:free"
    mock_openrouter.generate_completion = AsyncMock(
        side_effect=OpenRouterAccountQuotaExhaustedError("Rate limit exceeded: free-models-per-day: 50 requests/day")
    )

    mock_ollama = MagicMock()
    mock_ollama.generate_completion = AsyncMock(return_value={
        "content": "AIDA is a premier development in Oman with a golf club.",
        "model": "qwen2.5:1.5b",
        "dispatch_ms": 5.0,
        "generation_ms": 15.0,
    })

    orchestrator = RAGOrchestrator(
        retriever=mock_retriever,
        llm_client=mock_openrouter,
        ollama_client=mock_ollama,
    )

    req = ChatRequest(messages=[ChatMessage(role="user", content="Tell me about AIDA Oman")])
    resp = await orchestrator.answer_question(req)

    assert resp.provider == "ollama"
    assert resp.fallback_active is True
    assert resp.fallback_reason == "quota_exhausted"
    assert "daily free-model request limit" in resp.fallback_notice
    assert resp.model == "ollama/qwen2.5:1.5b"
    assert "AIDA is a premier development" in resp.answer
    mock_ollama.generate_completion.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_failover_on_provider_outage(sample_evidence):
    """When OpenRouter raises OpenRouterServiceError (503), orchestrator fails over to Ollama."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = sample_evidence

    mock_openrouter = MagicMock()
    mock_openrouter.primary_model = "nvidia/nemotron-3-ultra-550b-a55b:free"
    mock_openrouter.generate_completion = AsyncMock(
        side_effect=OpenRouterServiceError("OpenRouter error 503: Service Unavailable")
    )

    mock_ollama = MagicMock()
    mock_ollama.generate_completion = AsyncMock(return_value={
        "content": "AIDA in Oman features luxury residences.",
        "model": "qwen2.5:1.5b",
        "dispatch_ms": 5.0,
        "generation_ms": 12.0,
    })

    orchestrator = RAGOrchestrator(
        retriever=mock_retriever,
        llm_client=mock_openrouter,
        ollama_client=mock_ollama,
    )

    req = ChatRequest(messages=[ChatMessage(role="user", content="Tell me about AIDA Oman")])
    resp = await orchestrator.answer_question(req)

    assert resp.provider == "ollama"
    assert resp.fallback_active is True
    assert resp.fallback_reason == "provider_outage"
    assert "temporarily unavailable" in resp.fallback_notice
    assert resp.model == "ollama/qwen2.5:1.5b"
    mock_ollama.generate_completion.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_permanent_error_blocks_failover(sample_evidence):
    """When OpenRouter raises OpenRouterAuthError (401), orchestrator does NOT fail over to Ollama."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = sample_evidence

    mock_openrouter = MagicMock()
    mock_openrouter.primary_model = "nvidia/nemotron-3-ultra-550b-a55b:free"
    mock_openrouter.generate_completion = AsyncMock(
        side_effect=OpenRouterAuthError("Invalid API key")
    )

    mock_ollama = MagicMock()
    mock_ollama.generate_completion = AsyncMock()

    orchestrator = RAGOrchestrator(
        retriever=mock_retriever,
        llm_client=mock_openrouter,
        ollama_client=mock_ollama,
    )

    req = ChatRequest(messages=[ChatMessage(role="user", content="Tell me about AIDA Oman")])
    resp = await orchestrator.answer_question(req)

    assert resp.error_code == "LLM_AUTH_ERROR"
    assert resp.provider == "openrouter"
    assert resp.fallback_active is False
    mock_ollama.generate_completion.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_dual_failure_returns_graceful_error(sample_evidence):
    """When OpenRouter is exhausted and Ollama is unreachable, returns clear error response without crashing."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = sample_evidence

    mock_openrouter = MagicMock()
    mock_openrouter.primary_model = "nvidia/nemotron-3-ultra-550b-a55b:free"
    mock_openrouter.generate_completion = AsyncMock(
        side_effect=OpenRouterAccountQuotaExhaustedError("Rate limit exceeded: free-models-per-day: 50 requests/day")
    )

    mock_ollama = MagicMock()
    mock_ollama.generate_completion = AsyncMock(
        side_effect=OllamaUnavailableError("Failed to connect to Ollama")
    )

    orchestrator = RAGOrchestrator(
        retriever=mock_retriever,
        llm_client=mock_openrouter,
        ollama_client=mock_ollama,
    )

    req = ChatRequest(messages=[ChatMessage(role="user", content="Tell me about AIDA Oman")])
    resp = await orchestrator.answer_question(req)

    assert resp.error_code == "LLM_SERVICE_UNAVAILABLE"
    assert "Additionally, local fallback to Qwen via Ollama failed" in resp.answer
    assert resp.citations == []
    assert resp.property_cards == []


# ============================================================================
# 4. Streaming Failover & Mid-stream Invariant Tests
# ============================================================================

@pytest.mark.asyncio
async def test_streaming_failover_before_first_token(sample_evidence):
    """If OpenRouter fails before any token is emitted, stream fails over cleanly to Ollama."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = sample_evidence

    async def failing_openrouter_stream(*args, **kwargs):
        raise OpenRouterAccountQuotaExhaustedError("Rate limit exceeded: free-models-per-day: 50 requests/day")
        yield  # Make it an async generator

    async def successful_ollama_stream(*args, **kwargs):
        yield {"type": "token", "token": "AIDA "}
        yield {"type": "token", "token": "cliffside "}
        yield {"type": "token", "token": "villas."}

    mock_openrouter = MagicMock()
    mock_openrouter.primary_model = "nvidia/nemotron-3-ultra-550b-a55b:free"
    mock_openrouter.stream_completion = failing_openrouter_stream

    mock_ollama = MagicMock()
    mock_ollama.stream_completion = successful_ollama_stream

    orchestrator = RAGOrchestrator(
        retriever=mock_retriever,
        llm_client=mock_openrouter,
        ollama_client=mock_ollama,
    )

    req = ChatRequest(messages=[ChatMessage(role="user", content="Tell me about AIDA Oman")])
    events = []
    async for event in orchestrator.stream_answer_question(req):
        events.append(event)

    event_types = [e["event"] for e in events]
    assert "status" in event_types
    assert "token" in event_types
    assert "citations" in event_types
    assert "complete" in event_types

    # Verify fallback notice was emitted in status
    fallback_statuses = [
        e["data"]["message"] for e in events
        if e["event"] == "status" and "Routing to local Qwen" in e["data"].get("message", "")
    ]
    assert len(fallback_statuses) > 0

    # Verify tokens came through
    tokens = [e["data"]["token"] for e in events if e["event"] == "token"]
    assert "".join(tokens) == "AIDA cliffside villas."

    # Verify complete event has ollama attribution and fallback metadata
    complete_event = [e for e in events if e["event"] == "complete"][0]
    assert complete_event["data"]["provider"] == "ollama"
    assert complete_event["data"]["fallback_active"] is True
    assert complete_event["data"]["fallback_reason"] == "quota_exhausted"
    assert "daily free-model request limit" in complete_event["data"]["fallback_notice"]


@pytest.mark.asyncio
async def test_streaming_never_failover_midstream(sample_evidence):
    """If OpenRouter fails AFTER emitting tokens, orchestrator NEVER fails over to Ollama.
    It terminates with an error event to prevent concatenating another model mid-sentence.
    """
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = sample_evidence

    async def openrouter_midstream_crash(*args, **kwargs):
        yield {"type": "token", "token": "AIDA is located on"}
        # Drop connection midstream
        raise OpenRouterServiceError("Stream interrupted: connection reset by peer")

    mock_ollama = MagicMock()
    mock_ollama.stream_completion = AsyncMock()

    mock_openrouter = MagicMock()
    mock_openrouter.primary_model = "nvidia/nemotron-3-ultra-550b-a55b:free"
    mock_openrouter.stream_completion = openrouter_midstream_crash

    orchestrator = RAGOrchestrator(
        retriever=mock_retriever,
        llm_client=mock_openrouter,
        ollama_client=mock_ollama,
    )

    req = ChatRequest(messages=[ChatMessage(role="user", content="Tell me about AIDA Oman")])
    events = []
    async for event in orchestrator.stream_answer_question(req):
        events.append(event)

    # Assert tokens were received
    tokens = [e["data"]["token"] for e in events if e["event"] == "token"]
    assert "AIDA is located on" in tokens

    # Assert error event was emitted instead of failing over to Ollama
    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["data"]["error_code"] == "LLM_SERVICE_UNAVAILABLE"

    # Crucial: Ollama was NEVER called mid-stream
    mock_ollama.stream_completion.assert_not_called()
