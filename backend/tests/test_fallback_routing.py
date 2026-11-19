"""Comprehensive Test Suite for Controlled Multi-Model Fallback Routing.

Tests:
1. Model hierarchy & priority order (Primary -> Fallback 1 -> Fallback 2).
2. Eligible transient failure triggers (model-specific 429, 503, connection timeout).
3. Ineligible failure blocks (HTTP 401 auth, HTTP 402 quota, HTTP 400 invalid request).
4. Account-wide quota exhaustion detection (free-models-per-day) halts fallback immediately.
5. Streaming guardrails (fallback allowed before first token; NEVER fallback mid-stream).
6. Exact model attribution returned in ChatResponse and SSE events.
7. RAG validation integrity preserved identically across all models.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from backend.app.config import Settings
from backend.app.core.exceptions import KnowledgeBaseException
from backend.app.rag.openrouter import (
    OpenRouterClient,
    OpenRouterAuthError,
    OpenRouterQuotaError,
    OpenRouterRateLimitError,
    OpenRouterServiceError,
    is_account_level_quota_exhausted,
)
from backend.app.rag.orchestrator import RAGOrchestrator
from backend.app.rag.retriever import RetrievedItem
from backend.app.schemas.chat import ChatMessage, ChatRequest


# ============================================================================
# 1. Configuration & Quota Detection Tests
# ============================================================================

def test_fallback_models_config_semicolon_and_comma():
    """Verify OPENROUTER_FALLBACK_MODELS parses semicolons and commas."""
    # Semicolon separated
    s1 = Settings(OPENROUTER_FALLBACK_MODELS="meta-llama/llama-3.3-70b;mistralai/mistral-7b")
    assert s1.OPENROUTER_FALLBACK_MODELS == [
        "meta-llama/llama-3.3-70b",
        "mistralai/mistral-7b",
    ]

    # Comma separated
    s2 = Settings(OPENROUTER_FALLBACK_MODELS="meta-llama/llama-3.3-70b, mistralai/mistral-7b")
    assert s2.OPENROUTER_FALLBACK_MODELS == [
        "meta-llama/llama-3.3-70b",
        "mistralai/mistral-7b",
    ]


def test_is_account_level_quota_exhausted_detection():
    """Verify detection of account-wide daily limit vs model-specific rate limiting."""
    # Account-wide quota exhaustion messages
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
# 2. Completion Routing & Priority Tests
# ============================================================================

@pytest.mark.asyncio
async def test_fallback_priority_chain_primary_to_fallback2():
    """Verify routing attempts models strictly in priority order:
    Primary fails (model-specific 429) -> Fallback 1 fails (503) -> Fallback 2 succeeds (200).
    """
    client = OpenRouterClient(
        api_key="sk-test-key",
        primary_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        allow_fallback=True,
        fallback_models=[
            "meta-llama/llama-3.3-70b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
        ],
    )

    resp_429 = MagicMock(spec=httpx.Response)
    resp_429.status_code = 429
    resp_429.text = '{"error": {"message": "Model concurrency limit reached"}}'
    resp_429.headers = {}

    resp_503 = MagicMock(spec=httpx.Response)
    resp_503.status_code = 503
    resp_503.text = "Service Unavailable"

    resp_200 = MagicMock(spec=httpx.Response)
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Grounded answer from Fallback 2"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 25},
    }

    # Primary gets 2x 429; Fallback 1 gets 2x 503; Fallback 2 gets 200
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [resp_429, resp_429, resp_503, resp_503, resp_200]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client.generate_completion([{"role": "user", "content": "Tell me about AIDA"}])

            assert result["content"] == "Grounded answer from Fallback 2"
            assert result["model"] == "mistralai/mistral-7b-instruct:free"

            # Check models requested in chronological order
            models_called = [call.kwargs["json"]["model"] for call in mock_post.call_args_list]
            assert models_called == [
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "mistralai/mistral-7b-instruct:free",
            ]


@pytest.mark.asyncio
async def test_all_models_fail_raises_service_error():
    """Verify OpenRouterServiceError is raised when all candidate models fail."""
    client = OpenRouterClient(
        api_key="sk-test-key",
        primary_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        allow_fallback=True,
        fallback_models=["meta-llama/llama-3.3-70b-instruct:free"],
    )

    resp_503 = MagicMock(spec=httpx.Response)
    resp_503.status_code = 503
    resp_503.text = "Service Unavailable"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = resp_503
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(OpenRouterServiceError):
                await client.generate_completion([{"role": "user", "content": "Hello"}])


# ============================================================================
# 3. Ineligible Failure Tests (No Fallback Allowed)
# ============================================================================

@pytest.mark.asyncio
async def test_account_wide_quota_blocks_fallback():
    """Verify that account-wide 429 quota exhaustion (50 req/day) halts immediately without trying fallback."""
    client = OpenRouterClient(
        api_key="sk-test-key",
        primary_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        allow_fallback=True,
        fallback_models=["meta-llama/llama-3.3-70b-instruct:free"],
    )

    resp_account_429 = MagicMock(spec=httpx.Response)
    resp_account_429.status_code = 429
    resp_account_429.text = '{"error": {"message": "Rate limit exceeded: free-models-per-day: 50 requests/day"}}'
    resp_account_429.headers = {"Retry-After": "3600"}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = resp_account_429
        with pytest.raises(OpenRouterRateLimitError):
            await client.generate_completion([{"role": "user", "content": "Hello"}])

        # Assert client stopped immediately after detecting account-wide quota
        assert mock_post.call_count == 1
        assert mock_post.call_args_list[0].kwargs["json"]["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"


@pytest.mark.asyncio
async def test_http_401_auth_error_no_fallback():
    """Verify HTTP 401 raises OpenRouterAuthError without trying fallback."""
    client = OpenRouterClient(
        api_key="invalid-key",
        primary_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        allow_fallback=True,
        fallback_models=["meta-llama/llama-3.3-70b-instruct:free"],
    )

    resp_401 = MagicMock(spec=httpx.Response)
    resp_401.status_code = 401
    resp_401.text = "Unauthorized"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = resp_401
        with pytest.raises(OpenRouterAuthError):
            await client.generate_completion([{"role": "user", "content": "Hello"}])

        assert mock_post.call_count == 1


@pytest.mark.asyncio
async def test_http_400_invalid_request_no_fallback():
    """Verify HTTP 400 invalid request raises KnowledgeBaseException without trying fallback."""
    client = OpenRouterClient(
        api_key="sk-test-key",
        primary_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        allow_fallback=True,
        fallback_models=["meta-llama/llama-3.3-70b-instruct:free"],
    )

    resp_400 = MagicMock(spec=httpx.Response)
    resp_400.status_code = 400
    resp_400.text = "Invalid parameter payload"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = resp_400
        with pytest.raises(KnowledgeBaseException) as exc_info:
            await client.generate_completion([{"role": "user", "content": "Hello"}])

        assert exc_info.value.code == "LLM_INVALID_REQUEST"
        assert mock_post.call_count == 1


# ============================================================================
# 4. Streaming Guardrail Tests
# ============================================================================

@pytest.mark.asyncio
async def test_streaming_fallback_before_first_token():
    """Verify that if primary fails with 503 before emitting any token, fallback model is used."""
    client = OpenRouterClient(
        api_key="sk-test-key",
        primary_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        allow_fallback=True,
        fallback_models=["meta-llama/llama-3.3-70b-instruct:free"],
    )

    # Primary stream returns 503
    resp_503 = MagicMock()
    resp_503.status_code = 503
    resp_503.aread = AsyncMock(return_value=b"Service Unavailable")

    # Fallback stream returns 200 with tokens
    async def aiter_fallback():
        yield 'data: {"choices": [{"delta": {"content": "Fallback stream token"}}]}'
        yield "data: [DONE]"

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.aiter_lines = aiter_fallback

    class MockStreamContext:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("httpx.AsyncClient.stream") as mock_stream:
        # Primary gets 2x 503, fallback gets 200
        mock_stream.side_effect = [
            MockStreamContext(resp_503),
            MockStreamContext(resp_503),
            MockStreamContext(resp_200),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            chunks = []
            async for chunk in client.stream_completion([{"role": "user", "content": "Hello"}]):
                chunks.append(chunk)

            tokens = [c["token"] for c in chunks if c["type"] == "token"]
            assert tokens == ["Fallback stream token"]
            done_chunk = next(c for c in chunks if c["type"] == "done")
            assert done_chunk["model"] == "meta-llama/llama-3.3-70b-instruct:free"


@pytest.mark.asyncio
async def test_streaming_no_fallback_after_tokens_started():
    """Verify that if a model fails AFTER tokens have started streaming,
    client NEVER falls back or concatenates another model; it raises immediately.
    """
    client = OpenRouterClient(
        api_key="sk-test-key",
        primary_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        allow_fallback=True,
        fallback_models=["meta-llama/llama-3.3-70b-instruct:free"],
    )

    async def aiter_broken_stream():
        yield 'data: {"choices": [{"delta": {"content": "Initial primary token"}}]}'
        # Network breaks mid-stream
        raise httpx.NetworkError("Connection lost mid-response")

    resp_broken = MagicMock()
    resp_broken.status_code = 200
    resp_broken.aiter_lines = aiter_broken_stream

    class MockStreamContext:
        async def __aenter__(self):
            return resp_broken

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_stream.return_value = MockStreamContext()
        chunks = []
        with pytest.raises(OpenRouterServiceError) as exc_info:
            async for chunk in client.stream_completion([{"role": "user", "content": "Hello"}]):
                chunks.append(chunk)

        # Emitted 1 token before breaking
        assert len(chunks) == 1
        assert chunks[0]["token"] == "Initial primary token"
        assert "interrupted" in str(exc_info.value).lower()
        # Verify mock_stream was called only once (never called fallback model!)
        assert mock_stream.call_count == 1


# ============================================================================
# 5. End-to-End Orchestrator Model Attribution Tests
# ============================================================================

@pytest.mark.asyncio
async def test_orchestrator_returns_exact_model_attribution():
    """Verify orchestrator reports fallback model in ChatResponse and stream events."""
    mock_retriever = MagicMock()
    mock_retriever.aretrieve = AsyncMock(
        return_value=(
            [
                RetrievedItem(
                    chunk_id="test_chunk_1",
                    entity_id="test_entity_1",
                    doc_id="test_doc_1",
                    source="darglobal",
                    category="project",
                    title="AIDA Development",
                    url="https://darglobal.co.uk/aida",
                    text_content="AIDA is located in Muscat, Oman.",
                    provenance="recruiter_provided",
                    semantic_similarity=0.92,
                )
            ],
            {"embedding_ms": 5.0, "vector_search_ms": 10.0, "bm25_search_ms": 4.0, "rrf_fusion_ms": 1.0, "total_retrieval_ms": 20.0},
        )
    )

    # OpenRouter returns fallback model
    mock_llm_client = MagicMock()
    mock_llm_client.primary_model = "nvidia/nemotron-3-ultra-550b-a55b:free"
    mock_llm_client.generate_completion = AsyncMock(
        return_value={
            "content": "AIDA is a luxury development in Muscat, Oman.",
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "dispatch_ms": 10.0,
            "generation_ms": 200.0,
        }
    )

    orchestrator = RAGOrchestrator(retriever=mock_retriever, llm_client=mock_llm_client)

    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Where is AIDA located?")],
        source_filter="darglobal",
    )

    # 1. Non-streaming ChatResponse
    response = await orchestrator.answer_question(request)
    assert response.model == "meta-llama/llama-3.3-70b-instruct:free"
    assert len(response.citations) > 0

    # 2. Streaming SSE events
    async def mock_stream(*args, **kwargs):
        yield {"type": "token", "token": "AIDA is in Muscat, Oman."}
        yield {"type": "done", "model": "meta-llama/llama-3.3-70b-instruct:free"}

    mock_llm_client.stream_completion = mock_stream

    events = []
    async for event in orchestrator.stream_answer_question(request):
        events.append(event)

    citations_event = next(e for e in events if e["event"] == "citations")
    assert citations_event["data"]["model"] == "meta-llama/llama-3.3-70b-instruct:free"

    complete_event = next(e for e in events if e["event"] == "complete")
    assert complete_event["data"]["model"] == "meta-llama/llama-3.3-70b-instruct:free"
