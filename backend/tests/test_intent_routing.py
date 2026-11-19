"""Automated Test Suite for Lightweight Intent Routing.

Validates:
1. Pure greetings (e.g., 'hi', 'hello', 'hey', 'good morning') return fast canned responses
   without triggering ChromaDB retrieval, BM25 search, LLM calls, property citations, or demo warnings.
2. Acknowledgements (e.g., 'thanks', 'thank you', 'appreciate it') return fast polite responses
   without retrieval, citations, or demo warnings.
3. Safeguard: Short legitimate property questions ('AIDA?', 'Oman?', 'prices?', 'villas?', 'Wasalt?')
   are strictly classified as PROPERTY_QUESTION and routed through full RAG.
4. Out-of-scope questions (cooking, programming, trivia, medical) receive polite refusal without retrieval.
5. Streaming fast-path emits token, citations, and complete events without status:searching.
"""

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from backend.app.rag.intent import UserIntent, classify_intent
from backend.app.rag.orchestrator import RAGOrchestrator
from backend.app.rag.retriever import RetrievedItem
from backend.app.schemas.chat import ChatMessage, ChatRequest


# ============================================================================
# 1. Unit Tests for Deterministic Intent Classifier
# ============================================================================

@pytest.mark.parametrize(
    "greeting",
    [
        "hi",
        "hello",
        "hey",
        "hey there",
        "good morning",
        "Good Afternoon!",
        "good evening",
        "salam",
        "assalamu alaikum",
        "howdy",
        "hi assistant",
        "hello there",
        "what's up",
        "how are you",
    ],
)
def test_classify_intent_greetings(greeting: str):
    """Test pure greeting variations route to GREETING with a canned response."""
    intent, canned_response = classify_intent(greeting)
    assert intent == UserIntent.GREETING
    assert canned_response is not None
    assert "DarGlobal & Wasalt" in canned_response
    assert "How can I assist you" in canned_response


@pytest.mark.parametrize(
    "ack",
    [
        "thanks",
        "thank you",
        "thanks a lot",
        "thank you very much",
        "thanks so much",
        "thank you so much",
        "appreciate it",
        "much appreciated",
        "great thanks",
        "perfect thank you",
        "got it thanks",
    ],
)
def test_classify_intent_acknowledgements(ack: str):
    """Test acknowledgement phrases route to ACKNOWLEDGEMENT with a canned response."""
    intent, canned_response = classify_intent(ack)
    assert intent == UserIntent.ACKNOWLEDGEMENT
    assert canned_response is not None
    assert "welcome" in canned_response.lower()


@pytest.mark.parametrize(
    "short_query",
    [
        "AIDA?",
        "Oman?",
        "prices?",
        "villas?",
        "Wasalt?",
        "DarGlobal",
        "urban oasis",
        "pagani",
        "rent?",
        "payment plan?",
        "apartments in Riyadh",
        "penthouse",
        "freehold",
    ],
)
def test_classify_intent_short_property_queries_never_misclassified(short_query: str):
    """Safeguard: Short property or company queries must NEVER be classified as greetings."""
    intent, canned_response = classify_intent(short_query)
    assert intent == UserIntent.PROPERTY_QUESTION
    assert canned_response is None


@pytest.mark.parametrize(
    "out_of_scope",
    [
        "How do I bake a chocolate cake?",
        "Write a python script to sort a list",
        "Can you help me debug my c++ code?",
        "Write a poem about sunflowers",
        "What are the symptoms of flu?",
        "Who won the world cup in 2022?",
        "What is the capital of France?",
    ],
)
def test_classify_intent_out_of_scope(out_of_scope: str):
    """Test out-of-scope non-real-estate inquiries route to OUT_OF_SCOPE with a polite refusal."""
    intent, canned_response = classify_intent(out_of_scope)
    assert intent == UserIntent.OUT_OF_SCOPE
    assert canned_response is not None
    assert "information regarding this question is not currently available" in canned_response
    assert "DarGlobal" in canned_response


# ============================================================================
# 2. Orchestrator Integration Tests: Non-Streaming Fast-Path
# ============================================================================

@pytest.fixture
def mock_retriever():
    """Mock retriever with both sync and async methods."""
    retriever = MagicMock()
    retriever.retrieve = MagicMock(return_value=[])
    retriever.aretrieve = AsyncMock(return_value=([], {
        "embedding_ms": 0.0,
        "vector_search_ms": 0.0,
        "bm25_search_ms": 0.0,
        "rrf_fusion_ms": 0.0,
        "total_retrieval_ms": 0.0,
    }))
    return retriever


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = MagicMock()
    client.primary_model = "nvidia/nemotron-3-ultra-550b-a55b:free"
    client.generate_completion = AsyncMock(return_value={
        "content": "Sample LLM answer.",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "dispatch_ms": 10.0,
        "generation_ms": 20.0,
    })
    return client


@pytest.mark.asyncio
async def test_orchestrator_greeting_skips_retrieval_and_llm(mock_retriever, mock_llm_client):
    """Greeting query returns fast response without calling retriever, LLM, or showing warnings."""
    orchestrator = RAGOrchestrator(retriever=mock_retriever, llm_client=mock_llm_client)
    request = ChatRequest(messages=[ChatMessage(role="user", content="Hello!")])

    response = await orchestrator.answer_question(request)

    # Verify retriever and LLM were NEVER called
    mock_retriever.retrieve.assert_not_called()
    mock_retriever.aretrieve.assert_not_called()
    mock_llm_client.generate_completion.assert_not_called()

    # Verify response structure
    assert "DarGlobal & Wasalt Real Estate Knowledge Assistant" in response.answer
    assert response.citations == []
    assert response.property_cards == []
    assert response.provenance_warning is None
    assert response.data_mode == "conversational"
    assert response.model is None


@pytest.mark.asyncio
async def test_orchestrator_acknowledgement_skips_retrieval_and_llm(mock_retriever, mock_llm_client):
    """Acknowledgement query returns fast response without calling retriever or LLM."""
    orchestrator = RAGOrchestrator(retriever=mock_retriever, llm_client=mock_llm_client)
    request = ChatRequest(messages=[ChatMessage(role="user", content="Thank you so much!")])

    response = await orchestrator.answer_question(request)

    mock_retriever.retrieve.assert_not_called()
    mock_retriever.aretrieve.assert_not_called()
    mock_llm_client.generate_completion.assert_not_called()

    assert "welcome" in response.answer.lower()
    assert response.citations == []
    assert response.property_cards == []
    assert response.provenance_warning is None


@pytest.mark.asyncio
async def test_orchestrator_out_of_scope_skips_retrieval_and_llm(mock_retriever, mock_llm_client):
    """Out-of-scope query returns polite refusal without retrieval or LLM calls."""
    orchestrator = RAGOrchestrator(retriever=mock_retriever, llm_client=mock_llm_client)
    request = ChatRequest(messages=[ChatMessage(role="user", content="How do I bake a cake?")])

    response = await orchestrator.answer_question(request)

    mock_retriever.retrieve.assert_not_called()
    mock_retriever.aretrieve.assert_not_called()
    mock_llm_client.generate_completion.assert_not_called()

    assert "information regarding this question is not currently available" in response.answer
    assert response.citations == []
    assert response.provenance_warning is None


@pytest.mark.asyncio
async def test_orchestrator_short_property_query_routes_to_rag(mock_retriever, mock_llm_client):
    """Short property query ('AIDA?') must NOT take fast-path, and MUST invoke retriever & LLM."""
    mock_items = [
        RetrievedItem(
            chunk_id="chunk_aida_1",
            entity_id="entity_aida",
            title="AIDA Oman",
            source="darglobal",
            category="project",
            url="https://darglobal.co.uk/aida",
            provenance="verified_source",
            text_content="AIDA is a luxury master development in Oman.",
            rrf_score=0.04,
            semantic_similarity=0.89,
        )
    ]
    mock_retriever.aretrieve = AsyncMock(return_value=(mock_items, {"embedding_ms": 1.0, "total_retrieval_ms": 2.0}))

    orchestrator = RAGOrchestrator(retriever=mock_retriever, llm_client=mock_llm_client)
    request = ChatRequest(messages=[ChatMessage(role="user", content="AIDA?")])

    response = await orchestrator.answer_question(request)

    # Verify retrieval and LLM were actually called
    assert mock_retriever.aretrieve.called or mock_retriever.retrieve.called
    assert mock_llm_client.generate_completion.called
    assert response.grounded is True


# ============================================================================
# 3. Orchestrator Integration Tests: Streaming Fast-Path
# ============================================================================

@pytest.mark.asyncio
async def test_orchestrator_streaming_greeting_fast_path(mock_retriever, mock_llm_client):
    """Streaming greeting emits token, citations, and complete; never emits 'searching' status."""
    orchestrator = RAGOrchestrator(retriever=mock_retriever, llm_client=mock_llm_client)
    request = ChatRequest(messages=[ChatMessage(role="user", content="Good morning")])

    events = []
    async for event in orchestrator.stream_answer_question(request):
        events.append(event)

    event_names = [e["event"] for e in events]
    # Verify no 'status' event was emitted
    assert "status" not in event_names
    assert "token" in event_names
    assert "citations" in event_names
    assert "complete" in event_names

    # Check token content
    token_event = next(e for e in events if e["event"] == "token")
    assert "DarGlobal & Wasalt Real Estate Knowledge Assistant" in token_event["data"]["token"]

    # Check citations event
    cit_event = next(e for e in events if e["event"] == "citations")
    assert cit_event["data"]["citations"] == []
    assert cit_event["data"]["provenance_warning"] is None
    assert cit_event["data"]["model"] is None
    assert cit_event["data"]["data_mode"] == "conversational"

    # Check complete event
    complete_event = next(e for e in events if e["event"] == "complete")
    assert complete_event["data"]["provenance_warning"] is None
    assert complete_event["data"]["grounded"] is True
    assert complete_event["data"]["model"] is None
    assert complete_event["data"]["data_mode"] == "conversational"

    # Ensure zero retriever/LLM calls
    mock_retriever.retrieve.assert_not_called()
    mock_retriever.aretrieve.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_streaming_out_of_scope_fast_path(mock_retriever, mock_llm_client):
    """Streaming out-of-scope query emits refusal without search status or retrieval."""
    orchestrator = RAGOrchestrator(retriever=mock_retriever, llm_client=mock_llm_client)
    request = ChatRequest(messages=[ChatMessage(role="user", content="Write a python quicksort script")])

    events = []
    async for event in orchestrator.stream_answer_question(request):
        events.append(event)

    event_names = [e["event"] for e in events]
    assert "status" not in event_names
    assert "token" in event_names

    token_event = next(e for e in events if e["event"] == "token")
    assert "information regarding this question is not currently available" in token_event["data"]["token"]

    mock_retriever.retrieve.assert_not_called()
    mock_retriever.aretrieve.assert_not_called()
