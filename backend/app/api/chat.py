import asyncio
import json
from typing import Optional
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse

from backend.app.api.retrieve import get_retriever
from backend.app.core.logging import logger
from backend.app.core.security import limiter
from backend.app.rag.orchestrator import RAGOrchestrator
from backend.app.rag.retriever import HybridRetriever
from backend.app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["Chat"])

_orchestrator_instance: Optional[RAGOrchestrator] = None


def get_orchestrator(
    retriever: HybridRetriever = Depends(get_retriever),
) -> RAGOrchestrator:
    """Dependency provider for RAGOrchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = RAGOrchestrator(retriever=retriever)
    return _orchestrator_instance


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit query to knowledge assistant",
)
@limiter.limit("60/minute")
async def chat_endpoint(
    request: Request,
    payload: ChatRequest,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    """Submit a question to the DarGlobal & Wasalt Real Estate Knowledge Assistant.

    Validates message structure (strictly user/assistant roles), executes hybrid
    retrieval, checks data mode and provenance, queries OpenRouter, and returns
    a grounded answer with verified citations.
    """
    logger.info(
        f"Processing chat request: session_id={payload.session_id}, "
        f"source_filter={payload.source_filter}, turns={len(payload.messages)}"
    )

    response = await orchestrator.answer_question(payload)
    return response


@router.post(
    "/chat/stream",
    summary="Submit query to knowledge assistant with real-time SSE token streaming",
)
@limiter.limit("60/minute")
async def chat_stream_endpoint(
    request: Request,
    payload: ChatRequest,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator),
):
    """Submit a question and stream back real-time Server-Sent Events (SSE).

    Streams status, incremental tokens, validated citations, and completion metadata.
    """
    logger.info(
        f"Processing streaming chat request: session_id={payload.session_id}, "
        f"source_filter={payload.source_filter}, turns={len(payload.messages)}"
    )

    async def event_generator():
        try:
            async for event_dict in orchestrator.stream_answer_question(payload):
                # Check for client disconnect
                if await request.is_disconnected():
                    logger.info(f"Client disconnected during streaming [{payload.session_id}].")
                    break
                event_name = event_dict.get("event", "message")
                event_data = json.dumps(event_dict.get("data", {}))
                yield f"event: {event_name}\ndata: {event_data}\n\n"
        except asyncio.CancelledError:
            logger.info(f"Streaming task cancelled by client [{payload.session_id}].")
            raise
        except Exception as exc:
            logger.error(f"Error during streaming generation: {exc}")
            err_data = json.dumps({"code": "INTERNAL_ERROR", "message": "Streaming error occurred."})
            yield f"event: error\ndata: {err_data}\n\n"

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream; charset=utf-8",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )
