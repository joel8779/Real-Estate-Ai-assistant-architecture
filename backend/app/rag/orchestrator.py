"""Conversational RAG Orchestration Service.

Orchestrates input validation, conversational query contextualization, hybrid retrieval,
provenance assessment (verified vs demonstration mode), context fence construction,
OpenRouter LLM inference, numerical response validation, and citation assembly.
"""

import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from backend.app.config import get_settings
from backend.app.core.logging import logger
from backend.app.core.security import sanitize_input_text
from backend.app.rag.openrouter import (
    OpenRouterClient,
    LLMNotConfiguredException,
    OpenRouterRateLimitError,
    KnowledgeBaseException,
)
from backend.app.rag.intent import classify_intent, UserIntent
from backend.app.rag.prompt import (
    build_llm_messages,
    contextualize_followup_query,
)
from backend.app.rag.retriever import HybridRetriever
from backend.app.rag.validator import ResponseValidator
from backend.app.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
)


class RAGOrchestrator:
    """Dedicated orchestrator for the conversational real estate RAG pipeline."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: Optional[OpenRouterClient] = None,
    ):
        self.retriever = retriever
        self.llm_client = llm_client or OpenRouterClient()

    async def answer_question(self, payload: ChatRequest) -> ChatResponse:
        """Execute end-to-end conversational RAG pipeline."""
        start_time = time.perf_counter()
        settings = get_settings()
        request_id = payload.session_id or f"req_{uuid.uuid4().hex[:12]}"

        # 1. Input sanitization & validation
        t_pre_start = time.perf_counter()
        last_turn = payload.messages[-1]
        raw_query = last_turn.content.strip()
        sanitized_query = sanitize_input_text(raw_query)

        # 2. Lightweight Intent Routing (Zero-retrieval fast-path for greetings & acknowledgements)
        intent, canned_response = classify_intent(sanitized_query)
        is_conversational = intent in (UserIntent.GREETING, UserIntent.ACKNOWLEDGEMENT, UserIntent.OUT_OF_SCOPE)

        logger.info(
            f"RAG Request [{request_id}]: query='{sanitized_query[:40]}...', "
            f"intent='{intent.value}', "
            f"retrieval_executed={not is_conversational}, "
            f"llm_invoked={not is_conversational}"
        )

        if is_conversational:
            total_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                f"RAG Intent Fast-Path [{request_id}]: intent='{intent.value}' — skipping retrieval and LLM inference."
            )
            return ChatResponse(
                answer=canned_response,
                citations=[],
                property_cards=[],
                confidence_score=1.0,
                grounded=True,
                data_mode="conversational",
                provenance_warning=None,
                model=None,
                request_id=request_id,
                latency_ms=total_ms,
                stage_latencies={
                    "1_request_received": True,
                    "2_preprocessing_ms": round((time.perf_counter() - t_pre_start) * 1000.0, 2),
                    "3_embedding_ms": 0.0,
                    "4_vector_search_ms": 0.0,
                    "5_bm25_search_ms": 0.0,
                    "6_rrf_fusion_ms": 0.0,
                    "7_context_construction_ms": 0.0,
                    "8_llm_dispatch_ms": 0.0,
                    "9_llm_generation_ms": 0.0,
                    "10_total_end_to_end_ms": total_ms,
                },
            )

        # 3. Contextualize query for follow-up questions
        search_query = contextualize_followup_query(sanitized_query, payload.messages)
        if search_query != sanitized_query:
            logger.info(f"RAG Contextualized query [{request_id}]: '{search_query[:50]}...'")
        stage_2_preprocessing_ms = round((time.perf_counter() - t_pre_start) * 1000.0, 2)

        # 4-6. Hybrid Retrieval (concurrent semantic vector search and BM25 with timings)
        import asyncio
        if hasattr(self.retriever, "aretrieve") and asyncio.iscoroutinefunction(self.retriever.aretrieve):
            retrieved_items, retrieval_timings = await self.retriever.aretrieve(
                query=search_query,
                top_k=settings.RETRIEVAL_TOP_K,
                source_filter=payload.source_filter if payload.source_filter != "all" else None,
                category_filter=payload.category_filter,
                method="hybrid",
            )
        else:
            t_ret = time.perf_counter()
            retrieved_items = self.retriever.retrieve(
                query=search_query,
                top_k=settings.RETRIEVAL_TOP_K,
                source_filter=payload.source_filter if payload.source_filter != "all" else None,
                category_filter=payload.category_filter,
                method="hybrid",
            )
            retrieval_timings = {
                "embedding_ms": 0.0,
                "vector_search_ms": 0.0,
                "bm25_search_ms": 0.0,
                "rrf_fusion_ms": 0.0,
                "total_retrieval_ms": round((time.perf_counter() - t_ret) * 1000, 2),
            }

        stage_3_embedding_ms = retrieval_timings.get("embedding_ms", 0.0)
        stage_4_vector_search_ms = retrieval_timings.get("vector_search_ms", 0.0)
        stage_5_bm25_search_ms = retrieval_timings.get("bm25_search_ms", 0.0)
        stage_6_rrf_fusion_ms = retrieval_timings.get("rrf_fusion_ms", 0.0)
        total_retrieval_ms = retrieval_timings.get("total_retrieval_ms", 0.0)

        logger.info(
            f"RAG Retrieved [{request_id}]: {len(retrieved_items)} chunks for query "
            f"(embedding={stage_3_embedding_ms}ms, vector={stage_4_vector_search_ms}ms, "
            f"bm25={stage_5_bm25_search_ms}ms, rrf={stage_6_rrf_fusion_ms}ms)."
        )

        # 4. Provenance & Data Mode Assessment
        allow_demo_data = settings.ALLOW_UNVERIFIED_DEMO_DATA
        data_mode: str = "verified"
        provenance_warning: Optional[str] = None

        if not allow_demo_data:
            # Verified mode: filter out any demonstration_unverified records
            verified_items = [
                item for item in retrieved_items
                if item.provenance in ("verified_source", "recruiter_provided")
            ]
            if not verified_items:
                logger.info(
                    f"RAG Verified Mode [{request_id}]: No verified public records found (demo records rejected)."
                )
                return ChatResponse(
                    answer=(
                        "Verified public records are not currently available for this query because automated "
                        "access to live sources is restricted by website protection policies. "
                        "(Demonstration mode is currently disabled in system settings. "
                        "Set ALLOW_UNVERIFIED_DEMO_DATA=true to explore sample demonstration records)."
                    ),
                    citations=[],
                    property_cards=[],
                    confidence_score=0.0,
                    grounded=False,
                    data_mode="verified",
                    provenance_warning=None,
                    request_id=request_id,
                    latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
                )
            retrieved_items = verified_items
        else:
            # Demonstration mode enabled
            has_unverified = any(item.provenance == "demonstration_unverified" for item in retrieved_items)
            if has_unverified:
                data_mode = "demonstration"
                provenance_warning = (
                    "This prototype uses sample property information prepared for demonstration purposes. "
                    "Property details, pricing and availability have not been independently verified. "
                    "Please consult official sources for current information."
                )

        # 5. Evidence Sufficiency & Similarity Cutoff
        if not retrieved_items:
            total_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ChatResponse(
                answer=(
                    "Based on the available records, information regarding this question is not currently available "
                    "in our knowledge base. Please consult official DarGlobal or Wasalt representatives directly."
                ),
                citations=[],
                property_cards=[],
                confidence_score=0.0,
                grounded=False,
                data_mode=data_mode,
                provenance_warning=provenance_warning,
                request_id=request_id,
                latency_ms=total_ms,
            )

        top_similarity = (
            retrieved_items[0].semantic_similarity
            if retrieved_items[0].semantic_similarity is not None
            else 0.5
        )

        # Check if top match is too low to be considered reliable
        if top_similarity < settings.SIMILARITY_THRESHOLD and retrieved_items[0].bm25_score is None:
            logger.info(
                f"RAG Low Confidence [{request_id}]: Top similarity {top_similarity:.3f} "
                f"below threshold {settings.SIMILARITY_THRESHOLD}."
            )
            total_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ChatResponse(
                answer=(
                    f"Based on the available records, I could not find sufficiently reliable information "
                    f"regarding '{sanitized_query}'. Please verify directly with official representatives."
                ),
                citations=ResponseValidator.build_citations(retrieved_items[:2]),
                property_cards=[],
                confidence_score=round(top_similarity, 3),
                grounded=False,
                data_mode=data_mode,
                provenance_warning=provenance_warning,
                request_id=request_id,
                latency_ms=total_ms,
            )

        # 7. Context Construction
        t_ctx_start = time.perf_counter()
        llm_messages = build_llm_messages(
            query=sanitized_query,
            evidence_items=retrieved_items,
            conversation_history=payload.messages,
            data_mode=data_mode,
        )
        stage_7_context_construction_ms = round((time.perf_counter() - t_ctx_start) * 1000.0, 2)

        # 8-9. OpenRouter API Request & LLM Generation
        stage_8_llm_dispatch_ms = 0.0
        stage_9_llm_generation_ms = 0.0

        try:
            llm_result = await self.llm_client.generate_completion(llm_messages, request_id=request_id)
            answer_text = llm_result["content"].strip()
            model_used = llm_result.get("model", self.llm_client.primary_model)
            stage_8_llm_dispatch_ms = llm_result.get("dispatch_ms", 0.0)
            stage_9_llm_generation_ms = llm_result.get("generation_ms", 0.0)
            error_code = None
        except LLMNotConfiguredException as e:
            logger.warning(f"RAG LLM Not Configured [{request_id}]: {e.message}")
            total_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ChatResponse(
                answer=(
                    "OpenRouter LLM service is not configured. "
                    "Please set a valid OPENROUTER_API_KEY in the environment (.env) to enable AI synthesis."
                ),
                citations=[],
                property_cards=[],
                confidence_score=0.0,
                grounded=False,
                data_mode=data_mode,
                provenance_warning=None,
                error_code="LLM_NOT_CONFIGURED",
                request_id=request_id,
                latency_ms=total_ms,
                stage_latencies={
                    "1_request_received": True,
                    "2_preprocessing_ms": stage_2_preprocessing_ms,
                    "3_embedding_ms": stage_3_embedding_ms,
                    "4_vector_search_ms": stage_4_vector_search_ms,
                    "5_bm25_search_ms": stage_5_bm25_search_ms,
                    "6_rrf_fusion_ms": stage_6_rrf_fusion_ms,
                    "7_context_construction_ms": stage_7_context_construction_ms,
                    "8_llm_dispatch_ms": 0.0,
                    "9_llm_generation_ms": 0.0,
                    "10_total_end_to_end_ms": total_ms,
                },
            )
        except OpenRouterRateLimitError as e:
            logger.warning(f"RAG LLM Rate Limited [{request_id}]: {e.message} (retry_after={e.retry_after})")
            total_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ChatResponse(
                answer=e.message,
                citations=[],
                property_cards=[],
                confidence_score=0.0,
                grounded=False,
                data_mode=data_mode,
                provenance_warning=None,
                error_code="LLM_RATE_LIMITED",
                retry_after=e.retry_after,
                request_id=request_id,
                latency_ms=total_ms,
                stage_latencies={
                    "1_request_received": True,
                    "2_preprocessing_ms": stage_2_preprocessing_ms,
                    "3_embedding_ms": stage_3_embedding_ms,
                    "4_vector_search_ms": stage_4_vector_search_ms,
                    "5_bm25_search_ms": stage_5_bm25_search_ms,
                    "6_rrf_fusion_ms": stage_6_rrf_fusion_ms,
                    "7_context_construction_ms": stage_7_context_construction_ms,
                    "8_llm_dispatch_ms": 0.0,
                    "9_llm_generation_ms": 0.0,
                    "10_total_end_to_end_ms": total_ms,
                },
            )
        except KnowledgeBaseException as e:
            logger.error(f"RAG LLM Error [{request_id}]: {e.code} - {e.message}")
            total_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ChatResponse(
                answer=e.message,
                citations=[],
                property_cards=[],
                confidence_score=0.0,
                grounded=False,
                data_mode=data_mode,
                provenance_warning=None,
                error_code=e.code,
                request_id=request_id,
                latency_ms=total_ms,
                stage_latencies={
                    "1_request_received": True,
                    "2_preprocessing_ms": stage_2_preprocessing_ms,
                    "3_embedding_ms": stage_3_embedding_ms,
                    "4_vector_search_ms": stage_4_vector_search_ms,
                    "5_bm25_search_ms": stage_5_bm25_search_ms,
                    "6_rrf_fusion_ms": stage_6_rrf_fusion_ms,
                    "7_context_construction_ms": stage_7_context_construction_ms,
                    "8_llm_dispatch_ms": 0.0,
                    "9_llm_generation_ms": 0.0,
                    "10_total_end_to_end_ms": total_ms,
                },
            )
        except Exception as e:
            logger.error(f"RAG Unexpected Error [{request_id}]: {e}")
            total_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ChatResponse(
                answer="An unexpected error occurred while communicating with the AI model.",
                citations=[],
                property_cards=[],
                confidence_score=0.0,
                grounded=False,
                data_mode=data_mode,
                provenance_warning=provenance_warning,
                error_code="INTERNAL_ERROR",
                request_id=request_id,
                latency_ms=total_ms,
                stage_latencies={
                    "1_request_received": True,
                    "2_preprocessing_ms": stage_2_preprocessing_ms,
                    "3_embedding_ms": stage_3_embedding_ms,
                    "4_vector_search_ms": stage_4_vector_search_ms,
                    "5_bm25_search_ms": stage_5_bm25_search_ms,
                    "6_rrf_fusion_ms": stage_6_rrf_fusion_ms,
                    "7_context_construction_ms": stage_7_context_construction_ms,
                    "8_llm_dispatch_ms": 0.0,
                    "9_llm_generation_ms": 0.0,
                    "10_total_end_to_end_ms": total_ms,
                },
            )

        # 7. Response Validation & Grounding Checks
        is_num_valid, unsupported_nums = ResponseValidator.validate_numerical_claims(
            answer=answer_text,
            evidence_items=retrieved_items,
        )
        if not is_num_valid:
            logger.warning(
                f"RAG Unsupported numerical claims detected [{request_id}]: {unsupported_nums}"
            )
            answer_text += (
                f"\n\n*(Note: Specific numerical figures like {', '.join(unsupported_nums)} "
                f"could not be fully verified against official source documents. Please verify with official representatives.)*"
            )

        citations = ResponseValidator.build_citations(retrieved_items)
        property_cards = ResponseValidator.build_property_cards(retrieved_items)

        # 10. Total end-to-end response time
        total_end_to_end_ms = round((time.perf_counter() - start_time) * 1000, 2)

        stage_latencies = {
            "1_request_received": True,
            "2_preprocessing_ms": stage_2_preprocessing_ms,
            "3_embedding_ms": stage_3_embedding_ms,
            "4_vector_search_ms": stage_4_vector_search_ms,
            "5_bm25_search_ms": stage_5_bm25_search_ms,
            "6_rrf_fusion_ms": stage_6_rrf_fusion_ms,
            "7_context_construction_ms": stage_7_context_construction_ms,
            "8_llm_dispatch_ms": stage_8_llm_dispatch_ms,
            "9_llm_generation_ms": stage_9_llm_generation_ms,
            "10_total_end_to_end_ms": total_end_to_end_ms,
        }

        logger.info(
            f"RAG Stage Timings [{request_id}]: "
            f"Preprocessing={stage_2_preprocessing_ms}ms, "
            f"Embedding={stage_3_embedding_ms}ms, "
            f"VectorSearch={stage_4_vector_search_ms}ms, "
            f"BM25={stage_5_bm25_search_ms}ms, "
            f"RRF={stage_6_rrf_fusion_ms}ms, "
            f"ContextBuild={stage_7_context_construction_ms}ms, "
            f"LLMDispatch={stage_8_llm_dispatch_ms}ms, "
            f"LLMGeneration={stage_9_llm_generation_ms}ms, "
            f"Total={total_end_to_end_ms}ms | "
            f"RetrievalTotal={total_retrieval_ms}ms, "
            f"LLMTotal={round(stage_8_llm_dispatch_ms + stage_9_llm_generation_ms, 2)}ms"
        )

        return ChatResponse(
            answer=answer_text,
            citations=citations,
            property_cards=property_cards,
            confidence_score=round(top_similarity, 3),
            grounded=is_num_valid,
            data_mode=data_mode,
            provenance_warning=provenance_warning,
            error_code=error_code,
            request_id=request_id,
            latency_ms=total_end_to_end_ms,
            stage_latencies=stage_latencies,
            model=model_used,
        )

    async def stream_answer_question(
        self, payload: ChatRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute conversational RAG pipeline with real-time SSE token streaming."""
        start_time = time.perf_counter()
        settings = get_settings()
        request_id = payload.session_id or f"req_{uuid.uuid4().hex[:12]}"

        t_pre_start = time.perf_counter()
        last_turn = payload.messages[-1]
        raw_query = last_turn.content.strip()
        sanitized_query = sanitize_input_text(raw_query)

        # 1. Lightweight Intent Routing (Zero-retrieval fast-path for greetings & acknowledgements)
        intent, canned_response = classify_intent(sanitized_query)
        is_conversational = intent in (UserIntent.GREETING, UserIntent.ACKNOWLEDGEMENT, UserIntent.OUT_OF_SCOPE)

        logger.info(
            f"RAG Streaming Request [{request_id}]: query='{sanitized_query[:40]}...', "
            f"intent='{intent.value}', "
            f"retrieval_executed={not is_conversational}, "
            f"llm_invoked={not is_conversational}"
        )

        if is_conversational:
            total_ms = round((time.perf_counter() - start_time) * 1000, 2)
            stage_2_preprocessing_ms = round((time.perf_counter() - t_pre_start) * 1000.0, 2)
            logger.info(
                f"RAG Streaming Intent Fast-Path [{request_id}]: intent='{intent.value}' — skipping retrieval and LLM inference."
            )
            yield {"event": "token", "data": {"token": canned_response}}
            yield {
                "event": "citations",
                "data": {
                    "citations": [],
                    "property_cards": [],
                    "grounded": True,
                    "data_mode": "conversational",
                    "provenance_warning": None,
                    "model": None,
                },
            }
            yield {
                "event": "complete",
                "data": {
                    "request_id": request_id,
                    "latency_ms": total_ms,
                    "ttft_ms": total_ms,
                    "generation_ms": 0.0,
                    "total_tokens": len(canned_response.split()),
                    "model": None,
                    "grounded": True,
                    "data_mode": "conversational",
                    "provenance_warning": None,
                    "stage_latencies": {
                        "1_request_received": True,
                        "2_preprocessing_ms": stage_2_preprocessing_ms,
                        "3_embedding_ms": 0.0,
                        "4_vector_search_ms": 0.0,
                        "5_bm25_search_ms": 0.0,
                        "6_rrf_fusion_ms": 0.0,
                        "7_context_construction_ms": 0.0,
                        "ttft_ms": total_ms,
                        "generation_ms": 0.0,
                        "10_total_end_to_end_ms": total_ms,
                    },
                    "disclaimer": "This response is generated by PropIntel AI for informational purposes only.",
                },
            }
            return

        # 2. Status: Searching knowledge base
        yield {
            "event": "status",
            "data": {"stage": "searching", "message": "Searching the knowledge base..."},
        }

        # 3. Contextualize query
        search_query = contextualize_followup_query(sanitized_query, payload.messages)
        stage_2_preprocessing_ms = round((time.perf_counter() - t_pre_start) * 1000.0, 2)

        # 4. Hybrid Retrieval (concurrent semantic vector search and BM25)
        import asyncio
        if hasattr(self.retriever, "aretrieve") and asyncio.iscoroutinefunction(self.retriever.aretrieve):
            retrieved_items, retrieval_timings = await self.retriever.aretrieve(
                query=search_query,
                top_k=settings.RETRIEVAL_TOP_K,
                source_filter=payload.source_filter if payload.source_filter != "all" else None,
                category_filter=payload.category_filter,
                method="hybrid",
            )
        else:
            t_ret = time.perf_counter()
            retrieved_items = self.retriever.retrieve(
                query=search_query,
                top_k=settings.RETRIEVAL_TOP_K,
                source_filter=payload.source_filter if payload.source_filter != "all" else None,
                category_filter=payload.category_filter,
                method="hybrid",
            )
            retrieval_timings = {
                "embedding_ms": 0.0,
                "vector_search_ms": 0.0,
                "bm25_search_ms": 0.0,
                "rrf_fusion_ms": 0.0,
                "total_retrieval_ms": round((time.perf_counter() - t_ret) * 1000, 2),
            }

        stage_3_embedding_ms = retrieval_timings.get("embedding_ms", 0.0)
        stage_4_vector_search_ms = retrieval_timings.get("vector_search_ms", 0.0)
        stage_5_bm25_search_ms = retrieval_timings.get("bm25_search_ms", 0.0)
        stage_6_rrf_fusion_ms = retrieval_timings.get("rrf_fusion_ms", 0.0)
        total_retrieval_ms = retrieval_timings.get("total_retrieval_ms", 0.0)

        # 5. Provenance & Data Mode Assessment
        allow_demo_data = settings.ALLOW_UNVERIFIED_DEMO_DATA
        data_mode: str = "verified"
        provenance_warning: Optional[str] = None

        if not allow_demo_data:
            verified_items = [
                item for item in retrieved_items
                if item.provenance in ("verified_source", "recruiter_provided")
            ]
            if not verified_items:
                refusal_text = (
                    "Verified public records are not currently available for this query because automated "
                    "access to live sources is restricted by website protection policies. "
                    "(Demonstration mode is currently disabled in system settings. "
                    "Set ALLOW_UNVERIFIED_DEMO_DATA=true to explore sample demonstration records)."
                )
                yield {"event": "token", "data": {"token": refusal_text}}
                yield {
                    "event": "citations",
                    "data": {
                        "citations": [],
                        "property_cards": [],
                        "grounded": False,
                        "data_mode": "verified",
                        "provenance_warning": None,
                    },
                }
                total_ms = round((time.perf_counter() - start_time) * 1000, 2)
                yield {
                    "event": "complete",
                    "data": {
                        "request_id": request_id,
                        "latency_ms": total_ms,
                        "ttft_ms": total_ms,
                        "generation_ms": 0.0,
                        "total_tokens": 1,
                        "model": self.llm_client.primary_model,
                        "grounded": False,
                        "data_mode": "verified",
                        "provenance_warning": None,
                    },
                }
                return
            retrieved_items = verified_items
        else:
            has_unverified = any(item.provenance == "demonstration_unverified" for item in retrieved_items)
            if has_unverified:
                data_mode = "demonstration"
                provenance_warning = (
                    "This prototype uses sample property information prepared for demonstration purposes. "
                    "Property details, pricing and availability have not been independently verified. "
                    "Please consult official sources for current information."
                )

        # 5. Evidence Sufficiency & Similarity Cutoff
        if not retrieved_items:
            refusal_text = (
                "Based on the available records, information regarding this question is not currently available "
                "in our knowledge base. Please consult official DarGlobal or Wasalt representatives directly."
            )
            yield {"event": "token", "data": {"token": refusal_text}}
            yield {
                "event": "citations",
                "data": {
                    "citations": [],
                    "property_cards": [],
                    "grounded": False,
                    "data_mode": data_mode,
                    "provenance_warning": provenance_warning,
                },
            }
            total_ms = round((time.perf_counter() - start_time) * 1000, 2)
            yield {
                "event": "complete",
                "data": {
                    "request_id": request_id,
                    "latency_ms": total_ms,
                    "ttft_ms": total_ms,
                    "generation_ms": 0.0,
                    "total_tokens": 1,
                    "model": self.llm_client.primary_model,
                    "grounded": False,
                    "data_mode": data_mode,
                    "provenance_warning": provenance_warning,
                },
            }
            return

        top_similarity = (
            retrieved_items[0].semantic_similarity
            if retrieved_items[0].semantic_similarity is not None
            else 0.5
        )

        if top_similarity < settings.SIMILARITY_THRESHOLD and retrieved_items[0].bm25_score is None:
            refusal_text = (
                f"Based on the available records, I could not find sufficiently reliable information "
                f"regarding '{sanitized_query}'. Please verify directly with official representatives."
            )
            yield {"event": "token", "data": {"token": refusal_text}}
            cites = ResponseValidator.build_citations(retrieved_items[:2])
            yield {
                "event": "citations",
                "data": {
                    "citations": [c.model_dump() for c in cites],
                    "property_cards": [],
                    "grounded": False,
                    "data_mode": data_mode,
                    "provenance_warning": provenance_warning,
                },
            }
            total_ms = round((time.perf_counter() - start_time) * 1000, 2)
            yield {
                "event": "complete",
                "data": {
                    "request_id": request_id,
                    "latency_ms": total_ms,
                    "ttft_ms": total_ms,
                    "generation_ms": 0.0,
                    "total_tokens": 1,
                    "model": self.llm_client.primary_model,
                    "grounded": False,
                    "data_mode": data_mode,
                    "provenance_warning": provenance_warning,
                },
            }
            return

        # 6. Status: Preparing evidence
        yield {
            "event": "status",
            "data": {"stage": "preparing", "message": "Preparing evidence and context..."},
        }

        t_ctx_start = time.perf_counter()
        llm_messages = build_llm_messages(
            query=sanitized_query,
            evidence_items=retrieved_items,
            conversation_history=payload.messages,
            data_mode=data_mode,
        )
        stage_7_context_construction_ms = round((time.perf_counter() - t_ctx_start) * 1000.0, 2)

        # 7. Status: Generating answer
        yield {
            "event": "status",
            "data": {"stage": "generating", "message": "Synthesizing answer with Nemotron-3..."},
        }

        t_gen_start = time.perf_counter()
        first_token_time: Optional[float] = None
        accumulated_tokens: List[str] = []
        model_used: str = self.llm_client.primary_model

        try:
            async for chunk in self.llm_client.stream_completion(llm_messages, request_id=request_id):
                if chunk.get("type") == "token":
                    token = chunk.get("token", "")
                    if token:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        accumulated_tokens.append(token)
                        yield {"event": "token", "data": {"token": token}}
                elif chunk.get("type") == "done":
                    if chunk.get("model"):
                        model_used = chunk["model"]
        except LLMNotConfiguredException as e:
            logger.warning(f"RAG Streaming LLM Not Configured [{request_id}]: {e.message}")
            yield {
                "event": "error",
                "data": {
                    "code": "LLM_NOT_CONFIGURED",
                    "error_code": "LLM_NOT_CONFIGURED",
                    "message": (
                        "OpenRouter LLM service is not configured. "
                        "Please set a valid OPENROUTER_API_KEY in the environment (.env) to enable AI synthesis."
                    ),
                    "citations": [],
                    "property_cards": [],
                    "data_mode": data_mode,
                    "provenance_warning": None,
                },
            }
            return
        except OpenRouterRateLimitError as e:
            logger.warning(
                f"RAG Streaming LLM Rate Limited [{request_id}]: {e.message} (retry_after={e.retry_after})"
            )
            yield {
                "event": "error",
                "data": {
                    "code": "LLM_RATE_LIMITED",
                    "error_code": "LLM_RATE_LIMITED",
                    "message": e.message,
                    "retry_after": e.retry_after,
                    "citations": [],
                    "property_cards": [],
                    "data_mode": data_mode,
                    "provenance_warning": None,
                },
            }
            return
        except KnowledgeBaseException as e:
            logger.error(f"RAG Streaming LLM Error [{request_id}]: {e.code} - {e.message}")
            yield {
                "event": "error",
                "data": {
                    "code": e.code,
                    "error_code": e.code,
                    "message": e.message,
                    "retry_after": getattr(e, "retry_after", None),
                    "citations": [],
                    "property_cards": [],
                    "data_mode": data_mode,
                    "provenance_warning": None,
                },
            }
            return
        except Exception as e:
            logger.error(f"RAG Streaming Unexpected Error [{request_id}]: {e}")
            yield {
                "event": "error",
                "data": {
                    "code": "INTERNAL_ERROR",
                    "error_code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred while communicating with the AI model.",
                    "citations": [],
                    "property_cards": [],
                    "data_mode": data_mode,
                    "provenance_warning": None,
                },
            }
            return

        generation_ms = round((time.perf_counter() - t_gen_start) * 1000, 2)
        ttft_ms = (
            round((first_token_time - start_time) * 1000, 2)
            if first_token_time is not None
            else round((time.perf_counter() - start_time) * 1000, 2)
        )
        full_answer = "".join(accumulated_tokens).strip()

        # 8. Post-Generation Grounding and Numerical Validation
        is_num_valid, unsupported_nums = ResponseValidator.validate_numerical_claims(
            answer=full_answer,
            evidence_items=retrieved_items,
        )
        if not is_num_valid:
            logger.warning(
                f"RAG Streaming Unsupported numerical claims [{request_id}]: {unsupported_nums}"
            )
            correction_note = (
                f"\n\n*(Note: Specific numerical figures like {', '.join(unsupported_nums)} "
                f"could not be fully verified against official source documents. Please verify with official representatives.)*"
            )
            yield {"event": "token", "data": {"token": correction_note}}
            full_answer += correction_note

        # 9. Deliver citations only after final validation
        citations = ResponseValidator.build_citations(retrieved_items)
        property_cards = ResponseValidator.build_property_cards(retrieved_items)

        yield {
            "event": "citations",
            "data": {
                "citations": [c.model_dump() for c in citations],
                "property_cards": [p.model_dump() for p in property_cards],
                "grounded": is_num_valid,
                "data_mode": data_mode,
                "provenance_warning": provenance_warning,
                "model": model_used,
            },
        }

        # 10. Completion event with timing metadata
        total_end_to_end_ms = round((time.perf_counter() - start_time) * 1000, 2)
        stage_latencies = {
            "1_request_received": True,
            "2_preprocessing_ms": stage_2_preprocessing_ms,
            "3_embedding_ms": stage_3_embedding_ms,
            "4_vector_search_ms": stage_4_vector_search_ms,
            "5_bm25_search_ms": stage_5_bm25_search_ms,
            "6_rrf_fusion_ms": stage_6_rrf_fusion_ms,
            "7_context_construction_ms": stage_7_context_construction_ms,
            "ttft_ms": ttft_ms,
            "generation_ms": generation_ms,
            "10_total_end_to_end_ms": total_end_to_end_ms,
        }

        yield {
            "event": "complete",
            "data": {
                "request_id": request_id,
                "latency_ms": total_end_to_end_ms,
                "ttft_ms": ttft_ms,
                "generation_ms": generation_ms,
                "total_tokens": len(accumulated_tokens),
                "model": model_used,
                "grounded": is_num_valid,
                "data_mode": data_mode,
                "provenance_warning": provenance_warning,
                "stage_latencies": stage_latencies,
                "disclaimer": "This response is generated by PropIntel AI for informational purposes only. Terms, pricing, and availability are subject to change by respective developers and brokers.",
            },
        }
