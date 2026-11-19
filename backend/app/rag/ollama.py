"""Ollama Secondary Fallback LLM Client (Qwen 2.5 1.5B Instruct).

Provides an asynchronous client for local or Azure-hosted Ollama instances.
Activated when OpenRouter daily free-tier quota is exhausted or during genuine
OpenRouter provider outages. Communicates via Ollama's OpenAI-compatible API
(/v1/chat/completions) for seamless streaming and schema parity.
"""

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from backend.app.config import get_settings
from backend.app.core.logging import logger
from backend.app.core.exceptions import KnowledgeBaseException


class OllamaUnavailableError(KnowledgeBaseException):
    """Raised when Ollama instance cannot be reached or connection is refused."""

    def __init__(
        self,
        message: str = "Local AI fallback service (Ollama) is currently unavailable or unreachable.",
    ):
        super().__init__(message=message, code="OLLAMA_UNAVAILABLE", status_code=503)
        self.error_code = "OLLAMA_UNAVAILABLE"


class OllamaTimeoutError(KnowledgeBaseException):
    """Raised when Ollama inference exceeds configured timeout."""

    def __init__(
        self,
        message: str = "Local AI fallback generation timed out. Please try again.",
    ):
        super().__init__(message=message, code="OLLAMA_TIMEOUT", status_code=504)
        self.error_code = "OLLAMA_TIMEOUT"


class OllamaClient:
    """Asynchronous client for local or Azure Ollama instance targeting Qwen 2.5 1.5B."""

    _shared_client: Optional[httpx.AsyncClient] = None
    _client_lock: Optional[asyncio.Lock] = None

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        keep_alive: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        settings = get_settings()
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout_seconds = timeout_seconds or settings.OLLAMA_TIMEOUT_SECONDS
        self.keep_alive = keep_alive or settings.OLLAMA_KEEP_ALIVE
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS

    @classmethod
    async def get_shared_client(cls, timeout_seconds: float) -> httpx.AsyncClient:
        """Get or initialize persistent pooled HTTP client for Ollama."""
        if cls._client_lock is None:
            cls._client_lock = asyncio.Lock()

        if cls._shared_client is None or cls._shared_client.is_closed:
            async with cls._client_lock:
                if cls._shared_client is None or cls._shared_client.is_closed:
                    limits = httpx.Limits(
                        max_keepalive_connections=5,
                        max_connections=10,
                        keepalive_expiry=60.0,
                    )
                    cls._shared_client = httpx.AsyncClient(
                        timeout=httpx.Timeout(timeout_seconds, connect=5.0),
                        limits=limits,
                    )
        return cls._shared_client

    @classmethod
    async def close_shared_client(cls) -> None:
        """Close pooled HTTP client on application shutdown."""
        if cls._shared_client is not None and not cls._shared_client.is_closed:
            await cls._shared_client.aclose()
            cls._shared_client = None

    async def check_health(self) -> bool:
        """Check if Ollama service is reachable and responsive."""
        try:
            client = await self.get_shared_client(self.timeout_seconds)
            resp = await client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama healthcheck failed: {e}")
            return False

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request to Ollama via OpenAI-compatible endpoint.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Optional override for sampling temperature.
            max_tokens: Optional override for maximum tokens.
            request_id: Optional request tracing identifier.

        Returns:
            Dict containing 'content', 'model', 'usage', 'dispatch_ms', and 'generation_ms'.

        Raises:
            OllamaUnavailableError: If Ollama service is down or connection refused.
            OllamaTimeoutError: If request times out.
            KnowledgeBaseException: On unhandled HTTP error.
        """
        req_id = request_id or "req_ollama"
        t_start = time.perf_counter()
        temp = temperature if temperature is not None else self.temperature
        max_t = max_tokens or self.max_tokens

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_t,
            "stream": False,
        }

        logger.info(f"Ollama Fallback Dispatch [{req_id}]: endpoint='{self.base_url}', model='{self.model}'...")
        client = await self.get_shared_client(self.timeout_seconds)

        try:
            t_gen_start = time.perf_counter()
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            gen_ms = round((time.perf_counter() - t_gen_start) * 1000, 2)
            total_ms = round((time.perf_counter() - t_start) * 1000, 2)

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    raise OllamaUnavailableError("Ollama returned an empty response choices list.")

                content = choices[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                logger.info(
                    f"Ollama Fallback Success [{req_id}]: model='{self.model}', duration_ms={total_ms}"
                )
                return {
                    "content": content,
                    "model": self.model,
                    "usage": usage,
                    "dispatch_ms": 5.0,
                    "generation_ms": gen_ms,
                }

            if resp.status_code == 404:
                raise OllamaUnavailableError(
                    f"Model '{self.model}' is not installed in Ollama. Run 'ollama pull {self.model}' on the host/container."
                )

            raise OllamaUnavailableError(
                f"Ollama service returned HTTP {resp.status_code}: {resp.text[:120]}"
            )

        except httpx.ConnectError as e:
            logger.error(f"Ollama connection refused at {self.base_url}: {e}")
            raise OllamaUnavailableError(
                f"Local fallback service (Ollama) is not running or unreachable at {self.base_url}."
            ) from e
        except httpx.TimeoutException as e:
            logger.error(f"Ollama request timed out after {self.timeout_seconds}s: {e}")
            raise OllamaTimeoutError(
                f"Local fallback generation timed out after {self.timeout_seconds} seconds."
            ) from e
        except OllamaUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error communicating with Ollama: {e}")
            raise OllamaUnavailableError(f"Unexpected error communicating with local fallback: {e}") from e

    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat completion tokens from Ollama using OpenAI-compatible SSE.

        Yields:
            Dict containing:
                - {"type": "token", "token": str}: incremental text chunk
                - {"type": "done", "model": str}: completion signal

        Raises:
            OllamaUnavailableError: If Ollama service is unreachable.
            OllamaTimeoutError: On stream timeout.
        """
        req_id = request_id or "req_ollama_stream"
        temp = temperature if temperature is not None else self.temperature
        max_t = max_tokens or self.max_tokens

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_t,
            "stream": True,
        }

        logger.info(f"Ollama Fallback Stream Dispatch [{req_id}]: endpoint='{self.base_url}', model='{self.model}'...")
        client = await self.get_shared_client(self.timeout_seconds)

        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status_code == 200:
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                yield {"type": "done", "model": self.model}
                                return
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            choices = chunk.get("choices", [])
                            if choices:
                                content = choices[0].get("delta", {}).get("content")
                                if content:
                                    yield {"type": "token", "token": content}

                    yield {"type": "done", "model": self.model}
                    return

                if resp.status_code == 404:
                    raise OllamaUnavailableError(
                        f"Model '{self.model}' is not pulled in Ollama. Run 'ollama pull {self.model}'."
                    )

                error_bytes = await resp.aread()
                err_text = error_bytes.decode("utf-8", errors="ignore")
                raise OllamaUnavailableError(
                    f"Ollama stream returned HTTP {resp.status_code}: {err_text[:120]}"
                )

        except httpx.ConnectError as e:
            logger.error(f"Ollama stream connection refused at {self.base_url}: {e}")
            raise OllamaUnavailableError(
                f"Local fallback service (Ollama) is not running or unreachable at {self.base_url}."
            ) from e
        except httpx.TimeoutException as e:
            logger.error(f"Ollama stream timed out: {e}")
            raise OllamaTimeoutError(
                f"Local fallback stream timed out after {self.timeout_seconds} seconds."
            ) from e
        except OllamaUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error streaming from Ollama: {e}")
            raise OllamaUnavailableError(f"Unexpected error in local fallback stream: {e}") from e
