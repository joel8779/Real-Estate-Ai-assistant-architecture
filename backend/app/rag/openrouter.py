"""OpenRouter Chat Completions Client (Primary Provider).

Provides an asynchronous, resilient client for OpenRouter's OpenAI-compatible API
locked strictly to NVIDIA Nemotron 3 Ultra (nvidia/nemotron-3-ultra-550b-a55b:free).
Implements bounded retries on transient errors (502/503/504, connection timeouts)
and strict account-level quota exhaustion detection.
"""

import asyncio
import json
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from backend.app.config import get_settings
from backend.app.core.logging import logger
from backend.app.core.exceptions import KnowledgeBaseException


class LLMNotConfiguredException(KnowledgeBaseException):
    """Raised when OpenRouter API key is missing or not configured."""

    def __init__(self, message: str = "OpenRouter API key is not configured. Please set OPENROUTER_API_KEY in .env."):
        super().__init__(message=message, code="LLM_NOT_CONFIGURED", status_code=503)
        self.error_code = "LLM_NOT_CONFIGURED"


class OpenRouterAuthError(KnowledgeBaseException):
    """Raised on HTTP 401 Unauthorized from OpenRouter."""

    def __init__(self, message: str = "Invalid OpenRouter credentials or unauthorized access."):
        super().__init__(message=message, code="LLM_AUTH_ERROR", status_code=401)
        self.error_code = "LLM_AUTH_ERROR"


class OpenRouterQuotaError(KnowledgeBaseException):
    """Raised on HTTP 402 Payment Required or insufficient credits."""

    def __init__(self, message: str = "OpenRouter credit quota exceeded or payment required."):
        super().__init__(message=message, code="LLM_QUOTA_ERROR", status_code=402)
        self.error_code = "LLM_QUOTA_ERROR"


def parse_retry_after(headers: Any) -> Optional[int]:
    """Extract and parse Retry-After header in seconds, if provided."""
    if not hasattr(headers, "get"):
        return None
    val = headers.get("retry-after") or headers.get("Retry-After")
    if not val:
        return None
    val = str(val).strip()
    try:
        seconds = int(val)
        return seconds if seconds > 0 else None
    except ValueError:
        try:
            import datetime
            import email.utils
            dt = email.utils.parsedate_to_datetime(val)
            now = datetime.datetime.now(datetime.timezone.utc)
            diff = int((dt - now).total_seconds())
            return diff if diff > 0 else None
        except Exception:
            return None


def is_account_level_quota_exhausted(err_text: str) -> bool:
    """Detect if HTTP 429 represents an account-wide daily limit (e.g. 50 req/day).

    OpenRouter free models share an account-wide daily limit. When this is hit,
    trying another free model will also fail, so secondary fallback must be invoked.
    """
    if not err_text:
        return False
    lower = err_text.lower()
    quota_indicators = [
        "free-models-per-day",
        "requests/day",
        "requests per day",
        "account-wide",
        "account quota",
        "daily limit",
        "daily quota",
        "credit limit",
        "user-level",
    ]
    return any(indicator in lower for indicator in quota_indicators)


class OpenRouterRateLimitError(KnowledgeBaseException):
    """Raised when model-specific rate limits are exceeded across attempts."""

    def __init__(
        self,
        message: Optional[str] = None,
        retry_after: Optional[int] = None,
    ):
        if not message:
            if retry_after is not None and retry_after > 0:
                message = f"The AI service is temporarily busy. Please try again in {retry_after} seconds."
            else:
                message = "The AI service is temporarily busy. Please wait a moment and try again."
        super().__init__(message=message, code="LLM_RATE_LIMITED", status_code=429)
        self.error_code = "LLM_RATE_LIMITED"
        self.retry_after = retry_after


class OpenRouterAccountQuotaExhaustedError(OpenRouterRateLimitError):
    """Raised when OpenRouter account-wide free-model daily quota is exhausted."""

    def __init__(
        self,
        message: str = "OpenRouter free daily request limit reached for this account. Please try again tomorrow or upgrade your OpenRouter account.",
        retry_after: Optional[int] = None,
    ):
        super().__init__(message=message, retry_after=retry_after)
        self.error_code = "ACCOUNT_DAILY_QUOTA_EXHAUSTED"


class OpenRouterServiceError(KnowledgeBaseException):
    """Raised when OpenRouter service is unavailable or encounters upstream failure."""

    def __init__(self, message: str = "The AI service is temporarily unavailable. Please try again later."):
        super().__init__(message=message, code="LLM_SERVICE_UNAVAILABLE", status_code=503)
        self.error_code = "LLM_SERVICE_UNAVAILABLE"


class OpenRouterClient:
    """Asynchronous OpenRouter LLM client locked strictly to NVIDIA Nemotron 3 Ultra."""

    _shared_client: Optional[httpx.AsyncClient] = None
    _client_lock: Optional[asyncio.Lock] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        primary_model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.OPENROUTER_API_KEY
        self.base_url = (base_url or settings.OPENROUTER_BASE_URL).rstrip("/")
        self.primary_model = primary_model or settings.OPENROUTER_MODEL
        self.timeout_seconds = timeout_seconds or settings.LLM_TIMEOUT_SECONDS
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS

    @classmethod
    async def get_shared_client(cls, timeout_seconds: float) -> httpx.AsyncClient:
        """Get or initialize persistent pooled HTTP client."""
        if cls._client_lock is None:
            cls._client_lock = asyncio.Lock()

        if cls._shared_client is None or cls._shared_client.is_closed:
            async with cls._client_lock:
                if cls._shared_client is None or cls._shared_client.is_closed:
                    limits = httpx.Limits(
                        max_keepalive_connections=10,
                        max_connections=20,
                        keepalive_expiry=60.0,
                    )
                    cls._shared_client = httpx.AsyncClient(
                        timeout=httpx.Timeout(timeout_seconds, connect=10.0),
                        limits=limits,
                    )
        return cls._shared_client

    @classmethod
    async def close_shared_client(cls) -> None:
        """Close pooled HTTP client on application shutdown."""
        if cls._shared_client is not None and not cls._shared_client.is_closed:
            await cls._shared_client.aclose()
            cls._shared_client = None

    def is_configured(self) -> bool:
        """Check whether a non-empty API key is present."""
        return bool(self.api_key and self.api_key.strip())

    def _get_headers(self) -> Dict[str, str]:
        """Construct secure OpenRouter request headers."""
        if not self.is_configured():
            raise LLMNotConfiguredException()

        return {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "HTTP-Referer": "https://darglobal.co.uk",
            "X-Title": "DarGlobal Real Estate Knowledge Assistant",
            "Content-Type": "application/json",
        }

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request to OpenRouter (Nemotron 3 Ultra).

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Optional override for sampling temperature.
            max_tokens: Optional override for maximum tokens.
            request_id: Optional request tracing identifier.

        Returns:
            Dict containing 'content', 'model', 'usage', 'dispatch_ms', and 'generation_ms'.

        Raises:
            LLMNotConfiguredException: If API key is missing.
            OpenRouterAuthError: On 401 response (permanent).
            OpenRouterQuotaError: On 402 response (permanent).
            OpenRouterAccountQuotaExhaustedError: On account daily quota 429.
            OpenRouterRateLimitError: On model 429 after retry.
            OpenRouterServiceError: On 502/503/504 or network timeout after retry.
        """
        req_id = request_id or "req_direct"
        t_route_start = time.perf_counter()

        if not self.is_configured():
            raise LLMNotConfiguredException()

        t_dispatch_start = time.perf_counter()

        if os.environ.get("STREAM_TEST_MODE") == "true":
            return {
                "content": (
                    "AIDA is a prestigious cliffside master development located in Muscat, Oman, "
                    "developed in partnership with Trump International and featuring an 18-hole championship golf course."
                ),
                "model": self.primary_model,
                "usage": {"prompt_tokens": 100, "completion_tokens": 30},
                "dispatch_ms": 15.0,
                "generation_ms": 720.0,
            }

        headers = self._get_headers()
        temp = temperature if temperature is not None else self.temperature
        max_t = max_tokens or self.max_tokens
        client = await self.get_shared_client(self.timeout_seconds)
        dispatch_ms = round((time.perf_counter() - t_dispatch_start) * 1000, 2)

        model = self.primary_model
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_t,
        }

        logger.info(f"OpenRouter Primary Dispatch [{req_id}]: model='{model}'...")

        # Up to 2 attempts for transient errors (502/503/504, connection timeouts)
        for attempt in range(1, 3):
            try:
                t_gen_start = time.perf_counter()
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                gen_ms = round((time.perf_counter() - t_gen_start) * 1000, 2)
                total_routing_ms = round((time.perf_counter() - t_route_start) * 1000, 2)

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise ValueError("Empty choices list in OpenRouter response.")

                    message_obj = choices[0].get("message", {})
                    content = message_obj.get("content", "")
                    usage = data.get("usage", {})

                    logger.info(
                        f"OpenRouter Model Routing [{req_id}]: "
                        f"model='{model}', "
                        f"http_status=200, "
                        f"duration_ms={total_routing_ms}"
                    )
                    return {
                        "content": content,
                        "model": model,
                        "usage": usage,
                        "dispatch_ms": dispatch_ms,
                        "generation_ms": gen_ms,
                    }

                # Permanent client / auth errors: never retry, never failover
                if resp.status_code == 400:
                    raise KnowledgeBaseException(
                        message=f"Invalid request to AI model: {resp.text[:150]}",
                        code="LLM_INVALID_REQUEST",
                        status_code=400,
                    )

                if resp.status_code == 401:
                    raise OpenRouterAuthError()

                if resp.status_code == 402:
                    raise OpenRouterQuotaError()

                # Rate limits (429)
                if resp.status_code == 429:
                    retry_after = parse_retry_after(getattr(resp, "headers", None))
                    err_text = resp.text
                    if is_account_level_quota_exhausted(err_text):
                        logger.warning(
                            f"OpenRouter Account Daily Quota Exhausted [{req_id}]: {err_text[:120]}"
                        )
                        raise OpenRouterAccountQuotaExhaustedError(retry_after=retry_after)

                    # Model-specific transient rate limit: retry once with backoff
                    if attempt < 2:
                        logger.warning(f"OpenRouter 429 model rate limit [{req_id}], retrying once...")
                        await asyncio.sleep(1.0)
                        continue
                    raise OpenRouterRateLimitError(retry_after=retry_after)

                # Upstream outage (500, 502, 503, 504)
                if resp.status_code in (500, 502, 503, 504):
                    if attempt < 2:
                        logger.warning(f"OpenRouter HTTP {resp.status_code} [{req_id}], retrying once...")
                        await asyncio.sleep(1.0)
                        continue
                    raise OpenRouterServiceError(
                        f"OpenRouter upstream service error (HTTP {resp.status_code}): {resp.text[:120]}"
                    )

                raise OpenRouterServiceError(f"HTTP {resp.status_code}: {resp.text[:120]}")

            except (
                OpenRouterAuthError,
                OpenRouterQuotaError,
                OpenRouterAccountQuotaExhaustedError,
                OpenRouterRateLimitError,
                OpenRouterServiceError,
                KnowledgeBaseException,
            ):
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                if attempt < 2:
                    logger.warning(f"OpenRouter network/timeout error [{req_id}], retrying once: {net_err}")
                    await asyncio.sleep(1.0)
                    continue
                raise OpenRouterServiceError(f"OpenRouter network timeout / connection error: {str(net_err)}")

        raise OpenRouterServiceError("OpenRouter failed to respond after retries.")

    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat completion tokens from OpenRouter using Server-Sent Events (SSE).

        Yields:
            Dict containing:
                - {"type": "token", "token": str}: incremental text chunk
                - {"type": "usage", "usage": dict}: token usage if provided
                - {"type": "done", "model": str}: completion signal

        Raises:
            LLMNotConfiguredException: If API key is missing.
            OpenRouterAuthError: On 401 response (permanent).
            OpenRouterQuotaError: On 402 response (permanent).
            OpenRouterAccountQuotaExhaustedError: On account quota exhaustion before stream starts.
            OpenRouterRateLimitError: On model 429 before stream starts.
            OpenRouterServiceError: On 502/503/504, network timeout, or if stream drops mid-stream.
        """
        req_id = request_id or "req_direct"
        t_route_start = time.perf_counter()

        if not self.is_configured():
            raise LLMNotConfiguredException()

        if os.environ.get("STREAM_TEST_MODE") == "true":
            logger.info("STREAM_TEST_MODE active: Emitting calibrated token stream for verification.")
            test_tokens = [
                "AIDA ", "is a prestigious ", "cliffside master development ",
                "located in Muscat, Oman, ", "developed in partnership with Trump International ",
                "and featuring an 18-hole championship golf course."
            ]
            for tok in test_tokens:
                await asyncio.sleep(0.12)
                yield {"type": "token", "token": tok}
            yield {"type": "done", "model": self.primary_model}
            return

        headers = self._get_headers()
        temp = temperature if temperature is not None else self.temperature
        max_t = max_tokens or self.max_tokens
        client = await self.get_shared_client(self.timeout_seconds)

        model = self.primary_model
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_t,
            "stream": True,
        }

        logger.info(f"OpenRouter Primary Stream Dispatch [{req_id}]: model='{model}'...")

        # Up to 2 attempts before stream starts
        for attempt in range(1, 3):
            stream_started = False
            tokens_emitted = 0
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as resp:
                    total_routing_ms = round((time.perf_counter() - t_route_start) * 1000, 2)
                    if resp.status_code == 200:
                        async for line in resp.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("data:"):
                                data_str = line[5:].strip()
                                if data_str == "[DONE]":
                                    yield {"type": "done", "model": model}
                                    return
                                try:
                                    chunk = json.loads(data_str)
                                except json.JSONDecodeError:
                                    continue

                                if "error" in chunk:
                                    err_obj = chunk.get("error", {})
                                    err_msg = err_obj.get("message", "Stream chunk error")
                                    if stream_started:
                                        # NEVER switch providers mid-stream once tokens have been delivered!
                                        logger.error(f"Error received mid-stream on model '{model}': {err_msg}")
                                        raise OpenRouterServiceError(f"Stream interrupted: {err_msg}")
                                    if is_account_level_quota_exhausted(err_msg):
                                        raise OpenRouterAccountQuotaExhaustedError(message=err_msg)
                                    raise OpenRouterServiceError(f"OpenRouter stream error: {err_msg}")

                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        stream_started = True
                                        tokens_emitted += 1
                                        yield {"type": "token", "token": content}

                                usage = chunk.get("usage")
                                if usage:
                                    yield {"type": "usage", "usage": usage}

                        yield {"type": "done", "model": model}
                        return

                    # Error before stream started
                    error_bytes = await resp.aread()
                    err_text = error_bytes.decode("utf-8", errors="ignore")

                    if resp.status_code == 400:
                        raise KnowledgeBaseException(
                            message=f"Invalid request to AI model: {err_text[:150]}",
                            code="LLM_INVALID_REQUEST",
                            status_code=400,
                        )

                    if resp.status_code == 401:
                        raise OpenRouterAuthError()

                    if resp.status_code == 402:
                        raise OpenRouterQuotaError()

                    if resp.status_code == 429:
                        retry_after = parse_retry_after(getattr(resp, "headers", None))
                        if is_account_level_quota_exhausted(err_text):
                            logger.warning(
                                f"OpenRouter Stream Account Quota Exhausted [{req_id}]: {err_text[:120]}"
                            )
                            raise OpenRouterAccountQuotaExhaustedError(retry_after=retry_after)

                        if attempt < 2:
                            logger.warning(f"OpenRouter Stream 429 rate limit [{req_id}], retrying once...")
                            await asyncio.sleep(1.0)
                            continue
                        raise OpenRouterRateLimitError(retry_after=retry_after)

                    if resp.status_code in (500, 502, 503, 504):
                        if attempt < 2:
                            logger.warning(f"OpenRouter Stream HTTP {resp.status_code} [{req_id}], retrying once...")
                            await asyncio.sleep(1.0)
                            continue
                        raise OpenRouterServiceError(
                            f"OpenRouter stream service unavailable (HTTP {resp.status_code}): {err_text[:120]}"
                        )

                    raise OpenRouterServiceError(f"HTTP {resp.status_code}: {err_text[:120]}")

            except (
                OpenRouterAuthError,
                OpenRouterQuotaError,
                OpenRouterAccountQuotaExhaustedError,
                OpenRouterRateLimitError,
                OpenRouterServiceError,
                KnowledgeBaseException,
            ):
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                if stream_started:
                    # NEVER switch providers mid-stream!
                    logger.error(f"Network error mid-stream on model '{model}' after {tokens_emitted} tokens: {net_err}")
                    raise OpenRouterServiceError(f"Stream interrupted by network error: {net_err}")
                if attempt < 2:
                    logger.warning(f"OpenRouter Stream network error [{req_id}], retrying once: {net_err}")
                    await asyncio.sleep(1.0)
                    continue
                raise OpenRouterServiceError(f"OpenRouter stream network timeout / connection error: {str(net_err)}")

        raise OpenRouterServiceError("OpenRouter failed to start streaming after retries.")
