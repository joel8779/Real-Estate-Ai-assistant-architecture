"""OpenRouter Chat Completions Client.

Provides an asynchronous, resilient client for OpenRouter's OpenAI-compatible API
with automatic model fallbacks, bounded retries on transient errors (429/5xx),
and strict credential protection.
"""

import asyncio
import json
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
    trying another free fallback model will also fail, so fallback must NOT be attempted.
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
    """Raised when rate limits are exceeded across attempts."""

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


class OpenRouterServiceError(KnowledgeBaseException):
    """Raised when OpenRouter service is unavailable."""

    def __init__(self, message: str = "The AI service is temporarily unavailable. Please try again later."):
        super().__init__(message=message, code="LLM_SERVICE_UNAVAILABLE", status_code=503)
        self.error_code = "LLM_SERVICE_UNAVAILABLE"


class OpenRouterClient:
    """Asynchronous OpenRouter LLM client with connection pooling, model failover and retries."""

    _shared_client: Optional[httpx.AsyncClient] = None
    _client_lock: Optional[asyncio.Lock] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        primary_model: Optional[str] = None,
        allow_fallback: Optional[bool] = None,
        fallback_models: Optional[List[str]] = None,
        timeout_seconds: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.OPENROUTER_API_KEY
        self.base_url = (base_url or settings.OPENROUTER_BASE_URL).rstrip("/")
        self.primary_model = primary_model or settings.OPENROUTER_MODEL
        self.allow_fallback = (
            allow_fallback if allow_fallback is not None else settings.OPENROUTER_ALLOW_FALLBACK
        )
        self.fallback_models = (
            fallback_models if fallback_models is not None else settings.OPENROUTER_FALLBACK_MODELS
        )
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
        """Send chat completion request to OpenRouter with automatic fallback chain and timing metrics.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Optional override for sampling temperature.
            max_tokens: Optional override for maximum tokens.
            request_id: Optional request tracing identifier.

        Returns:
            Dict containing 'content', 'model', 'usage', 'dispatch_ms', and 'generation_ms'.

        Raises:
            LLMNotConfiguredException: If API key is missing.
            OpenRouterAuthError: On 401 response.
            OpenRouterQuotaError: On 402 response.
            OpenRouterServiceError: If all candidate models fail.
        """
        import time

        req_id = request_id or "req_direct"
        t_route_start = time.perf_counter()

        if not self.is_configured():
            raise LLMNotConfiguredException()

        t_dispatch_start = time.perf_counter()

        import os
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

        # Build candidate models list: primary, and fallbacks only if allow_fallback is enabled (Priority 1, 2, 3)
        candidate_models = [self.primary_model]
        if self.allow_fallback:
            for m in self.fallback_models:
                if m and m not in candidate_models:
                    candidate_models.append(m)
                if len(candidate_models) >= 3:
                    break
        else:
            logger.debug(
                f"Model locked to '{self.primary_model}'. Automatic fallback is disabled by default."
            )

        headers = self._get_headers()
        temp = temperature if temperature is not None else self.temperature
        max_t = max_tokens or self.max_tokens

        last_error_msg = ""
        client = await self.get_shared_client(self.timeout_seconds)

        dispatch_ms = round((time.perf_counter() - t_dispatch_start) * 1000, 2)

        for model_idx, model in enumerate(candidate_models):
            priority_label = (
                "Priority 1 (Primary)" if model_idx == 0
                else f"Priority {model_idx + 1} (Fallback {model_idx})"
            )
            logger.info(f"OpenRouter Attempting [{req_id}]: model='{model}', priority='{priority_label}'...")

            # Allow up to 2 attempts per model for transient errors
            for attempt in range(1, 3):
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_t,
                }

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
                            f"attempt_priority='{priority_label}', "
                            f"http_status=200, "
                            f"failure_classification='NONE (SUCCESS)', "
                            f"fallback_triggered={model_idx > 0}, "
                            f"final_model='{model}', "
                            f"total_routing_duration_ms={total_routing_ms}"
                        )
                        return {
                            "content": content,
                            "model": model,
                            "usage": usage,
                            "dispatch_ms": dispatch_ms,
                            "generation_ms": gen_ms,
                        }

                    # Specific error status handling
                    if resp.status_code == 400:
                        logger.error(
                            f"OpenRouter Model Routing [{req_id}]: "
                            f"model='{model}', "
                            f"attempt_priority='{priority_label}', "
                            f"http_status=400, "
                            f"failure_classification='INVALID_REQUEST_400', "
                            f"fallback_triggered=False, "
                            f"final_model=None, "
                            f"total_routing_duration_ms={total_routing_ms}"
                        )
                        raise KnowledgeBaseException(
                            message=f"Invalid request to AI model: {resp.text[:150]}",
                            code="LLM_INVALID_REQUEST",
                            status_code=400,
                        )

                    if resp.status_code == 401:
                        logger.error(
                            f"OpenRouter Model Routing [{req_id}]: "
                            f"model='{model}', "
                            f"attempt_priority='{priority_label}', "
                            f"http_status=401, "
                            f"failure_classification='AUTH_ERROR_401', "
                            f"fallback_triggered=False, "
                            f"final_model=None, "
                            f"total_routing_duration_ms={total_routing_ms}"
                        )
                        raise OpenRouterAuthError()

                    if resp.status_code == 402:
                        logger.error(
                            f"OpenRouter Model Routing [{req_id}]: "
                            f"model='{model}', "
                            f"attempt_priority='{priority_label}', "
                            f"http_status=402, "
                            f"failure_classification='QUOTA_EXCEEDED_402', "
                            f"fallback_triggered=False, "
                            f"final_model=None, "
                            f"total_routing_duration_ms={total_routing_ms}"
                        )
                        raise OpenRouterQuotaError()

                    if resp.status_code == 429:
                        retry_after = parse_retry_after(getattr(resp, "headers", None))
                        err_text = resp.text
                        if is_account_level_quota_exhausted(err_text):
                            logger.warning(
                                f"OpenRouter Model Routing [{req_id}]: "
                                f"model='{model}', "
                                f"attempt_priority='{priority_label}', "
                                f"http_status=429, "
                                f"failure_classification='ACCOUNT_DAILY_QUOTA_EXHAUSTED', "
                                f"fallback_triggered=False, "
                                f"final_model=None, "
                                f"total_routing_duration_ms={total_routing_ms}"
                            )
                            raise OpenRouterRateLimitError(
                                message="OpenRouter free daily request limit reached for this account. Please try again tomorrow or upgrade your OpenRouter account.",
                                retry_after=retry_after,
                            )

                        fallback_triggered = bool(self.allow_fallback and model != candidate_models[-1])
                        logger.warning(
                            f"OpenRouter Model Routing [{req_id}]: "
                            f"model='{model}', "
                            f"attempt_priority='{priority_label}', "
                            f"http_status=429, "
                            f"failure_classification='MODEL_RATE_LIMITED_429', "
                            f"fallback_triggered={fallback_triggered}, "
                            f"final_model=None, "
                            f"total_routing_duration_ms={total_routing_ms}"
                        )
                        if self.allow_fallback:
                            if attempt < 2:
                                await asyncio.sleep(1.0 * attempt)
                                continue
                            if model != candidate_models[-1]:
                                last_error_msg = f"Rate limited on model '{model}'"
                                break
                        raise OpenRouterRateLimitError(retry_after=retry_after)

                    if resp.status_code in (500, 502, 503, 504, 404):
                        fail_class = "MODEL_NOT_FOUND_404" if resp.status_code == 404 else f"UPSTREAM_{resp.status_code}"
                        fallback_triggered = bool(self.allow_fallback and model != candidate_models[-1])
                        logger.warning(
                            f"OpenRouter Model Routing [{req_id}]: "
                            f"model='{model}', "
                            f"attempt_priority='{priority_label}', "
                            f"http_status={resp.status_code}, "
                            f"failure_classification='{fail_class}', "
                            f"fallback_triggered={fallback_triggered}, "
                            f"final_model=None, "
                            f"total_routing_duration_ms={total_routing_ms}"
                        )
                        if attempt < 2:
                            await asyncio.sleep(1.0 * attempt)
                            continue
                        if self.allow_fallback and model != candidate_models[-1]:
                            last_error_msg = f"HTTP {resp.status_code} on {model}"
                            break
                        last_error_msg = f"HTTP {resp.status_code} on {model}"
                        break  # Try next model

                    # Other unexpected status codes
                    last_error_msg = f"HTTP {resp.status_code}: {resp.text[:150]}"
                    logger.warning(f"Unexpected status code for '{model}': {last_error_msg}")
                    break

                except (OpenRouterAuthError, OpenRouterQuotaError, OpenRouterRateLimitError, KnowledgeBaseException):
                    raise
                except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                    total_routing_ms = round((time.perf_counter() - t_route_start) * 1000, 2)
                    fallback_triggered = bool(self.allow_fallback and model != candidate_models[-1])
                    logger.warning(
                        f"OpenRouter Model Routing [{req_id}]: "
                        f"model='{model}', "
                        f"attempt_priority='{priority_label}', "
                        f"http_status=0, "
                        f"failure_classification='NETWORK_OR_TIMEOUT_ERROR', "
                        f"fallback_triggered={fallback_triggered}, "
                        f"final_model=None, "
                        f"total_routing_duration_ms={total_routing_ms}"
                    )
                    if attempt < 2:
                        await asyncio.sleep(1.0)
                        continue
                    if self.allow_fallback and model != candidate_models[-1]:
                        last_error_msg = f"Network timeout on {model}: {str(net_err)}"
                        break
                    last_error_msg = f"Network timeout on {model}: {str(net_err)}"
                    break

        # If we exited the loop without returning, all candidate models failed
        logger.error(f"All candidate OpenRouter models failed. Last error: {last_error_msg}")
        raise OpenRouterServiceError(
            f"All configured models failed to respond. Last error: {last_error_msg}"
        )

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
            OpenRouterAuthError: On 401 response.
            OpenRouterQuotaError: On 402 response.
            OpenRouterRateLimitError: On 429 response.
            OpenRouterServiceError: If stream fails or model returns 5xx/network error.
        """
        req_id = request_id or "req_direct"
        t_route_start = time.perf_counter()

        if not self.is_configured():
            raise LLMNotConfiguredException()

        import os
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

        # Build candidate models list: primary, and fallbacks only if allow_fallback is enabled (Priority 1, 2, 3)
        candidate_models = [self.primary_model]
        if self.allow_fallback:
            for m in self.fallback_models:
                if m and m not in candidate_models:
                    candidate_models.append(m)
                if len(candidate_models) >= 3:
                    break
        else:
            logger.debug(
                f"Model locked to '{self.primary_model}'. Automatic fallback is disabled by default."
            )

        headers = self._get_headers()
        temp = temperature if temperature is not None else self.temperature
        max_t = max_tokens or self.max_tokens

        client = await self.get_shared_client(self.timeout_seconds)
        last_error_msg = ""

        for model_idx, model in enumerate(candidate_models):
            priority_label = (
                "Priority 1 (Primary)" if model_idx == 0
                else f"Priority {model_idx + 1} (Fallback {model_idx})"
            )
            logger.info(f"OpenRouter Stream Attempting [{req_id}]: model='{model}', priority='{priority_label}'...")

            for attempt in range(1, 3):
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_t,
                    "stream": True,
                }

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
                            logger.info(
                                f"OpenRouter Model Routing [{req_id}]: "
                                f"model='{model}', "
                                f"attempt_priority='{priority_label}', "
                                f"http_status=200, "
                                f"failure_classification='NONE (SUCCESS)', "
                                f"fallback_triggered={model_idx > 0}, "
                                f"final_model='{model}', "
                                f"total_routing_duration_ms={total_routing_ms}"
                            )
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
                                            logger.error(f"Error received mid-stream on model '{model}': {err_msg}")
                                            raise OpenRouterServiceError(f"Stream interrupted: {err_msg}")
                                        else:
                                            if is_account_level_quota_exhausted(err_msg):
                                                raise OpenRouterRateLimitError(message=err_msg)
                                            if self.allow_fallback and model != candidate_models[-1]:
                                                last_error_msg = f"Stream error on {model}: {err_msg}"
                                                break
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

                        # Non-200 responses before stream started
                        error_bytes = await resp.aread()
                        err_text = error_bytes.decode("utf-8", errors="ignore")

                        if resp.status_code == 400:
                            logger.error(
                                f"OpenRouter Model Routing [{req_id}]: "
                                f"model='{model}', "
                                f"attempt_priority='{priority_label}', "
                                f"http_status=400, "
                                f"failure_classification='INVALID_REQUEST_400', "
                                f"fallback_triggered=False, "
                                f"final_model=None, "
                                f"total_routing_duration_ms={total_routing_ms}"
                            )
                            raise KnowledgeBaseException(
                                message=f"Invalid request to AI model: {err_text[:150]}",
                                code="LLM_INVALID_REQUEST",
                                status_code=400,
                            )

                        if resp.status_code == 401:
                            logger.error(
                                f"OpenRouter Model Routing [{req_id}]: "
                                f"model='{model}', "
                                f"attempt_priority='{priority_label}', "
                                f"http_status=401, "
                                f"failure_classification='AUTH_ERROR_401', "
                                f"fallback_triggered=False, "
                                f"final_model=None, "
                                f"total_routing_duration_ms={total_routing_ms}"
                            )
                            raise OpenRouterAuthError()

                        if resp.status_code == 402:
                            logger.error(
                                f"OpenRouter Model Routing [{req_id}]: "
                                f"model='{model}', "
                                f"attempt_priority='{priority_label}', "
                                f"http_status=402, "
                                f"failure_classification='QUOTA_EXCEEDED_402', "
                                f"fallback_triggered=False, "
                                f"final_model=None, "
                                f"total_routing_duration_ms={total_routing_ms}"
                            )
                            raise OpenRouterQuotaError()

                        if resp.status_code == 429:
                            retry_after = parse_retry_after(getattr(resp, "headers", None))
                            if is_account_level_quota_exhausted(err_text):
                                logger.warning(
                                    f"OpenRouter Model Routing [{req_id}]: "
                                    f"model='{model}', "
                                    f"attempt_priority='{priority_label}', "
                                    f"http_status=429, "
                                    f"failure_classification='ACCOUNT_DAILY_QUOTA_EXHAUSTED', "
                                    f"fallback_triggered=False, "
                                    f"final_model=None, "
                                    f"total_routing_duration_ms={total_routing_ms}"
                                )
                                raise OpenRouterRateLimitError(
                                    message="OpenRouter free daily request limit reached for this account. Please try again tomorrow or upgrade your OpenRouter account.",
                                    retry_after=retry_after,
                                )

                            fallback_triggered = bool(self.allow_fallback and model != candidate_models[-1])
                            logger.warning(
                                f"OpenRouter Model Routing [{req_id}]: "
                                f"model='{model}', "
                                f"attempt_priority='{priority_label}', "
                                f"http_status=429, "
                                f"failure_classification='MODEL_RATE_LIMITED_429', "
                                f"fallback_triggered={fallback_triggered}, "
                                f"final_model=None, "
                                f"total_routing_duration_ms={total_routing_ms}"
                            )
                            if self.allow_fallback:
                                if attempt < 2:
                                    await asyncio.sleep(1.0 * attempt)
                                    continue
                                if model != candidate_models[-1]:
                                    last_error_msg = f"Rate limited on model '{model}'"
                                    break
                            raise OpenRouterRateLimitError(retry_after=retry_after)

                        if resp.status_code in (500, 502, 503, 504, 404):
                            fail_class = "MODEL_NOT_FOUND_404" if resp.status_code == 404 else f"UPSTREAM_{resp.status_code}"
                            fallback_triggered = bool(self.allow_fallback and model != candidate_models[-1])
                            logger.warning(
                                f"OpenRouter Model Routing [{req_id}]: "
                                f"model='{model}', "
                                f"attempt_priority='{priority_label}', "
                                f"http_status={resp.status_code}, "
                                f"failure_classification='{fail_class}', "
                                f"fallback_triggered={fallback_triggered}, "
                                f"final_model=None, "
                                f"total_routing_duration_ms={total_routing_ms}"
                            )
                            if attempt < 2:
                                await asyncio.sleep(1.0 * attempt)
                                continue
                            if self.allow_fallback and model != candidate_models[-1]:
                                last_error_msg = f"HTTP {resp.status_code} on {model}: {err_text[:150]}"
                                break
                            last_error_msg = f"HTTP {resp.status_code} on {model}: {err_text[:150]}"
                            break

                        last_error_msg = f"HTTP {resp.status_code}: {err_text[:150]}"
                        break

                except (OpenRouterAuthError, OpenRouterQuotaError, OpenRouterRateLimitError, KnowledgeBaseException):
                    raise
                except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                    if stream_started:
                        # Tokens were already emitted! Strictly DO NOT fallback or concatenate another model!
                        logger.error(f"Network error mid-stream for model '{model}' after {tokens_emitted} tokens: {net_err}")
                        raise OpenRouterServiceError(f"Stream interrupted by network error: {net_err}")

                    total_routing_ms = round((time.perf_counter() - t_route_start) * 1000, 2)
                    fallback_triggered = bool(self.allow_fallback and model != candidate_models[-1])
                    logger.warning(
                        f"OpenRouter Model Routing [{req_id}]: "
                        f"model='{model}', "
                        f"attempt_priority='{priority_label}', "
                        f"http_status=0, "
                        f"failure_classification='NETWORK_OR_TIMEOUT_ERROR', "
                        f"fallback_triggered={fallback_triggered}, "
                        f"final_model=None, "
                        f"total_routing_duration_ms={total_routing_ms}"
                    )
                    if attempt < 2:
                        await asyncio.sleep(1.0)
                        continue
                    if self.allow_fallback and model != candidate_models[-1]:
                        last_error_msg = f"Network timeout on {model}: {str(net_err)}"
                        break
                    last_error_msg = f"Network timeout on {model}: {str(net_err)}"
                    break

        raise OpenRouterServiceError(
            f"All configured models failed to start streaming. Last error: {last_error_msg}"
        )
