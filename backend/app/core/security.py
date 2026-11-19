"""Security and Request Validation Controls.

Provides request size validation, rate limiting, and input sanitization.
"""

from typing import Callable
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.app.core.exceptions import RateLimitExceededException
from backend.app.config import get_settings


# Global rate limiter instance keyed by client IP
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{get_settings().RATE_LIMIT_PER_MINUTE}/minute"],
)


async def rate_limit_error_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Handle rate limit violations with standardized JSON error."""
    raise RateLimitExceededException(
        message="Too many requests. Please wait before submitting another query.",
        details={"limit": exc.detail},
    )


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce maximum request body size (Content-Length check)."""

    def __init__(self, app, max_bytes: int = 65536):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > self.max_bytes:
                    return JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content={
                            "error": {
                                "code": "PAYLOAD_TOO_LARGE",
                                "message": f"Request payload exceeds maximum allowed size of {self.max_bytes} bytes.",
                                "details": {"max_bytes": self.max_bytes},
                            }
                        },
                    )
            except ValueError:
                pass

        return await call_next(request)


def sanitize_input_text(text: str, max_chars: int = 4000) -> str:
    """Strip null bytes, dangerous control characters, and trim to max length."""
    if not text:
        return ""
    # Strip null bytes and non-printable control characters except standard whitespace
    cleaned = "".join(
        ch
        for ch in text
        if ch == "\n" or ch == "\r" or ch == "\t" or (ord(ch) >= 32 and ord(ch) != 127)
    )
    return cleaned.strip()[:max_chars]
