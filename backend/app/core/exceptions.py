"""Custom Domain Exceptions and Error Handlers."""

from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from backend.app.core.logging import logger


class KnowledgeBaseException(Exception):
    """Base exception for all domain-specific errors."""

    def __init__(self, message: str, code: str = "INTERNAL_ERROR", status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class ModelUnavailableException(KnowledgeBaseException):
    """Raised when no configured LLM is reachable or responds successfully."""

    def __init__(self, message: str = "Configured AI models are currently unavailable.", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="MODEL_UNAVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
        )


class RateLimitExceededException(KnowledgeBaseException):
    """Raised when client exceeds allowed request frequency."""

    def __init__(self, message: str = "Rate limit exceeded. Please slow down your requests.", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details,
        )


class PayloadTooLargeException(KnowledgeBaseException):
    """Raised when request body exceeds maximum allowed size."""

    def __init__(self, max_bytes: int):
        super().__init__(
            message=f"Request payload exceeds maximum allowed size of {max_bytes} bytes.",
            code="PAYLOAD_TOO_LARGE",
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            details={"max_bytes": max_bytes},
        )


async def domain_exception_handler(request: Request, exc: KnowledgeBaseException) -> JSONResponse:
    """Standardized JSON response for custom domain exceptions."""
    logger.warning(
        f"Domain exception on {request.method} {request.url.path}: {exc.code} - {exc.message}",
        extra={"extra_fields": {"details": exc.details, "code": exc.code}},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler for unhandled server exceptions."""
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {str(exc)}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again later.",
                "details": {},
            }
        },
    )
