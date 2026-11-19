"""DarGlobal & Wasalt Real Estate AI Knowledge Assistant.

FastAPI Application Entry Point.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from backend.app.config import get_settings
from backend.app.core.logging import logger, setup_logging
from backend.app.core.security import limiter, RequestSizeLimitMiddleware, rate_limit_error_handler
from backend.app.core.exceptions import (
    KnowledgeBaseException,
    domain_exception_handler,
    global_exception_handler,
)
from backend.app.api import health_router, chat_router, retrieve_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle management."""
    settings = get_settings()
    setup_logging(log_level=settings.LOG_LEVEL, json_format=not settings.DEBUG)
    logger.info(
        f"Starting {settings.APP_NAME} v{settings.APP_VERSION} [{settings.ENVIRONMENT}]"
    )
    logger.info(f"CORS Allowed Origins: {settings.CORS_ORIGINS}")
    logger.info(f"Max Request Size: {settings.MAX_REQUEST_SIZE_BYTES} bytes")
    logger.info(f"Rate Limit: {settings.RATE_LIMIT_PER_MINUTE} req/min")
    yield
    from backend.app.rag.openrouter import OpenRouterClient
    await OpenRouterClient.close_shared_client()
    logger.info(f"Shutting down {settings.APP_NAME}")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Grounded AI Knowledge Assistant for DarGlobal luxury developments and Wasalt properties.",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # Attach Rate Limiter state
    app.state.limiter = limiter

    # Exception Handlers
    app.add_exception_handler(RateLimitExceeded, rate_limit_error_handler)
    app.add_exception_handler(KnowledgeBaseException, domain_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    # Middlewares (Executed in reverse order of addition)
    # 1. Enforce payload size limits before body parsing
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.MAX_REQUEST_SIZE_BYTES)

    # 2. CORS allowlisting
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register API Routers
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(retrieve_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
