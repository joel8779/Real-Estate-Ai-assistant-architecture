"""Health and Readiness Probes.

Kubernetes and load-balancer compatible probes:
- /healthz: Liveness probe (process is running)
- /readyz: Readiness probe (dependencies, storage, and models ready)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, Response, status

from backend.app.config import get_settings

router = APIRouter(tags=["Health"])


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def liveness_probe() -> Dict[str, Any]:
    """Liveness probe to verify the application process is alive."""
    settings = get_settings()
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@router.get("/readyz")
async def readiness_probe(response: Response) -> Dict[str, Any]:
    """Readiness probe to verify application dependencies and storage volumes."""
    settings = get_settings()
    checks: Dict[str, Any] = {}
    is_ready = True

    # 1. Check persistent data storage paths
    storage_paths = [
        Path(settings.INGESTION_DATA_DIR),
        Path(settings.CHROMA_PERSIST_DIRECTORY),
    ]
    for p in storage_paths:
        try:
            p.mkdir(parents=True, exist_ok=True)
            checks[f"storage_{p.name}"] = "available"
        except Exception as e:
            checks[f"storage_{p.name}"] = f"error: {str(e)}"
            is_ready = False

    # 2. Check OpenRouter API configuration
    checks["openrouter_configured"] = bool(settings.OPENROUTER_API_KEY)
    checks["openrouter_model"] = settings.OPENROUTER_MODEL

    # 3. Check embedding configuration
    checks["embedding_model"] = settings.EMBEDDING_MODEL_NAME
    checks["similarity_threshold"] = settings.SIMILARITY_THRESHOLD

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unready",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        }

    return {
        "status": "ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }
