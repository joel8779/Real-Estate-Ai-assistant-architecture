"""Tests for Application Configuration Parsing."""

from backend.app.config import Settings


def test_cors_origins_parsing():
    """Verify CORS origins string, list, and JSON parsing."""
    s1 = Settings(CORS_ORIGINS=["http://localhost:5173"])
    assert s1.CORS_ORIGINS == ["http://localhost:5173"]

    s2 = Settings(CORS_ORIGINS='["http://localhost:3000", "http://localhost:5173"]')
    assert s2.CORS_ORIGINS == ["http://localhost:3000", "http://localhost:5173"]

    s3 = Settings(CORS_ORIGINS="http://localhost:3000,http://localhost:5173")
    assert s3.CORS_ORIGINS == ["http://localhost:3000", "http://localhost:5173"]


def test_fallback_models_parsing():
    """Verify fallback models list and string parsing."""
    s = Settings(OPENROUTER_FALLBACK_MODELS="model-a,model-b")
    assert s.OPENROUTER_FALLBACK_MODELS == ["model-a", "model-b"]
