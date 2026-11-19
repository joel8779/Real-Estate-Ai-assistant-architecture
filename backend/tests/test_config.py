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


def test_ollama_config_defaults_and_overrides():
    """Verify Ollama settings defaults and custom configuration."""
    s_default = Settings()
    assert s_default.OLLAMA_ENABLED is True
    assert s_default.OLLAMA_BASE_URL == "http://localhost:11434"
    assert s_default.OLLAMA_MODEL == "qwen2.5:1.5b"
    assert s_default.OLLAMA_TIMEOUT_SECONDS == 60.0
    assert s_default.OLLAMA_KEEP_ALIVE == "5m"
    assert s_default.OPENROUTER_MODEL == "nvidia/nemotron-3-ultra-550b-a55b:free"

    s_custom = Settings(
        OLLAMA_ENABLED=False,
        OLLAMA_BASE_URL="http://ollama:11434",
        OLLAMA_MODEL="qwen2.5:1.5b",
        OLLAMA_TIMEOUT_SECONDS=30.0,
        OLLAMA_KEEP_ALIVE="10m",
    )
    assert s_custom.OLLAMA_ENABLED is False
    assert s_custom.OLLAMA_BASE_URL == "http://ollama:11434"
    assert s_custom.OLLAMA_MODEL == "qwen2.5:1.5b"
    assert s_custom.OLLAMA_TIMEOUT_SECONDS == 30.0
    assert s_custom.OLLAMA_KEEP_ALIVE == "10m"
