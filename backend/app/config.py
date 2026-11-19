"""Application Configuration Module.

Loads configuration from environment variables and .env files using Pydantic Settings.
"""

from functools import lru_cache
import json
import re
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core application settings with environment variable bindings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Information
    APP_NAME: str = "DarGlobal Knowledge Assistant"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security & Network Controls
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5178",
        "http://127.0.0.1:5178",
    ]
    MAX_REQUEST_SIZE_BYTES: int = Field(default=65536, description="Max 64 KB")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Max requests/min per IP")

    # OpenRouter LLM Configuration
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API Key")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    OPENROUTER_ALLOW_FALLBACK: bool = Field(
        default=False,
        description="Disable automatic fallback by default to prevent silent model switching.",
    )
    OPENROUTER_FALLBACK_MODELS: Union[List[str], str] = [
        "nvidia/nemotron-3.5-lightning:free",
        "google/gemma-4-26b-a4b-it:free",
    ]
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    LLM_TIMEOUT_SECONDS: float = 45.0

    # Vector Database & Embeddings
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"
    CHROMA_COLLECTION_NAME: str = "darglobal_wasalt_knowledge"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RETRIEVAL_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = Field(
        default=0.45,
        description="Cosine similarity cutoff (0.0 to 1.0). Configurable & calibrated via evaluation.",
    )

    # Provenance & Demonstration Safeguards
    ALLOW_UNVERIFIED_DEMO_DATA: bool = Field(
        default=False,
        description="Explicit toggle to allow demonstration_unverified records. Defaults to False for strict verified mode.",
    )

    # Source Ingestion Toggles
    ENABLE_DARGLOBAL_INGESTION: bool = True
    ENABLE_WASALT_INGESTION: bool = True
    INGESTION_DATA_DIR: str = "./data"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS_ORIGINS from comma-separated string or JSON list."""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return ["*"]

    @field_validator("OPENROUTER_FALLBACK_MODELS", mode="before")
    @classmethod
    def parse_fallback_models(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse fallback models from comma-separated string or JSON list."""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [m.strip() for m in re.split(r"[,;]", v) if m.strip()]
        return []


@lru_cache()
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    return Settings()
