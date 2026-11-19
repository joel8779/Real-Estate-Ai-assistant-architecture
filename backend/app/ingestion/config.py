"""Ingestion Pipeline Configuration.

Defines rate limiting, retry parameters, domain allowlists, and directory paths.
"""

from pathlib import Path
from typing import List
from pydantic import BaseModel, Field


class IngestionSettings(BaseModel):
    """Configuration settings for website crawling and normalization."""

    # Domain allowlist - strictly rejects crawling any host outside this list
    allowed_domains: List[str] = Field(
        default=["darglobal.co.uk", "wasalt.sa", "cdn.wasalt.sa"],
        description="Explicitly permitted domain allowlist",
    )

    # Politeness & Rate Limits
    request_delay_seconds: float = Field(
        default=0.5,
        ge=0.0,
        description="Delay in seconds between successive requests to the same domain",
    )
    max_concurrency: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum concurrent HTTP requests to prevent server strain",
    )
    timeout_seconds: float = Field(
        default=15.0,
        ge=2.0,
        description="HTTP request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Maximum retry attempts on transient network or 5xx server errors",
    )
    backoff_factor: float = Field(
        default=1.5,
        description="Exponential backoff multiplier between retry attempts",
    )

    # Crawling Scope Bounds
    max_pages_per_source: int = Field(
        default=25,
        ge=1,
        le=500,
        description="Upper bound on pages scraped per source run to prevent bulk overloading",
    )
    user_agent: str = (
        "DarGlobalWasaltAssistant/1.0 (+http://localhost/compliance; polite-index-bot)"
    )

    # Filesystem Paths
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    seed_data_dir: Path = Path("data/seed_data")


# Default ingestion settings instance
default_ingestion_settings = IngestionSettings()
