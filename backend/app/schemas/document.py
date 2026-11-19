"""Document and Knowledge Entity Schemas.

Defines models for raw crawled documents, normalized knowledge entities,
embedded vector chunks, and ingestion execution reports with strict provenance tracking.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# Provenance classification tiers
ProvenanceType = Literal["verified_source", "recruiter_provided", "demonstration_unverified"]


class ContentCategory(str, Enum):
    """Categorical classification of crawled website content."""

    COMPANY_INFO = "company_info"
    PROJECT = "project"
    LISTING = "listing"
    LOCATION = "location"
    AMENITY = "amenity"
    INVESTMENT = "investment"
    NEWS = "news"
    FAQ = "faq"
    LEGAL = "legal"
    OTHER = "other"


class RawDocument(BaseModel):
    """Raw scraped web document representation."""

    id: str = Field(..., description="Unique hash of canonical URL")
    source: Literal["darglobal", "wasalt"]
    url: str
    title: str
    content_type: Literal["html", "json_ld", "next_data", "text"]
    raw_content: str
    content_hash: str = Field(default="", description="SHA-256 hash of raw payload")
    provenance: ProvenanceType = Field(
        default="demonstration_unverified",
        description="Data provenance tier (verified_source, recruiter_provided, demonstration_unverified)",
    )
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    http_status: int = 200
    headers: Dict[str, str] = Field(default_factory=dict)


class NormalizedEntity(BaseModel):
    """Standardized multi-source knowledge entity."""

    id: str = Field(..., description="Unique entity ID (deterministic from canonical URL)")
    source: Literal["darglobal", "wasalt"]
    category: ContentCategory
    url: str = Field(..., description="Canonical source URL for citation attribution")
    title: str
    headline: Optional[str] = None
    description: str = Field(..., description="Cleaned, normalized textual knowledge content")
    content_hash: str = Field(..., description="SHA-256 hash of normalized text for deduplication")
    provenance: ProvenanceType = Field(
        default="demonstration_unverified",
        description="Data provenance tier (verified_source, recruiter_provided, demonstration_unverified)",
    )

    # Real estate specific attributes (only populated when explicitly provided)
    developer: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    property_type: Optional[str] = None
    price: Optional[float] = None
    price_currency: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    area_sqm: Optional[float] = None
    amenities: List[str] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list)

    # General metadata and timestamps
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VectorChunk(BaseModel):
    """Chunk of textual content indexed in vector database."""

    chunk_id: str = Field(..., description="Unique deterministic chunk ID: <doc_id>_chunk_<idx>")
    entity_id: str = Field(..., description="Parent entity document ID")
    source: Literal["darglobal", "wasalt"]
    category: ContentCategory
    title: str
    url: str
    text_content: str
    provenance: ProvenanceType = Field(
        default="demonstration_unverified",
        description="Data provenance tier carried over from parent document",
    )
    country: Optional[str] = None
    city: Optional[str] = None
    property_type: Optional[str] = None
    price: Optional[float] = None
    price_currency: Optional[str] = None
    bedrooms: Optional[int] = None
    area_sqm: Optional[float] = None


class IngestionFailure(BaseModel):
    """Record of a failed URL fetch or extraction."""

    url: str
    reason: str
    status_code: Optional[int] = None
    error_type: str = "FETCH_OR_PARSE_ERROR"


class IngestionReport(BaseModel):
    """Summary report generated at completion of an ingestion run."""

    source: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    live_discovered: int = 0
    live_processed: int = 0
    live_restricted: int = 0
    seed_discovered: int = 0
    seed_processed: int = 0
    total_processed: int = 0
    duplicates_detected: int = 0
    provenance_breakdown: Dict[str, int] = Field(default_factory=dict)
    categories_breakdown: Dict[str, int] = Field(default_factory=dict)
    failed_urls: List[IngestionFailure] = Field(default_factory=list)
    access_restrictions: List[Dict[str, str]] = Field(default_factory=list)
