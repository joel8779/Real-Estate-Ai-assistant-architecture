"""Chat and Conversation Data Schemas.

Enforces strict client message roles ('user' and 'assistant' only),
ensuring system instructions remain strictly backend-controlled.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


from backend.app.schemas.document import ProvenanceType


class ChatMessage(BaseModel):
    """Client conversation message.

    Accepts ONLY 'user' and 'assistant' roles. System instructions must never
    be supplied or influenced by client payloads.
    """

    role: Literal["user", "assistant"] = Field(
        ...,
        description="Message author role. Strictly restricted to 'user' and 'assistant'.",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Message textual content.",
    )

    @field_validator("content")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Message content cannot be blank.")
        return trimmed


class Citation(BaseModel):
    """Grounded source citation reference."""

    title: str = Field(..., description="Source article or property title")
    url: str = Field(..., description="Canonical source URL")
    source: Literal["darglobal", "wasalt"] = Field(..., description="Publishing platform")
    category: str = Field(..., description="Content category (e.g., project, company_info)")
    snippet: str = Field(..., description="Contextual text snippet supporting the answer")
    provenance: ProvenanceType = Field(
        default="demonstration_unverified",
        description="Data provenance tier (verified_source, recruiter_provided, demonstration_unverified)",
    )


class PropertyCardData(BaseModel):
    """Structured property card metadata for rich UI presentation."""

    title: str
    developer: Optional[str] = None
    location: str
    property_type: Optional[str] = None
    price: Optional[str] = None
    bedrooms: Optional[str] = None
    amenities: List[str] = Field(default_factory=list)
    url: str
    image_url: Optional[str] = None


class ChatRequest(BaseModel):
    """Inbound chat conversation request."""

    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Chronological conversation turn history (max 20 turns).",
    )
    source_filter: Literal["all", "darglobal", "wasalt"] = Field(
        default="all",
        description="Filter retrieval context by specific domain source.",
    )
    category_filter: Optional[str] = Field(
        default=None,
        description="Optional content category filter (e.g., 'project', 'company_info').",
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional client session identifier for request tracing.",
    )


class ChatResponse(BaseModel):
    """Outbound chat response with grounding metadata and citations."""

    answer: str = Field(..., description="Synthesized, grounded answer text")
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of verified citations supporting the answer",
    )
    property_cards: List[PropertyCardData] = Field(
        default_factory=list,
        description="Relevant structured property cards if query relates to listings/projects",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Retrieval confidence score based on similarity calibration",
    )
    grounded: bool = Field(
        default=True,
        description="Whether the answer was fully grounded in retrieved source context",
    )
    data_mode: Optional[Literal["verified", "demonstration", "conversational"]] = Field(
        default="demonstration",
        description="Operational data mode for the response ('verified', 'demonstration', or 'conversational')",
    )
    provenance_warning: Optional[str] = Field(
        default=None,
        description="Visible warning when answer is generated from prototype demonstration records",
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable error code if generation failed or LLM not configured",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request tracing identifier",
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Total end-to-end response processing latency in milliseconds",
    )
    stage_latencies: Optional[dict] = Field(
        default=None,
        description="Granular stage-by-stage latency measurements in milliseconds",
    )
    retry_after: Optional[int] = Field(
        default=None,
        description="Optional retry cooldown duration in seconds if provider is rate-limited",
    )
    model: Optional[str] = Field(
        default=None,
        description="Identifier of the LLM model that generated this response (primary or fallback)",
    )
    disclaimer: str = Field(
        default="Information provided is grounded in available records from DarGlobal and Wasalt. Verify terms, availability, and prices directly with official representatives.",
        description="Standard legal disclaimer",
    )

