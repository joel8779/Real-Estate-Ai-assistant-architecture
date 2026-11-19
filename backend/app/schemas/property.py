"""Property and Listing Query Schemas."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.document import NormalizedEntity


class PropertyFilter(BaseModel):
    """Filters for structured property search."""

    source: Optional[Literal["darglobal", "wasalt"]] = None
    country: Optional[str] = None
    city: Optional[str] = None
    property_type: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_bedrooms: Optional[int] = None
    limit: int = Field(default=10, ge=1, le=50)
    offset: int = Field(default=0, ge=0)


class PropertyListResponse(BaseModel):
    """Paginated collection of normalized properties."""

    total: int
    items: List[NormalizedEntity]
    limit: int
    offset: int
