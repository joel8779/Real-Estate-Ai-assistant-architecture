"""Unit Tests for Wasalt and DarGlobal Extractors."""

from pathlib import Path
import pytest
from datetime import datetime, timezone

from backend.app.schemas.document import ContentCategory, RawDocument
from backend.app.ingestion.extractors import (
    WasaltExtractor,
    DarGlobalExtractor,
    AccessRestrictedError,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_wasalt_extractor_json_ld_and_next_data():
    """Verify Wasalt extractor extracts JSON-LD, Next.js props, and handles missing fields."""
    fixture_path = FIXTURES_DIR / "wasalt_sample.html"
    with open(fixture_path, "r", encoding="utf-8") as f:
        html = f.read()

    raw_doc = RawDocument(
        id="doc_wasalt_123",
        source="wasalt",
        url="https://wasalt.sa/en/property/sale/villa-al-narjis-5841999",
        title="Sample Villa",
        content_type="html",
        raw_content=html,
    )

    extractor = WasaltExtractor()
    assert extractor.can_handle("wasalt", raw_doc.url) is True

    entities = extractor.extract(raw_doc)
    assert len(entities) == 1
    entity = entities[0]

    # Verify extracted structured fields
    assert entity.source == "wasalt"
    assert entity.category == ContentCategory.LISTING
    assert entity.title == "Luxury Modern Villa in Al-Narjis, Riyadh"
    assert entity.price == 2800000.0
    assert entity.price_currency == "SAR"
    assert entity.bedrooms == 5
    assert entity.area_sqm == 350.0
    assert entity.country == "Saudi Arabia"
    assert entity.city == "Riyadh"
    assert entity.district == "Al-Narjis"

    # Verify Next.js coordinates and attributes
    assert entity.metadata.get("latitude") == "24.8214"
    assert any("Property Age: New / 2026" in a for a in entity.amenities)

    # Verify no fabricated data: bathrooms was not in source, must remain None
    assert entity.bathrooms is None

    # Verify content hash presence
    assert len(entity.content_hash) == 64


def test_darglobal_extractor_branded_project():
    """Verify DarGlobal extractor identifies brand partners, locations, and avoids fabricated prices."""
    fixture_path = FIXTURES_DIR / "darglobal_sample.html"
    with open(fixture_path, "r", encoding="utf-8") as f:
        html = f.read()

    raw_doc = RawDocument(
        id="doc_dg_123",
        source="darglobal",
        url="https://darglobal.co.uk/projects/aida",
        title="AIDA Oman",
        content_type="html",
        raw_content=html,
    )

    extractor = DarGlobalExtractor()
    assert extractor.can_handle("darglobal", raw_doc.url) is True

    entities = extractor.extract(raw_doc)
    assert len(entities) == 1
    entity = entities[0]

    assert entity.source == "darglobal"
    assert entity.category == ContentCategory.PROJECT
    assert entity.developer == "DarGlobal"
    assert entity.country == "Oman"
    assert entity.city == "Muscat"
    assert entity.metadata.get("brand_partner") == "Trump International"
    assert entity.property_type == "Luxury Villa"

    # Price was not specified in the marketing blurb, must NOT be fabricated
    assert entity.price is None
    assert entity.price_currency is None
    assert len(entity.content_hash) == 64


def test_darglobal_extractor_raises_access_restriction_on_waf():
    """Verify extractor detects Imperva challenge and raises AccessRestrictedError."""
    fixture_path = FIXTURES_DIR / "incapsula_challenge.html"
    with open(fixture_path, "r", encoding="utf-8") as f:
        html = f.read()

    raw_doc = RawDocument(
        id="doc_dg_blocked",
        source="darglobal",
        url="https://darglobal.co.uk/projects/urban-oasis",
        title="Urban Oasis",
        content_type="html",
        raw_content=html,
    )

    extractor = DarGlobalExtractor()
    with pytest.raises(AccessRestrictedError) as exc_info:
        extractor.extract(raw_doc)

    assert "Imperva/Incapsula" in exc_info.value.reason
