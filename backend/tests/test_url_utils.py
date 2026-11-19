"""Unit Tests for URL Canonicalization, Hashing, and Domain Verification."""

import pytest
from backend.app.ingestion.url_utils import (
    canonicalize_url,
    generate_doc_id,
    compute_content_hash,
)


def test_canonicalize_url_basic():
    """Verify basic URL normalization and lowercase transformation."""
    raw = "HTTPS://Wasalt.SA/en/Property/Sale/Villa-123/"
    expected = "https://wasalt.sa/en/Property/Sale/Villa-123"
    assert canonicalize_url(raw) == expected


def test_canonicalize_url_strips_tracking_and_fragments():
    """Verify stripping of marketing tracking parameters and URL fragments."""
    raw = (
        "https://darglobal.co.uk/projects/aida/?utm_source=facebook&utm_medium=cpc"
        "&utm_campaign=launch&fbclid=IwAR123#overview"
    )
    expected = "https://darglobal.co.uk/projects/aida"
    assert canonicalize_url(raw) == expected


def test_canonicalize_url_preserves_legitimate_queries():
    """Verify legitimate non-tracking query parameters are preserved and sorted."""
    raw = "https://wasalt.sa/en/search?page=2&type=villa&utm_source=google"
    expected = "https://wasalt.sa/en/search?page=2&type=villa"
    assert canonicalize_url(raw) == expected


def test_canonicalize_url_domain_allowlist():
    """Verify that domain allowlisting permits valid hosts and rejects foreign ones."""
    allowed = ["darglobal.co.uk", "wasalt.sa"]

    # Allowed
    assert canonicalize_url("https://darglobal.co.uk/about", allowed_domains=allowed) == "https://darglobal.co.uk/about"
    assert canonicalize_url("https://wasalt.sa/en", allowed_domains=allowed) == "https://wasalt.sa/en"

    # Disallowed
    with pytest.raises(ValueError, match="not in the allowed domain list"):
        canonicalize_url("https://unauthorized-domain.com/hack", allowed_domains=allowed)


def test_generate_doc_id():
    """Verify deterministic document ID generation."""
    url1 = "https://wasalt.sa/en/property/1"
    url2 = "https://wasalt.sa/en/property/1"
    url3 = "https://wasalt.sa/en/property/2"

    id1 = generate_doc_id(url1)
    id2 = generate_doc_id(url2)
    id3 = generate_doc_id(url3)

    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("doc_")
    assert len(id1) == 20  # "doc_" (4) + 16 hex chars


def test_compute_content_hash():
    """Verify content hash normalizes whitespace and is deterministic."""
    t1 = "Villa for sale in Riyadh.\nLand area: 350 SQM."
    t2 = "  Villa  for   sale in Riyadh. \n\n Land area: 350 SQM.  "
    t3 = "Villa for sale in Jeddah.\nLand area: 400 SQM."

    h1 = compute_content_hash(t1)
    h2 = compute_content_hash(t2)
    h3 = compute_content_hash(t3)

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # SHA-256 hex string
