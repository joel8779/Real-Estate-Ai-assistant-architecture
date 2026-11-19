"""URL Canonicalization, Hashing, and Domain Verification Utilities."""

import hashlib
import re
from typing import List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


# Common marketing/tracking query parameters to strip
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "sfmc_activityid",
    "sfmc_journey_id",
    "sfmc_journey_name",
    "dealercode",
    "ltclid",
    "fbclid",
    "gclid",
    "gclsrc",
    "dclid",
    "zanpid",
    "trk",
    "ref",
}


def canonicalize_url(url: str, allowed_domains: Optional[List[str]] = None) -> str:
    """Transform a URL into its deterministic canonical representation.

    - Ensures http/https scheme.
    - Converts scheme and hostname to lowercase.
    - Strips fragment identifiers (#...).
    - Filters out marketing, tracking, and analytics query parameters.
    - Normalizes trailing slashes (preserves single root slash).
    - Validates against an allowed domain list if provided.
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string.")

    url = url.strip()
    split = urlsplit(url)

    if split.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme '{split.scheme}'. Only http and https allowed.")

    netloc = split.netloc.lower()
    # Strip default ports if present
    if netloc.endswith(":80") and split.scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and split.scheme == "https":
        netloc = netloc[:-4]

    # Domain allowlist check
    if allowed_domains:
        domain_matched = False
        for allowed in allowed_domains:
            if netloc == allowed or netloc.endswith("." + allowed):
                domain_matched = True
                break
        if not domain_matched:
            raise ValueError(f"Domain '{netloc}' is not in the allowed domain list: {allowed_domains}")

    # Path normalization
    path = split.path or "/"
    # Clean redundant slashes
    path = re.sub(r"/+", "/", path)
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    # Query parameters normalization: strip tracking params, sort remaining
    filtered_query = []
    if split.query:
        query_pairs = parse_qsl(split.query, keep_blank_values=False)
        for key, value in query_pairs:
            if key.lower() not in TRACKING_PARAMS:
                filtered_query.append((key, value))
        filtered_query.sort()

    clean_query = urlencode(filtered_query)

    # Reconstruct canonical URL (fragment is completely dropped)
    return urlunsplit((split.scheme.lower(), netloc, path, clean_query, ""))


def generate_doc_id(canonical_url: str) -> str:
    """Generate a deterministic, collisions-resistant document ID from a canonical URL."""
    sha = hashlib.sha256(canonical_url.strip().encode("utf-8")).hexdigest()
    return f"doc_{sha[:16]}"


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of cleaned textual content for deduplication."""
    # Normalize internal whitespace before hashing
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
