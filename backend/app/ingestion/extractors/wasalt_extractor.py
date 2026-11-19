"""Wasalt Data Extractor.

Extracts structured property listings, location guides, and platform information
from Wasalt HTML using Schema.org JSON-LD, Next.js hydration payloads, and cleaned HTML.
"""

import json
import re
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup

from backend.app.schemas.document import ContentCategory, NormalizedEntity, RawDocument
from backend.app.ingestion.extractors.base import BaseExtractor, AccessRestrictedError
from backend.app.ingestion.cleaner import clean_html_content, detect_access_restriction
from backend.app.ingestion.url_utils import canonicalize_url, compute_content_hash, generate_doc_id


class WasaltExtractor(BaseExtractor):
    """Extractor for Wasalt Saudi Arabia property portal."""

    def can_handle(self, source: str, url: str) -> bool:
        return source == "wasalt" or "wasalt.sa" in url

    def extract(self, raw_doc: RawDocument) -> List[NormalizedEntity]:
        # Check for access restrictions
        restriction = detect_access_restriction(
            raw_doc.raw_content, raw_doc.http_status, raw_doc.headers
        )
        if restriction:
            raise AccessRestrictedError(raw_doc.url, restriction)

        canonical_url = canonicalize_url(raw_doc.url)
        doc_id = generate_doc_id(canonical_url)

        # Determine category based on URL structure
        category = self._classify_category(canonical_url)

        # Attempt structured extraction
        json_ld_entities = self._extract_json_ld(raw_doc.raw_content)
        next_data = self._extract_next_data(raw_doc.raw_content)

        # Fallback to standard clean HTML content
        page_title, clean_text, meta_attrs = clean_html_content(raw_doc.raw_content)
        title = page_title or raw_doc.title or "Wasalt Property"

        # Property specific fields
        property_type: Optional[str] = None
        price: Optional[float] = None
        price_currency: Optional[str] = "SAR"
        bedrooms: Optional[int] = None
        bathrooms: Optional[int] = None
        area_sqm: Optional[float] = None
        country: Optional[str] = "Saudi Arabia"
        city: Optional[str] = None
        district: Optional[str] = None
        amenities: List[str] = []
        images: List[str] = []
        metadata: Dict[str, Any] = {}

        description = clean_text

        # 1. Enrich from Schema.org JSON-LD if available
        for block in json_ld_entities:
            b_type = block.get("@type", "")
            if b_type in ("SingleFamilyResidence", "Apartment", "House", "Product", "Place"):
                property_type = b_type
                if block.get("name"):
                    title = block["name"]

                if block.get("description"):
                    # Clean HTML inside description if present
                    desc_soup = BeautifulSoup(str(block["description"]), "html.parser")
                    desc_text = desc_soup.get_text(separator="\n", strip=True)
                    if desc_text:
                        description = desc_text

                # Bedrooms / Rooms
                if block.get("numberOfBedrooms"):
                    try:
                        bedrooms = int(block["numberOfBedrooms"])
                    except (ValueError, TypeError):
                        pass
                elif block.get("numberOfRooms"):
                    try:
                        bedrooms = int(block["numberOfRooms"])
                    except (ValueError, TypeError):
                        pass

                # Offers & Pricing
                offers = block.get("offers")
                if isinstance(offers, list) and len(offers) > 0:
                    offers = offers[0]
                if isinstance(offers, dict):
                    if offers.get("price"):
                        try:
                            price = float(offers["price"])
                        except (ValueError, TypeError):
                            pass
                    if offers.get("priceCurrency"):
                        price_currency = str(offers["priceCurrency"])

                # Address details
                addr = block.get("address")
                if isinstance(addr, dict):
                    country = addr.get("addressCountry") or country
                    city = addr.get("addressLocality")
                    district = addr.get("streetAddress") or addr.get("addressRegion")

                if block.get("image"):
                    img = block["image"]
                    if isinstance(img, str):
                        images.append(img)
                    elif isinstance(img, list):
                        images.extend([str(i) for i in img if isinstance(i, str)])

        # 2. Enrich from Next.js data if available
        if next_data:
            prop_details = next_data.get("props", {}).get("pageProps", {}).get("propertyDetailsV3", {})
            if prop_details:
                # Location coordinates
                loc = prop_details.get("location", {})
                if loc.get("lat") and loc.get("lon"):
                    metadata["latitude"] = loc["lat"]
                    metadata["longitude"] = loc["lon"]

                # Property info
                p_info = prop_details.get("propertyInfo", {})
                if p_info:
                    if p_info.get("address") and not district:
                        district = str(p_info["address"])
                    if p_info.get("propertyMainType") and not property_type:
                        property_type = str(p_info["propertyMainType"])

                # Attributes (carpetArea, bedrooms)
                attrs = prop_details.get("attributes", [])
                if isinstance(attrs, list):
                    for attr in attrs:
                        key = attr.get("key")
                        val = attr.get("value")
                        if key == "carpetArea" and val:
                            try:
                                area_sqm = float(val)
                            except (ValueError, TypeError):
                                pass
                        elif key == "noOfBedrooms" and val and bedrooms is None:
                            try:
                                bedrooms = int(val)
                            except (ValueError, TypeError):
                                pass

                # Additional specifications as amenities
                add_attrs = prop_details.get("additionalAttributes", [])
                if isinstance(add_attrs, list):
                    for item in add_attrs:
                        label = item.get("label")
                        val = item.get("value")
                        if label and val:
                            amenities.append(f"{label}: {val}")

        # Compute content hash for deduplication
        content_hash = compute_content_hash(description)

        entity = NormalizedEntity(
            id=doc_id,
            source="wasalt",
            category=category,
            url=canonical_url,
            title=title,
            headline=f"{property_type or 'Property'} in {city or district or 'Saudi Arabia'}",
            description=description,
            content_hash=content_hash,
            provenance=raw_doc.provenance,
            developer=None,  # Wasalt is a multi-developer/agency listing portal
            country=country,
            city=city,
            district=district,
            property_type=property_type,
            price=price,
            price_currency=price_currency if price else None,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            area_sqm=area_sqm,
            amenities=amenities,
            images=images,
            metadata=metadata,
        )

        return [entity]

    def _classify_category(self, url: str) -> ContentCategory:
        """Classify Wasalt content based on URL taxonomy."""
        lower_url = url.lower()
        if "/property/sale" in lower_url or "/property/rent" in lower_url:
            return ContentCategory.LISTING
        if "/properties-for-sale" in lower_url or "/properties-for-rent" in lower_url or "/villas-for-sale" in lower_url:
            return ContentCategory.LOCATION
        if "/s/wasalt-business" in lower_url or "/s/masterplan" in lower_url:
            return ContentCategory.COMPANY_INFO
        if "/s/privacy" in lower_url or "/s/terms" in lower_url:
            return ContentCategory.LEGAL
        if "/faq" in lower_url:
            return ContentCategory.FAQ
        return ContentCategory.OTHER

    def _extract_json_ld(self, html: str) -> List[Dict[str, Any]]:
        """Extract all JSON-LD application script blocks from HTML."""
        blocks: List[Dict[str, Any]] = []
        if not html:
            return blocks

        pattern = re.compile(
            r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            re.DOTALL | re.IGNORECASE,
        )
        matches = pattern.findall(html)
        for match in matches:
            try:
                data = json.loads(match.strip())
                if isinstance(data, dict):
                    blocks.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            blocks.append(item)
            except Exception:
                continue
        return blocks

    def _extract_next_data(self, html: str) -> Optional[Dict[str, Any]]:
        """Extract Next.js hydration __NEXT_DATA__ JSON script from HTML."""
        if not html:
            return None
        match = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            html,
            re.DOTALL,
        )
        if match:
            try:
                return json.loads(match.group(1).strip())
            except Exception:
                return None
        return None
