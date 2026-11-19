"""DarGlobal Data Extractor.

Extracts luxury project profiles, company information, investment releases, and FAQs
from permitted DarGlobal HTML and verified offline records.
Enforces strict bot barrier detection without attempting stealth evasion.
"""

import re
from typing import Any, Dict, List, Optional

from backend.app.schemas.document import ContentCategory, NormalizedEntity, RawDocument
from backend.app.ingestion.extractors.base import BaseExtractor, AccessRestrictedError
from backend.app.ingestion.cleaner import clean_html_content, detect_access_restriction
from backend.app.ingestion.url_utils import canonicalize_url, compute_content_hash, generate_doc_id


# Recognizable luxury co-branding partners of DarGlobal
LUXURY_PARTNERS = [
    "Trump International",
    "Missoni",
    "Pagani",
    "Automobili Lamborghini",
    "Elie Saab",
    "Aston Martin",
    "W Hotels",
    "Mouawad",
    "Dolce & Gabbana",
]

# Primary geographic hubs
GEO_HUBS = {
    "dubai": ("United Arab Emirates", "Dubai"),
    "oman": ("Oman", "Muscat"),
    "muscat": ("Oman", "Muscat"),
    "qatar": ("Qatar", "Doha"),
    "doha": ("Qatar", "Doha"),
    "spain": ("Spain", "Costa del Sol"),
    "malaga": ("Spain", "Costa del Sol"),
    "benahavis": ("Spain", "Benahavís"),
    "london": ("United Kingdom", "London"),
    "riyadh": ("Saudi Arabia", "Riyadh"),
    "jeddah": ("Saudi Arabia", "Jeddah"),
}


class DarGlobalExtractor(BaseExtractor):
    """Extractor for DarGlobal luxury real estate developments."""

    def can_handle(self, source: str, url: str) -> bool:
        return source == "darglobal" or "darglobal.co.uk" in url

    def extract(self, raw_doc: RawDocument) -> List[NormalizedEntity]:
        # Detect bot barriers or access restrictions
        restriction = detect_access_restriction(
            raw_doc.raw_content, raw_doc.http_status, raw_doc.headers
        )
        if restriction:
            raise AccessRestrictedError(raw_doc.url, restriction)

        canonical_url = canonicalize_url(raw_doc.url)
        doc_id = generate_doc_id(canonical_url)

        category = self._classify_category(canonical_url)
        page_title, clean_text, meta_attrs = clean_html_content(raw_doc.raw_content)

        title = page_title or raw_doc.title or "DarGlobal Project"
        description = clean_text or meta_attrs.get("description", "")

        # Detect luxury co-branding partner
        brand_partner: Optional[str] = None
        combined_text = f"{title} {description}"
        for partner in LUXURY_PARTNERS:
            if partner.lower() in combined_text.lower():
                brand_partner = partner
                break

        # Detect location (country, city)
        country: Optional[str] = None
        city: Optional[str] = None
        lower_combined = combined_text.lower()
        for loc_key, (c_name, city_name) in GEO_HUBS.items():
            if re.search(rf"\b{loc_key}\b", lower_combined):
                country = c_name
                city = city_name
                break

        # Detect property type
        property_type: Optional[str] = None
        if "villa" in lower_combined or "villas" in lower_combined:
            property_type = "Luxury Villa"
        elif "penthouse" in lower_combined:
            property_type = "Penthouse"
        elif "residence" in lower_combined or "apartment" in lower_combined:
            property_type = "Branded Residence"

        metadata: Dict[str, Any] = {
            "developer": "DarGlobal",
            "heritage": "Dar Al Arkan",
            "stock_listing": "London Stock Exchange (LSE: DAR)",
        }
        if brand_partner:
            metadata["brand_partner"] = brand_partner

        content_hash = compute_content_hash(description)

        entity = NormalizedEntity(
            id=doc_id,
            source="darglobal",
            category=category,
            url=canonical_url,
            title=title,
            headline=f"DarGlobal {brand_partner or 'Luxury'} Development in {city or country or 'Global'}",
            description=description,
            content_hash=content_hash,
            provenance=raw_doc.provenance,
            developer="DarGlobal",
            country=country,
            city=city,
            property_type=property_type,
            price=None,  # Off-plan luxury developments list pricing upon consultation
            price_currency=None,
            bedrooms=None,
            bathrooms=None,
            area_sqm=None,
            amenities=[],
            images=[],
            metadata=metadata,
        )

        return [entity]

    def _classify_category(self, url: str) -> ContentCategory:
        """Classify DarGlobal content based on URL path."""
        lower = url.lower()
        if any(term in lower for term in ["/projects", "/aida", "/davinci", "/urban-oasis", "/tierra-viva", "/les-vagues"]):
            return ContentCategory.PROJECT
        if any(term in lower for term in ["/about", "/careers", "/development-management", "/contact"]):
            return ContentCategory.COMPANY_INFO
        if any(term in lower for term in ["/investor", "/insights", "/exclusive-membership", "/dg-circle"]):
            return ContentCategory.INVESTMENT
        if any(term in lower for term in ["/press", "/blog", "/news"]):
            return ContentCategory.NEWS
        if "/faq" in lower:
            return ContentCategory.FAQ
        return ContentCategory.OTHER
