"""Response Validation & Grounding Engine.

Validates LLM-generated answers against retrieved evidence, verifies citations,
extracts structured property cards, and checks numerical claims for hallucination.
"""

import re
from typing import List, Optional, Set, Tuple
from backend.app.rag.retriever import RetrievedItem
from backend.app.schemas.chat import Citation, PropertyCardData


class ResponseValidator:
    """Validates synthesized responses against retrieved source context."""

    @staticmethod
    def build_citations(items: List[RetrievedItem], max_citations: int = 5) -> List[Citation]:
        """Create structured citations strictly from retrieved evidence chunks.

        Ensures deduplication by canonical URL and preserves provenance metadata.
        """
        citations: List[Citation] = []
        seen_urls: Set[str] = set()

        for item in items:
            if item.url in seen_urls:
                continue
            seen_urls.add(item.url)

            # Generate a clean concise snippet from the chunk text
            clean_lines = [
                line.strip()
                for line in item.text_content.split("\n")
                if line.strip() and not line.startswith("[Title:")
            ]
            snippet = clean_lines[0] if clean_lines else item.title
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."

            source_val = "darglobal" if item.source.lower() == "darglobal" else "wasalt"

            citations.append(
                Citation(
                    title=item.title,
                    url=item.url,
                    source=source_val,
                    category=item.category,
                    snippet=snippet,
                    provenance=item.provenance,
                )
            )

            if len(citations) >= max_citations:
                break

        return citations

    @staticmethod
    def build_property_cards(items: List[RetrievedItem]) -> List[PropertyCardData]:
        """Extract structured property cards from retrieved real estate chunks."""
        cards: List[PropertyCardData] = []
        seen_urls: Set[str] = set()

        for item in items:
            meta = item.metadata or {}
            # Only generate cards if entity is a project or listing
            if item.category not in ("project", "listing") or item.url in seen_urls:
                continue

            title = item.title
            price = meta.get("price")
            currency = meta.get("price_currency", "SAR")
            price_str = f"{price:,.0f} {currency}" if price else None

            bedrooms = meta.get("bedrooms")
            bed_str = f"{bedrooms} Bedrooms" if bedrooms else None

            location_parts = filter(None, [meta.get("district"), meta.get("city"), meta.get("country")])
            location_str = ", ".join(location_parts) or "Prime Location"

            cards.append(
                PropertyCardData(
                    title=title,
                    developer=meta.get("developer") or ("DarGlobal" if item.source == "darglobal" else None),
                    location=location_str,
                    property_type=meta.get("property_type") or ("Luxury Project" if item.source == "darglobal" else "Residential"),
                    price=price_str,
                    bedrooms=bed_str,
                    amenities=[],
                    url=item.url,
                    image_url=None,
                )
            )
            seen_urls.add(item.url)

        return cards

    @staticmethod
    def validate_numerical_claims(
        answer: str,
        evidence_items: List[RetrievedItem],
    ) -> Tuple[bool, List[str]]:
        """Verify that specific numeric claims (prices, areas, bedroom counts) exist in evidence.

        Returns:
            Tuple[is_valid, list_of_unsupported_numbers]
        """
        if not evidence_items:
            return True, []

        # Combine all evidence text
        combined_evidence = " ".join([item.text_content for item in evidence_items])

        # Extract large numbers or currency figures (e.g., 850,000, 1,600,000, 2,000,000)
        # and standalone numbers >= 10
        pattern = r"\b\d{1,3}(?:,\d{3})+\b|\b\d{2,}\b"
        answer_numbers = set(re.findall(pattern, answer))

        unsupported = []
        for num in answer_numbers:
            # Normalize comma formatting for comparison
            normalized_num = num.replace(",", "")
            # Skip common non-property figures like years (2024, 2025, 2026, 2017)
            if normalized_num in ("2017", "2023", "2024", "2025", "2026", "2030"):
                continue

            # Check if either formatted or unformatted number exists in evidence
            if num not in combined_evidence and normalized_num not in combined_evidence:
                unsupported.append(num)

        is_valid = len(unsupported) == 0
        return is_valid, unsupported
