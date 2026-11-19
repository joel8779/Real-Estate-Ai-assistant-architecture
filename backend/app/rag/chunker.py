"""Document Processing and Context-Preserving Semantic Chunker.

Splits normalized records into overlapping chunks while prepending structured
real estate specifications to prevent context fragmentation.
"""

import re
from typing import List, Optional
from backend.app.schemas.document import NormalizedEntity, VectorChunk


class DocumentChunker:
    """Chunks normalized entities into deterministic, context-rich vector chunks."""

    def __init__(self, target_chunk_size: int = 500, chunk_overlap: int = 100):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def build_context_header(self, entity: NormalizedEntity) -> str:
        """Construct a contextual metadata header prepended to every chunk.

        Ensures property specifications (price, location, bedrooms, developer)
        remain attached to text even when long descriptions are split.
        """
        parts = [f"Title: {entity.title}", f"Source: {entity.source.upper()}", f"Category: {entity.category.value}"]

        if entity.developer:
            parts.append(f"Developer: {entity.developer}")
        if entity.property_type:
            parts.append(f"Type: {entity.property_type}")
        if entity.country or entity.city:
            loc = ", ".join(filter(None, [entity.district, entity.city, entity.country]))
            parts.append(f"Location: {loc}")
        if entity.price:
            parts.append(f"Price: {entity.price:,.0f} {entity.price_currency or 'SAR'}")
        if entity.bedrooms:
            parts.append(f"Bedrooms: {entity.bedrooms}")
        if entity.area_sqm:
            parts.append(f"Area: {entity.area_sqm} SQM")
        if entity.provenance:
            parts.append(f"Provenance: {entity.provenance}")

        return "[" + " | ".join(parts) + "]\n\n"

    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping passages using paragraph and sentence boundaries."""
        if not text or not text.strip():
            return []

        # Split on paragraph breaks first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        chunks: List[str] = []
        current_chunk = ""

        for para in paragraphs:
            # If paragraph itself is longer than chunk size, split by sentences
            if len(para) > self.target_chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) > self.target_chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        # Overlap with end of current chunk
                        words = current_chunk.split()
                        overlap_words = words[-max(1, self.chunk_overlap // 10):]
                        current_chunk = " ".join(overlap_words) + " " + sent
                    else:
                        current_chunk = (current_chunk + " " + sent).strip()
            else:
                if len(current_chunk) + len(para) > self.target_chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    words = current_chunk.split()
                    overlap_words = words[-max(1, self.chunk_overlap // 10):]
                    current_chunk = " ".join(overlap_words) + "\n\n" + para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text.strip()]

    def chunk_entity(self, entity: NormalizedEntity) -> List[VectorChunk]:
        """Convert a NormalizedEntity into a list of VectorChunks.

        Short documents are kept intact. Long descriptions are split into
        context-prepended chunks with deterministic IDs.
        """
        raw_text = entity.description.strip()
        header = self.build_context_header(entity)

        # If document description is short enough, keep it intact in a single chunk
        if len(raw_text) <= self.target_chunk_size:
            text_passages = [raw_text]
        else:
            text_passages = self.split_text(raw_text)

        chunks: List[VectorChunk] = []
        for idx, passage in enumerate(text_passages):
            chunk_id = f"{entity.id}_chunk_{idx}"
            full_chunk_text = header + passage

            chunks.append(
                VectorChunk(
                    chunk_id=chunk_id,
                    entity_id=entity.id,
                    source=entity.source,
                    category=entity.category,
                    title=entity.title,
                    url=entity.url,
                    text_content=full_chunk_text,
                    provenance=entity.provenance,
                    country=entity.country,
                    city=entity.city,
                    property_type=entity.property_type,
                    price=entity.price,
                    price_currency=entity.price_currency,
                    bedrooms=entity.bedrooms,
                    area_sqm=entity.area_sqm,
                )
            )

        return chunks

    def chunk_documents(self, entities: List[NormalizedEntity]) -> List[VectorChunk]:
        """Convert a list of NormalizedEntity objects into VectorChunks."""
        all_chunks: List[VectorChunk] = []
        for entity in entities:
            all_chunks.extend(self.chunk_entity(entity))
        return all_chunks
