"""ChromaDB Persistent Vector Store Manager.

Provides incremental indexing, chunk deduplication, metadata filtering,
and cosine similarity search under data/chroma/.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.app.schemas.document import ContentCategory, VectorChunk
from backend.app.config import get_settings
from backend.app.core.logging import logger


class ChromaVectorStore:
    """Manages persistent ChromaDB vector storage and semantic retrieval."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.persist_dir = str(persist_dir or Path(settings.CHROMA_PERSIST_DIRECTORY))
        self.collection_name = collection_name or "darglobal_wasalt_knowledge"

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Connecting to persistent ChromaDB at '{self.persist_dir}'...")
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Ensure collection exists with cosine distance
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{self.collection_name}' ready with {self.collection.count()} chunks."
        )

    def count(self) -> int:
        """Get total number of indexed chunks in the collection."""
        return self.collection.count()

    def index_chunks(
        self,
        chunks: List[VectorChunk],
        embeddings: List[List[float]],
    ) -> int:
        """Incrementally index vector chunks with automatic removal of obsolete entity chunks.

        Returns:
            Number of newly indexed chunks.
        """
        if not chunks:
            return 0

        ids: List[str] = []
        embeddings_list: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        # Find distinct parent entities to clean up any obsolete chunks first
        distinct_entities = {chunk.entity_id for chunk in chunks}
        for entity_id in distinct_entities:
            self.delete_by_entity_id(entity_id)

        for chunk, emb in zip(chunks, embeddings):
            ids.append(chunk.chunk_id)
            embeddings_list.append(emb)
            documents.append(chunk.text_content)

            # Metadata must be strings, ints, floats, or bools for Chroma
            meta: Dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "entity_id": chunk.entity_id,
                "source": chunk.source,
                "category": chunk.category.value,
                "title": chunk.title,
                "url": chunk.url,
                "provenance": chunk.provenance,
            }
            if chunk.country:
                meta["country"] = chunk.country
            if chunk.city:
                meta["city"] = chunk.city
            if chunk.property_type:
                meta["property_type"] = chunk.property_type
            if chunk.price is not None:
                meta["price"] = float(chunk.price)
            if chunk.price_currency:
                meta["price_currency"] = chunk.price_currency
            if chunk.bedrooms is not None:
                meta["bedrooms"] = int(chunk.bedrooms)
            if chunk.area_sqm is not None:
                meta["area_sqm"] = float(chunk.area_sqm)

            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Successfully indexed {len(ids)} chunks in ChromaDB.")
        return len(ids)

    def delete_by_entity_id(self, entity_id: str) -> None:
        """Delete all chunks associated with a specific parent entity."""
        try:
            self.collection.delete(where={"entity_id": entity_id})
        except Exception:
            pass  # Silent if entity had no previous chunks

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        source_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform dense vector search with metadata filtering.

        Returns:
            List of dictionaries containing chunk details, cosine distance, and similarity.
        """
        if self.count() == 0:
            return []

        # Construct Chroma where filter
        filter_conditions = []
        if source_filter and source_filter.lower() != "all":
            filter_conditions.append({"source": {"$eq": source_filter.lower()}})
        if category_filter:
            filter_conditions.append({"category": {"$eq": category_filter.lower()}})

        where_clause = None
        if len(filter_conditions) == 1:
            where_clause = filter_conditions[0]
        elif len(filter_conditions) > 1:
            where_clause = {"$and": filter_conditions}

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, max(1, self.count())),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        output: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return output

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for i, chunk_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            doc_text = docs[i] if i < len(docs) else ""
            dist = distances[i] if i < len(distances) else 1.0
            # Cosine similarity: 1.0 - distance
            similarity = max(0.0, min(1.0, 1.0 - dist))

            output.append({
                "chunk_id": chunk_id,
                "entity_id": meta.get("entity_id", ""),
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "category": meta.get("category", ""),
                "url": meta.get("url", ""),
                "provenance": meta.get("provenance", "demonstration_unverified"),
                "text_content": doc_text,
                "cosine_distance": dist,
                "similarity": similarity,
                "metadata": meta,
            })

        return output

    def get_all_chunks(self) -> List[VectorChunk]:
        """Retrieve all indexed chunks from ChromaDB for BM25 synchronization."""
        if self.count() == 0:
            return []

        data = self.collection.get(include=["documents", "metadatas"])
        chunks: List[VectorChunk] = []

        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])

        for i, chunk_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            doc = docs[i] if i < len(docs) else ""

            cat_str = meta.get("category", "other")
            try:
                cat = ContentCategory(cat_str)
            except ValueError:
                cat = ContentCategory.OTHER

            chunks.append(
                VectorChunk(
                    chunk_id=chunk_id,
                    entity_id=meta.get("entity_id", ""),
                    source=meta.get("source", "darglobal"),
                    category=cat,
                    title=meta.get("title", ""),
                    url=meta.get("url", ""),
                    text_content=doc,
                    provenance=meta.get("provenance", "demonstration_unverified"),
                    country=meta.get("country"),
                    city=meta.get("city"),
                    property_type=meta.get("property_type"),
                    price=meta.get("price"),
                    price_currency=meta.get("price_currency"),
                    bedrooms=meta.get("bedrooms"),
                    area_sqm=meta.get("area_sqm"),
                )
            )

        return chunks
