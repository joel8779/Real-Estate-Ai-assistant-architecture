"""Embedding Generation Engine using Sentence Transformers and HuggingFace PyTorch.

Loads sentence-transformers/all-MiniLM-L6-v2 once as a thread-safe singleton,
producing 384-dimensional normalized embeddings for local semantic search.
"""

import threading
from typing import List, Optional
import torch
import torch.nn.functional as F

from backend.app.config import get_settings
from backend.app.core.logging import logger


class Embedder:
    """Thread-safe, cached local embedding engine for all-MiniLM-L6-v2."""

    _instance: Optional["Embedder"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to extract sentence representation from token embeddings."""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _load_model(self) -> None:
        """Load tokenizer and model weights into memory once."""
        try:
            from transformers import AutoTokenizer, AutoModel

            # Ensure model name is fully qualified for HuggingFace if short name provided
            hf_model_id = self.model_name
            if not hf_model_id.startswith("sentence-transformers/") and "/" not in hf_model_id:
                hf_model_id = f"sentence-transformers/{hf_model_id}"

            logger.info(f"Loading local embedding model: '{hf_model_id}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            self.model = AutoModel.from_pretrained(hf_model_id)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"Embedding model '{hf_model_id}' successfully loaded and ready.")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
            raise RuntimeError(
                f"Could not load embedding model '{self.model_name}'. "
                f"Verify dependencies or network connectivity: {str(e)}"
            ) from e

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> "Embedder":
        """Thread-safe singleton getter."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name=model_name)
        return cls._instance

    def embed_query(self, query: str) -> List[float]:
        """Generate normalized 384-dimensional embedding for a search query."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Embedding model is not loaded.")

        cleaned = query.strip()
        inputs = self.tokenizer(
            [cleaned],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings[0].tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate normalized embeddings for a batch of documents."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Embedding model is not loaded.")
        if not documents:
            return []

        cleaned = [d.strip() for d in documents]
        inputs = self.tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()
