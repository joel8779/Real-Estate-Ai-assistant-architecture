"""Ingestion and Normalization Package."""

from backend.app.ingestion.config import IngestionSettings, default_ingestion_settings
from backend.app.ingestion.url_utils import canonicalize_url, compute_content_hash, generate_doc_id
from backend.app.ingestion.cleaner import clean_html_content, detect_access_restriction
from backend.app.ingestion.storage import StorageManager
from backend.app.ingestion.crawler import Crawler

__all__ = [
    "IngestionSettings",
    "default_ingestion_settings",
    "canonicalize_url",
    "compute_content_hash",
    "generate_doc_id",
    "clean_html_content",
    "detect_access_restriction",
    "StorageManager",
    "Crawler",
]
