"""Extractors package."""

from backend.app.ingestion.extractors.base import BaseExtractor, AccessRestrictedError
from backend.app.ingestion.extractors.wasalt_extractor import WasaltExtractor
from backend.app.ingestion.extractors.darglobal_extractor import DarGlobalExtractor

__all__ = [
    "BaseExtractor",
    "AccessRestrictedError",
    "WasaltExtractor",
    "DarGlobalExtractor",
]
