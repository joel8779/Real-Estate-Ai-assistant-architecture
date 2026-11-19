"""Base Extractor Interface."""

from abc import ABC, abstractmethod
from typing import List
from backend.app.schemas.document import RawDocument, NormalizedEntity


class AccessRestrictedError(Exception):
    """Raised when website access controls, WAF challenges, or CAPTCHAs are detected."""

    def __init__(self, url: str, reason: str):
        super().__init__(f"Access restricted on {url}: {reason}")
        self.url = url
        self.reason = reason


class BaseExtractor(ABC):
    """Abstract base class for source-specific content extractors."""

    @abstractmethod
    def can_handle(self, source: str, url: str) -> bool:
        """Check if this extractor handles the given source and URL."""
        pass

    @abstractmethod
    def extract(self, raw_doc: RawDocument) -> List[NormalizedEntity]:
        """Extract one or more normalized knowledge entities from a raw document.

        Raises:
            AccessRestrictedError: if bot challenges or access controls are detected.
        """
        pass
