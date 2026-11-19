"""Unit Tests for Storage Manager and Deduplication."""

import shutil
from pathlib import Path
import pytest

from backend.app.schemas.document import ContentCategory, NormalizedEntity, RawDocument
from backend.app.ingestion.storage import StorageManager
from backend.app.ingestion.url_utils import compute_content_hash


@pytest.fixture
def temp_storage(tmp_path: Path):
    """Fixture providing isolated temporary storage directories."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    storage = StorageManager(raw_dir=raw_dir, processed_dir=processed_dir)
    yield storage
    # Clean up
    if tmp_path.exists():
        shutil.rmtree(tmp_path)


def test_save_and_load_raw(temp_storage: StorageManager):
    """Verify raw documents are written and loaded accurately."""
    doc = RawDocument(
        id="doc_raw_001",
        source="wasalt",
        url="https://wasalt.sa/en/sample",
        title="Sample Page",
        content_type="html",
        raw_content="<html><body>Sample</body></html>",
    )
    path = temp_storage.save_raw(doc)
    assert path.exists()

    loaded = temp_storage.load_all_raw(source="wasalt")
    assert len(loaded) == 1
    assert loaded[0].id == "doc_raw_001"
    assert loaded[0].url == "https://wasalt.sa/en/sample"


def test_save_and_deduplicate_normalized(temp_storage: StorageManager):
    """Verify normalized records deduplication by canonical URL and content hash."""
    desc = "Exclusive waterfront residence in Dubai Canal."
    c_hash = compute_content_hash(desc)

    entity1 = NormalizedEntity(
        id="doc_norm_001",
        source="darglobal",
        category=ContentCategory.PROJECT,
        url="https://darglobal.co.uk/projects/urban-oasis",
        title="Urban Oasis",
        description=desc,
        content_hash=c_hash,
    )

    # First save: 1 saved, 0 duplicates
    saved, dups = temp_storage.save_normalized([entity1])
    assert saved == 1
    assert dups == 0

    # Second save with exact same URL and content: 0 saved, 1 duplicate
    saved2, dups2 = temp_storage.save_normalized([entity1])
    assert saved2 == 0
    assert dups2 == 1

    # Verify persistent records count is still 1
    records = temp_storage.load_all_normalized()
    assert len(records) == 1


def test_persistence_across_storage_instances(temp_storage: StorageManager):
    """Verify deduplication index and records persist when a new StorageManager is instantiated."""
    desc = "AIDA cliffside community overlooking Gulf of Oman."
    c_hash = compute_content_hash(desc)

    entity = NormalizedEntity(
        id="doc_norm_aida",
        source="darglobal",
        category=ContentCategory.PROJECT,
        url="https://darglobal.co.uk/projects/aida",
        title="AIDA Oman",
        description=desc,
        content_hash=c_hash,
    )
    temp_storage.save_normalized([entity])

    # Instantiate brand new StorageManager pointing to same directories
    reloaded_storage = StorageManager(
        raw_dir=temp_storage.raw_dir,
        processed_dir=temp_storage.processed_dir,
    )

    # Verify reloaded index detects duplicate
    saved, dups = reloaded_storage.save_normalized([entity])
    assert saved == 0
    assert dups == 1

    records = reloaded_storage.load_all_normalized()
    assert len(records) == 1
    assert records[0].title == "AIDA Oman"
