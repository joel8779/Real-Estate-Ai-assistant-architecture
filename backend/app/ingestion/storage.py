"""Storage Management and Deduplication Engine.

Saves raw crawl artifacts under data/raw/ and normalized knowledge entities under data/processed/.
Implements deduplication via canonical URL tracking and content hashing.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone

from backend.app.schemas.document import NormalizedEntity, RawDocument, IngestionReport
from backend.app.core.logging import logger


class StorageManager:
    """Manages raw and normalized dataset persistence and deduplication."""

    def __init__(self, raw_dir: Path = Path("data/raw"), processed_dir: Path = Path("data/processed")):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        # Ensure directories exist
        (self.raw_dir / "darglobal").mkdir(parents=True, exist_ok=True)
        (self.raw_dir / "wasalt").mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.processed_dir / "records.jsonl"
        self.json_path = self.processed_dir / "records.json"

        # In-memory index of seen entities for deduplication
        self._seen_urls: Set[str] = set()
        self._seen_hashes: Set[str] = set()
        self._load_existing_index()

    def _load_existing_index(self) -> None:
        """Load seen URLs and content hashes from existing processed records."""
        if not self.jsonl_path.exists():
            return

        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    url = data.get("url")
                    c_hash = data.get("content_hash")
                    if url:
                        self._seen_urls.add(url)
                    if c_hash:
                        self._seen_hashes.add(c_hash)
        except Exception as e:
            logger.warning(f"Error loading existing storage index: {e}")

    def save_raw(self, doc: RawDocument) -> Path:
        """Save raw document to disk under data/raw/<source>/<doc_id>.json."""
        target_dir = self.raw_dir / doc.source
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{doc.id}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc.model_dump_json(indent=2))

        return file_path

    def save_normalized(self, entities: List[NormalizedEntity]) -> Tuple[int, int]:
        """Save normalized records with URL and content hash deduplication.

        Returns:
            Tuple[saved_count, duplicate_count]
        """
        saved_count = 0
        duplicate_count = 0
        to_append: List[NormalizedEntity] = []

        for entity in entities:
            # Check duplicate by canonical URL and content hash
            if entity.url in self._seen_urls and entity.content_hash in self._seen_hashes:
                duplicate_count += 1
                logger.debug(f"Skipping duplicate record: {entity.url} (hash={entity.content_hash[:8]})")
                continue

            self._seen_urls.add(entity.url)
            self._seen_hashes.add(entity.content_hash)
            to_append.append(entity)
            saved_count += 1

        if to_append:
            # Append to records.jsonl
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                for item in to_append:
                    f.write(item.model_dump_json() + "\n")

            # Mirror to formatted JSON array
            self._sync_json_array()

        return saved_count, duplicate_count

    def _sync_json_array(self) -> None:
        """Synchronize records.jsonl into a formatted records.json array."""
        all_records = self.load_all_normalized()
        with open(self.json_path, "w", encoding="utf-8") as f:
            records_data = [r.model_dump(mode="json") for r in all_records]
            json.dump(records_data, f, indent=2, ensure_ascii=False)

    def load_all_normalized(self) -> List[NormalizedEntity]:
        """Read all normalized entities from disk."""
        records: List[NormalizedEntity] = []
        if not self.jsonl_path.exists():
            return records

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(NormalizedEntity.model_validate_json(line))
        return records

    def load_all_raw(self, source: Optional[str] = None) -> List[RawDocument]:
        """Read all raw documents from data/raw/."""
        raw_docs: List[RawDocument] = []
        sources = [source] if source else ["darglobal", "wasalt"]

        for s in sources:
            source_dir = self.raw_dir / s
            if not source_dir.exists():
                continue
            for file_path in source_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_docs.append(RawDocument.model_validate_json(f.read()))
                except Exception as e:
                    logger.warning(f"Error reading raw document {file_path}: {e}")

        return raw_docs

    def save_report(self, report: IngestionReport, custom_path: Optional[Path] = None) -> Path:
        """Save ingestion summary report in JSON and Markdown formats."""
        report_json_path = custom_path or (self.processed_dir / "ingestion_report.json")
        report_md_path = report_json_path.with_suffix(".md")

        # Save JSON
        with open(report_json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        # Generate Markdown
        md_lines = [
            f"# Ingestion Execution Report — {report.source.upper()}",
            f"- **Execution Time:** {report.started_at.isoformat()} to {report.completed_at.isoformat()} ({report.duration_seconds:.2f}s)",
            f"- **Live URLs Discovered:** {report.live_discovered}",
            f"- **Live URLs Successfully Processed:** {report.live_processed}",
            f"- **Live URLs Restricted by WAF/Policy:** {report.live_restricted}",
            f"- **Seed Documents Discovered:** {report.seed_discovered}",
            f"- **Seed Documents Processed:** {report.seed_processed}",
            f"- **Total Records Indexed:** {report.total_processed}",
            f"- **Duplicates Skipped:** {report.duplicates_detected}",
            "",
            "## Provenance Breakdown",
        ]
        for prov, count in report.provenance_breakdown.items():
            md_lines.append(f"- **{prov}:** {count}")

        md_lines.extend(["", "## Records by Category"])
        for cat, count in report.categories_breakdown.items():
            md_lines.append(f"- **{cat}:** {count}")

        if report.access_restrictions:
            md_lines.extend(["", "## Detected Access Restrictions"])
            for res in report.access_restrictions:
                md_lines.append(f"- **{res.get('url')}:** {res.get('reason')}")

        if report.failed_urls:
            md_lines.extend(["", "## Failed URLs"])
            for fail in report.failed_urls:
                md_lines.append(f"- **{fail.url}** ({fail.error_type}): {fail.reason}")

        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        return report_json_path
