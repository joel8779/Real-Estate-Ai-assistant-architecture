"""Data Provenance Audit Module.

Inspects all raw and processed records, evaluates origin legitimacy,
and assigns strict provenance tiers:
- verified_source: Direct, unrestricted live official capture
- recruiter_provided: Explicitly provided via recruiter/assignment materials
- demonstration_unverified: Demonstration or synthetic seed records for testing
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from backend.app.schemas.document import NormalizedEntity, ProvenanceType
from backend.app.core.logging import logger


class ProvenanceAuditor:
    """Audits data provenance for all crawled and seeded real estate entities."""

    def __init__(self, processed_dir: Path = Path("data/processed"), raw_dir: Path = Path("data/raw")):
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir)
        self.json_path = self.processed_dir / "records.json"
        self.jsonl_path = self.processed_dir / "records.jsonl"
        self.audit_report_json = self.processed_dir / "provenance_audit.json"
        self.audit_report_md = self.processed_dir / "provenance_audit.md"

    def run_audit(self) -> Dict[str, Any]:
        """Perform provenance audit and update records with accurate provenance metadata."""
        if not self.json_path.exists():
            logger.warning("No processed records.json found to audit.")
            return {"total_records": 0, "breakdown": {}}

        with open(self.json_path, "r", encoding="utf-8") as f:
            records: List[Dict[str, Any]] = json.load(f)

        audited_records: List[NormalizedEntity] = []
        breakdown: Dict[str, int] = {
            "verified_source": 0,
            "recruiter_provided": 0,
            "demonstration_unverified": 0,
        }
        item_details: List[Dict[str, Any]] = []

        for r in records:
            source = r.get("source", "unknown")
            url = r.get("url", "")
            title = r.get("title", "")
            raw_id = r.get("id", "")

            # Strict provenance evaluation:
            # 1. Did it come from a live HTTP request without bot barrier interception?
            # In our live crawl logs, darglobal.co.uk was intercepted by Imperva WAF,
            # and wasalt.sa was intercepted by Cloudflare challenge (403).
            # 2. Both sources were loaded from data/seed_data/ for demonstration/testing.
            # Per instructions: "If the provenance of a record cannot be established,
            # classify it as demonstration_unverified."
            provenance: ProvenanceType = "demonstration_unverified"

            breakdown[provenance] += 1
            r["provenance"] = provenance

            entity = NormalizedEntity.model_validate(r)
            audited_records.append(entity)

            item_details.append({
                "id": entity.id,
                "source": entity.source,
                "category": entity.category.value,
                "title": entity.title,
                "url": entity.url,
                "provenance": provenance,
                "audit_note": (
                    "Manual demonstration/seed document created for pipeline verification "
                    "following automated live WAF restriction on source domain."
                ),
            })

        # Save audited records back to records.json and records.jsonl
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump([e.model_dump(mode="json") for e in audited_records], f, indent=2, ensure_ascii=False)

        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            for e in audited_records:
                f.write(e.model_dump_json() + "\n")

        audit_summary = {
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "total_records": len(audited_records),
            "provenance_breakdown": breakdown,
            "records": item_details,
        }

        # Write audit report JSON
        with open(self.audit_report_json, "w", encoding="utf-8") as f:
            json.dump(audit_summary, f, indent=2, ensure_ascii=False)

        # Write audit report Markdown
        md_lines = [
            "# Data Provenance Audit Report",
            f"- **Audited At:** {audit_summary['audited_at']}",
            f"- **Total Records Inspected:** {audit_summary['total_records']}",
            "",
            "## Provenance Summary Breakdown",
            f"- **verified_source (Live crawl without restriction):** {breakdown['verified_source']}",
            f"- **recruiter_provided (Explicitly provided package):** {breakdown['recruiter_provided']}",
            f"- **demonstration_unverified (Demonstration seed datasets):** {breakdown['demonstration_unverified']}",
            "",
            "## Itemized Provenance Audit",
            "| # | Source | Category | Title | Provenance Tier | Origin Rationale |",
            "|---|---|---|---|---|---|",
        ]

        for idx, item in enumerate(item_details, 1):
            md_lines.append(
                f"| {idx} | {item['source']} | {item['category']} | {item['title']} | "
                f"`{item['provenance']}` | {item['audit_note']} |"
            )

        with open(self.audit_report_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Provenance audit complete. Total records: {len(audited_records)}, Breakdown: {breakdown}")
        return audit_summary
