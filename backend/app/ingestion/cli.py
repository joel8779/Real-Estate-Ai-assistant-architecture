"""Ingestion Command-Line Interface (CLI).

Provides commands to crawl, normalize, deduplicate, reprocess, and report on
public real estate data from DarGlobal and Wasalt.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from backend.app.ingestion.config import IngestionSettings
from backend.app.ingestion.crawler import Crawler
from backend.app.ingestion.storage import StorageManager
from backend.app.core.logging import logger, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DarGlobal & Wasalt Real Estate Knowledge Ingestion Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["all", "wasalt", "darglobal"],
        default="wasalt",
        help="Target data source to ingest.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=15,
        help="Maximum pages to discover and process per source.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Politeness delay in seconds between HTTP requests.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess existing raw documents in data/raw without making network requests.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Custom file path for JSON ingestion report output.",
    )
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    setup_logging(log_level="INFO", json_format=False)

    print("=" * 70)
    print("DarGlobal & Wasalt AI Knowledge Assistant — Ingestion Pipeline")
    print("=" * 70)
    print(f"Target Source  : {args.source.upper()}")
    print(f"Reprocess Only : {args.reprocess}")
    print(f"Max Pages      : {args.max_pages}")
    print(f"Polite Delay   : {args.delay}s")
    print("-" * 70)

    settings = IngestionSettings(
        max_pages_per_source=args.max_pages,
        request_delay_seconds=args.delay,
    )
    storage = StorageManager(
        raw_dir=settings.raw_data_dir,
        processed_dir=settings.processed_data_dir,
    )
    crawler = Crawler(settings=settings, storage=storage)

    try:
        report = await crawler.run_ingestion(
            source=args.source,
            max_pages=args.max_pages,
            reprocess_only=args.reprocess,
        )

        print("\n" + "=" * 70)
        print("INGESTION EXECUTION REPORT")
        print("=" * 70)
        print(f"Source Scope              : {report.source}")
        print(f"Duration                  : {report.duration_seconds:.2f} seconds")
        print(f"Live URLs Discovered      : {report.live_discovered}")
        print(f"Live URLs Processed       : {report.live_processed}")
        print(f"Live URLs Restricted (WAF): {report.live_restricted}")
        print(f"Seed Documents Discovered : {report.seed_discovered}")
        print(f"Seed Documents Processed  : {report.seed_processed}")
        print(f"Total Records Indexed     : {report.total_processed}")
        print(f"Duplicates Skipped        : {report.duplicates_detected}")
        print("-" * 70)
        print("Provenance Breakdown      :")
        for prov, count in report.provenance_breakdown.items():
            print(f"  • {prov:<25} : {count}")
        print("-" * 70)
        print("Records by Category       :")
        for cat, count in report.categories_breakdown.items():
            print(f"  • {cat.replace('_', ' ').capitalize():<25} : {count}")

        if report.access_restrictions:
            print("-" * 70)
            print("ACCESS RESTRICTIONS DETECTED (Respecting Website Policies):")
            for res in report.access_restrictions:
                print(f"  [RESTRICTED] {res.get('url')}")
                print(f"               Reason: {res.get('reason')}")

        if report.failed_urls:
            print("-" * 70)
            print(f"Failed URLs ({len(report.failed_urls)}):")
            for fail in report.failed_urls[:5]:
                print(f"  [FAILED] {fail.url} -> {fail.reason}")

        print("=" * 70)
        print(f"Normalized records saved to : data/processed/records.jsonl")
        print(f"Audit report saved to       : data/processed/ingestion_report.json")
        print("=" * 70)

        return 0
    except Exception as e:
        logger.error(f"Fatal error during ingestion run: {e}", exc_info=True)
        return 1


def main():
    exit_code = asyncio.run(main_async())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
