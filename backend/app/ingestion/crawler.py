"""Asynchronous Web Crawler and Sitemap Discovery Engine.

Implements polite crawling with rate limiting, concurrency controls, exponential
backoff on transient errors, domain allowlisting, and strict bot barrier reporting.
"""

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
import httpx

from backend.app.schemas.document import RawDocument, IngestionReport, IngestionFailure
from backend.app.ingestion.config import IngestionSettings, default_ingestion_settings
from backend.app.ingestion.url_utils import canonicalize_url, compute_content_hash, generate_doc_id
from backend.app.ingestion.cleaner import detect_access_restriction
from backend.app.ingestion.extractors import WasaltExtractor, DarGlobalExtractor, AccessRestrictedError
from backend.app.ingestion.storage import StorageManager
from backend.app.core.logging import logger


class Crawler:
    """Robust, compliant crawler for permitted real estate content."""

    def __init__(
        self,
        settings: Optional[IngestionSettings] = None,
        storage: Optional[StorageManager] = None,
    ):
        self.settings = settings or default_ingestion_settings
        self.storage = storage or StorageManager(
            raw_dir=self.settings.raw_data_dir,
            processed_dir=self.settings.processed_dir,
        )
        self.semaphore = asyncio.Semaphore(self.settings.max_concurrency)
        self.wasalt_extractor = WasaltExtractor()
        self.darglobal_extractor = DarGlobalExtractor()

    async def fetch_url(self, client: httpx.AsyncClient, url: str) -> Optional[RawDocument]:
        """Fetch a single URL with retries, exponential backoff, and domain validation.

        Returns:
            RawDocument if successful, or raises AccessRestrictedError if bot barrier detected.
        """
        canonical = canonicalize_url(url, allowed_domains=self.settings.allowed_domains)

        async with self.semaphore:
            # Politeness delay between requests
            await asyncio.sleep(self.settings.request_delay_seconds)

            retries = 0
            while retries <= self.settings.max_retries:
                try:
                    logger.debug(f"Fetching URL: {canonical} (attempt {retries + 1})")
                    response = await client.get(
                        canonical,
                        timeout=self.settings.timeout_seconds,
                        follow_redirects=True,
                    )

                    headers_dict = {k: v for k, v in response.headers.items()}

                    # Check for access control barriers immediately
                    restriction = detect_access_restriction(
                        response.text, response.status_code, headers_dict
                    )
                    if restriction:
                        raise AccessRestrictedError(canonical, restriction)

                    # Non-retryable client errors (404, 400, etc.)
                    if 400 <= response.status_code < 500:
                        logger.warning(f"Client error {response.status_code} fetching {canonical}")
                        return None

                    # Transient server error (500, 502, 503, 504)
                    if response.status_code >= 500:
                        retries += 1
                        if retries > self.settings.max_retries:
                            logger.error(f"Server error {response.status_code} on {canonical} after {retries} retries.")
                            return None
                        wait_time = self.settings.backoff_factor ** retries
                        await asyncio.sleep(wait_time)
                        continue

                    # Determine source domain
                    source = "wasalt" if "wasalt.sa" in canonical else "darglobal"
                    doc_id = generate_doc_id(canonical)
                    content_hash = compute_content_hash(response.text)

                    return RawDocument(
                        id=doc_id,
                        source=source,
                        url=canonical,
                        title="",
                        content_type="html",
                        raw_content=response.text,
                        content_hash=content_hash,
                        scraped_at=datetime.now(timezone.utc),
                        http_status=response.status_code,
                        headers=headers_dict,
                    )

                except AccessRestrictedError:
                    raise  # Propagate access restrictions to report them clearly

                except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as exc:
                    retries += 1
                    if retries > self.settings.max_retries:
                        logger.error(f"Network failure on {canonical} after {retries} retries: {exc}")
                        return None
                    wait_time = self.settings.backoff_factor ** retries
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    logger.error(f"Unexpected error fetching {canonical}: {e}")
                    return None

        return None

    async def discover_wasalt_urls(self, client: httpx.AsyncClient, limit: int = 25) -> List[str]:
        """Discover permitted URLs from Wasalt sitemaps."""
        discovered: List[str] = []
        sitemap_urls = [
            "https://cdn.wasalt.sa/sitemap/static_sitemap_en_sa.xml.gz",
            "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz",
            "https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz",
        ]

        for sm_url in sitemap_urls:
            if len(discovered) >= limit:
                break
            try:
                resp = await client.get(sm_url, timeout=10.0)
                if resp.status_code == 200:
                    urls = re.findall(r"<loc>(.*?)</loc>", resp.text)
                    for u in urls:
                        u_clean = u.strip()
                        if "/en" in u_clean and u_clean not in discovered:
                            discovered.append(u_clean)
                            if len(discovered) >= limit:
                                break
            except Exception as e:
                logger.warning(f"Error fetching Wasalt sitemap {sm_url}: {e}")

        # Ensure seed fallback URLs are present
        fallback_seeds = [
            "https://wasalt.sa/en/s/wasalt-business",
            "https://wasalt.sa/en/s/masterplan",
            "https://wasalt.sa/en/s/privacy-policy",
            "https://wasalt.sa/en/s/terms-and-conditions",
        ]
        for fb in fallback_seeds:
            if fb not in discovered:
                discovered.append(fb)

        return discovered[:limit]

    async def discover_darglobal_urls(self, client: httpx.AsyncClient, limit: int = 25) -> List[str]:
        """Discover permitted URLs from DarGlobal sitemap."""
        discovered: List[str] = []
        sitemap_url = "https://darglobal.co.uk/sitemap.xml"

        try:
            resp = await client.get(sitemap_url, timeout=10.0)
            if resp.status_code == 200:
                urls = re.findall(r"<loc>(.*?)</loc>", resp.text)
                for u in urls:
                    u_clean = u.strip()
                    if u_clean not in discovered:
                        discovered.append(u_clean)
                        if len(discovered) >= limit:
                            break
        except Exception as e:
            logger.warning(f"Error fetching DarGlobal sitemap: {e}")

        # Fallback seeds
        fallback_seeds = [
            "https://darglobal.co.uk/about",
            "https://darglobal.co.uk/projects",
            "https://darglobal.co.uk/investor",
            "https://darglobal.co.uk/press",
            "https://darglobal.co.uk/faq",
        ]
        for fb in fallback_seeds:
            if fb not in discovered:
                discovered.append(fb)

        return discovered[:limit]

    async def run_ingestion(
        self,
        source: str = "all",
        max_pages: Optional[int] = None,
        reprocess_only: bool = False,
    ) -> IngestionReport:
        """Execute full ingestion pipeline for specified source(s)."""
        started_at = datetime.now(timezone.utc)
        limit = max_pages or self.settings.max_pages_per_source

        report = IngestionReport(
            source=source,
            started_at=started_at,
            completed_at=started_at,
            duration_seconds=0.0,
        )

        headers = {"User-Agent": self.settings.user_agent}

        # Reprocess offline raw files if requested
        if reprocess_only:
            logger.info("Executing offline reprocessing of existing raw documents...")
            raw_docs = self.storage.load_all_raw(source if source != "all" else None)
            report.seed_discovered = len(raw_docs)

            for doc in raw_docs:
                try:
                    extractor = self.wasalt_extractor if doc.source == "wasalt" else self.darglobal_extractor
                    entities = extractor.extract(doc)
                    saved, dups = self.storage.save_normalized(entities)
                    report.seed_processed += saved
                    report.total_processed += saved
                    report.duplicates_detected += dups
                    for ent in entities:
                        cat_name = ent.category.value
                        report.categories_breakdown[cat_name] = report.categories_breakdown.get(cat_name, 0) + 1
                        prov_key = str(ent.provenance)
                        report.provenance_breakdown[prov_key] = report.provenance_breakdown.get(prov_key, 0) + 1
                except Exception as e:
                    report.failed_urls.append(IngestionFailure(url=doc.url, reason=str(e)))

            completed_at = datetime.now(timezone.utc)
            report.completed_at = completed_at
            report.duration_seconds = (completed_at - started_at).total_seconds()
            self.storage.save_report(report)
            return report

        # Live Crawling Mode
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            target_urls: List[str] = []

            if source in ("wasalt", "all"):
                wasalt_urls = await self.discover_wasalt_urls(client, limit=limit)
                target_urls.extend(wasalt_urls)

            if source in ("darglobal", "all"):
                darglobal_urls = await self.discover_darglobal_urls(client, limit=limit)
                target_urls.extend(darglobal_urls)

            report.live_discovered = len(target_urls)
            logger.info(f"Discovered {len(target_urls)} candidate URLs for live inspection")

            for url in target_urls:
                try:
                    doc = await self.fetch_url(client, url)
                    if not doc:
                        continue

                    # If successful live fetch, tag as verified_source
                    doc.provenance = "verified_source"
                    self.storage.save_raw(doc)

                    # Extract normalized entities
                    extractor = self.wasalt_extractor if doc.source == "wasalt" else self.darglobal_extractor
                    entities = extractor.extract(doc)

                    saved, dups = self.storage.save_normalized(entities)
                    report.live_processed += saved
                    report.total_processed += saved
                    report.duplicates_detected += dups

                    for ent in entities:
                        cat_name = ent.category.value
                        report.categories_breakdown[cat_name] = report.categories_breakdown.get(cat_name, 0) + 1
                        prov_key = str(ent.provenance)
                        report.provenance_breakdown[prov_key] = report.provenance_breakdown.get(prov_key, 0) + 1

                except AccessRestrictedError as are:
                    logger.warning(f"Source access control restriction encountered on {are.url}: {are.reason}")
                    report.live_restricted += 1
                    report.access_restrictions.append({"url": are.url, "reason": are.reason})
                    report.failed_urls.append(
                        IngestionFailure(
                            url=are.url,
                            reason=are.reason,
                            error_type="ACCESS_RESTRICTION",
                        )
                    )

                except Exception as e:
                    logger.error(f"Extraction failed for {url}: {e}")
                    report.failed_urls.append(IngestionFailure(url=url, reason=str(e)))

            # Ingest permitted/provided offline seed documents for requested source(s)
            sources_to_seed = ["wasalt", "darglobal"] if source == "all" else [source]
            for src_name in sources_to_seed:
                seed_docs = self.load_seed_documents(src_name)
                report.seed_discovered += len(seed_docs)
                if seed_docs:
                    logger.info(f"Ingesting {len(seed_docs)} permitted/provided offline documents for {src_name}...")
                    extractor = self.wasalt_extractor if src_name == "wasalt" else self.darglobal_extractor
                    for s_doc in seed_docs:
                        self.storage.save_raw(s_doc)
                        entities = extractor.extract(s_doc)
                        saved, dups = self.storage.save_normalized(entities)
                        report.seed_processed += saved
                        report.total_processed += saved
                        report.duplicates_detected += dups
                        for ent in entities:
                            cat_name = ent.category.value
                            report.categories_breakdown[cat_name] = report.categories_breakdown.get(cat_name, 0) + 1
                            prov_key = str(ent.provenance)
                            report.provenance_breakdown[prov_key] = report.provenance_breakdown.get(prov_key, 0) + 1

        completed_at = datetime.now(timezone.utc)
        report.completed_at = completed_at
        report.duration_seconds = (completed_at - started_at).total_seconds()

        # Persist report to data/processed/ingestion_report.json & .md
        self.storage.save_report(report)
        return report

    def load_seed_documents(self, source: str) -> List[RawDocument]:
        """Load explicitly permitted or recruiter-provided seed documents from data/seed_data/."""
        seed_dir = self.settings.seed_data_dir / source
        loaded: List[RawDocument] = []
        if not seed_dir.exists():
            return loaded

        for json_file in seed_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    doc = RawDocument(
                        id=data.get("id", generate_doc_id(data.get("url", str(json_file)))),
                        source=source,
                        url=data.get("url", f"https://darglobal.co.uk/{json_file.stem}"),
                        title=data.get("title", ""),
                        content_type=data.get("content_type", "html"),
                        raw_content=data.get("raw_content", ""),
                        content_hash=compute_content_hash(data.get("raw_content", "")),
                        scraped_at=datetime.now(timezone.utc),
                        http_status=data.get("http_status", 200),
                        headers=data.get("headers", {}),
                    )
                    loaded.append(doc)
            except Exception as e:
                logger.warning(f"Error loading seed file {json_file}: {e}")
        return loaded
