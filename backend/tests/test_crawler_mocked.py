"""Unit and Integration Tests for Crawler with Mocked HTTP Responses."""

from pathlib import Path
from unittest.mock import AsyncMock, patch
import httpx
import pytest

from backend.app.schemas.document import ContentCategory
from backend.app.ingestion.config import IngestionSettings
from backend.app.ingestion.crawler import Crawler
from backend.app.ingestion.storage import StorageManager
from backend.app.ingestion.extractors import AccessRestrictedError


@pytest.fixture
def mock_crawler(tmp_path: Path):
    """Fixture providing a crawler configured with temporary directories."""
    settings = IngestionSettings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        seed_data_dir=tmp_path / "seed_data",
        max_retries=2,
        request_delay_seconds=0.01,
        backoff_factor=1.1,
    )
    storage = StorageManager(
        raw_dir=settings.raw_data_dir,
        processed_dir=settings.processed_data_dir,
    )
    return Crawler(settings=settings, storage=storage)


@pytest.mark.asyncio
async def test_crawler_domain_restriction(mock_crawler: Crawler):
    """Verify crawler rejects URLs from unapproved domains."""
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="not in the allowed domain list"):
            await mock_crawler.fetch_url(client, "https://unauthorized-domain.com/listing")


@pytest.mark.asyncio
async def test_crawler_retries_on_transient_500(mock_crawler: Crawler):
    """Verify crawler retries on 500 server errors with exponential backoff."""
    call_count = 0

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            return httpx.Response(500, request=request, text="Internal Server Error")
        return httpx.Response(200, request=request, text="<html><body>Success</body></html>")

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        doc = await mock_crawler.fetch_url(client, "https://wasalt.sa/en/sample")
        assert doc is not None
        assert call_count == 2
        assert "Success" in doc.raw_content


@pytest.mark.asyncio
async def test_crawler_non_retry_on_404(mock_crawler: Crawler):
    """Verify crawler does not retry on 404 client errors."""
    call_count = 0

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(404, request=request, text="Not Found")

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        doc = await mock_crawler.fetch_url(client, "https://wasalt.sa/en/missing-page")
        assert doc is None
        assert call_count == 1  # No retries on 404


@pytest.mark.asyncio
async def test_crawler_detects_incapsula_restriction(mock_crawler: Crawler):
    """Verify crawler detects Incapsula barrier and raises AccessRestrictedError."""
    challenge_html = "<html><script src=\"/_Incapsula_Resource?SWJIYLWA=123\"></script></html>"

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, text=challenge_html)

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(AccessRestrictedError) as exc_info:
            await mock_crawler.fetch_url(client, "https://darglobal.co.uk/projects")

        assert "Imperva/Incapsula" in exc_info.value.reason


@pytest.mark.asyncio
async def test_crawler_full_mocked_ingestion_run(mock_crawler: Crawler):
    """Verify full ingestion run aggregates discovered, processed, and skipped counts into IngestionReport."""
    sample_wasalt = """
    <html>
        <head><title>Wasalt Business Solutions</title></head>
        <body><main><h1>Wasalt Business Portal</h1><p>Comprehensive B2B tools for Saudi real estate.</p></main></body>
    </html>
    """

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "sitemap" in url_str:
            return httpx.Response(200, request=request, text="<urlset><loc>https://wasalt.sa/en/s/wasalt-business</loc></urlset>")
        return httpx.Response(200, request=request, text=sample_wasalt)

    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
        report = await mock_crawler.run_ingestion(source="wasalt", max_pages=2)

        assert report.source == "wasalt"
        assert report.total_processed >= 1
        assert report.duration_seconds >= 0.0
        assert "company_info" in report.categories_breakdown or "other" in report.categories_breakdown
