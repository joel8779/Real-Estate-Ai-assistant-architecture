"""Unit Tests for HTML Cleaner and Access Restriction Detector."""

from pathlib import Path
from backend.app.ingestion.cleaner import clean_html_content, detect_access_restriction

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_clean_html_content_strips_boilerplate():
    """Verify that scripts, styles, navs, and footers are removed and title/text extracted."""
    html = """
    <html>
        <head>
            <title>Test Luxury Villa</title>
            <meta name="description" content="Exclusive private villa in Riyadh.">
            <script>alert('track');</script>
            <style>body { color: red; }</style>
        </head>
        <body>
            <nav><a href="/">Home</a></nav>
            <h1>Exclusive Villa</h1>
            <p>Spacious living room with pool views.</p>
            <footer>Copyright 2026</footer>
        </body>
    </html>
    """
    title, text, meta = clean_html_content(html)
    assert title == "Test Luxury Villa"
    assert meta["description"] == "Exclusive private villa in Riyadh."
    assert "alert" not in text
    assert "color: red" not in text
    assert "Home" not in text
    assert "Copyright 2026" not in text
    assert "Exclusive Villa" in text
    assert "Spacious living room with pool views." in text


def test_detect_access_restriction_incapsula():
    """Verify detection of Imperva / Incapsula challenge script."""
    challenge_path = FIXTURES_DIR / "incapsula_challenge.html"
    with open(challenge_path, "r", encoding="utf-8") as f:
        html = f.read()

    result = detect_access_restriction(html, status_code=200)
    assert result is not None
    assert "Imperva/Incapsula" in result


def test_detect_access_restriction_status_codes():
    """Verify detection of HTTP 403 and 429 blocks."""
    res_403 = detect_access_restriction("", status_code=403)
    assert res_403 is not None
    assert "403 Forbidden" in res_403

    res_429 = detect_access_restriction("", status_code=429)
    assert res_429 is not None
    assert "429 Too Many Requests" in res_429


def test_detect_access_restriction_clean_page():
    """Verify clean pages return None (no restrictions detected)."""
    clean_html = "<html><head><title>Clean Page</title></head><body><h1>Welcome</h1></body></html>"
    assert detect_access_restriction(clean_html, status_code=200) is None
