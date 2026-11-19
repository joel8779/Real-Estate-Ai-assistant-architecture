"""HTML Cleansing, Text Normalization, and Access Restriction Detection."""

import re
from typing import Dict, Optional, Tuple
from bs4 import BeautifulSoup


# Tags to strip entirely including their inner text
BOILERPLATE_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "nav",
    "footer",
    "form",
    "button",
    "select",
    "input",
}


def detect_access_restriction(html: str, status_code: int = 200, headers: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Inspect response content and headers to detect bot challenges, WAF barriers, and CAPTCHAs.

    Returns a detailed description if an access restriction is detected, otherwise None.
    """
    headers = headers or {}
    lower_html = html.lower() if html else ""

    # 1. Imperva / Incapsula Bot Challenge Detection
    if "_incapsula_resource" in lower_html or "swjiylwa" in lower_html:
        return "ACCESS_RESTRICTION: Imperva/Incapsula challenge script detected. Automated access restricted."
    if any("incap" in k.lower() or "incap" in v.lower() for k, v in headers.items()):
        if len(html) < 1500 and ("script" in lower_html or "iframe" in lower_html):
            return "ACCESS_RESTRICTION: Imperva/Incapsula challenge detected via response headers and challenge body."

    # 2. Cloudflare Bot Challenge / Turnstile Detection
    if "cf-chl-" in lower_html or "cf-browser-verification" in lower_html or "just a moment..." in lower_html:
        return "ACCESS_RESTRICTION: Cloudflare challenge page detected."

    # 3. Generic CAPTCHA or reCAPTCHA
    if "g-recaptcha" in lower_html or "h-captcha" in lower_html or "captcha-delivery" in lower_html:
        return "ACCESS_RESTRICTION: CAPTCHA challenge detected."

    # 4. HTTP 403 Forbidden / 401 Unauthorized / 429 Too Many Requests
    if status_code == 403:
        return "ACCESS_RESTRICTION: HTTP 403 Forbidden - Access denied by server policy."
    if status_code == 429:
        return "ACCESS_RESTRICTION: HTTP 429 Too Many Requests - Rate limit enforced by server."

    return None


def clean_html_content(raw_html: str) -> Tuple[str, str, Dict[str, str]]:
    """Clean raw HTML into stripped, readable textual content and extract basic metadata.

    Returns:
        (page_title, cleaned_text, meta_attributes)
    """
    if not raw_html or not raw_html.strip():
        return "", "", {}

    soup = BeautifulSoup(raw_html, "html.parser")

    # Extract Page Title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find("meta", property="og:title"):
        title = soup.find("meta", property="og:title").get("content", "").strip()

    # Extract Meta Description
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag["content"].strip()

    # Remove boilerplate and script tags
    for tag in soup(BOILERPLATE_TAGS):
        tag.decompose()

    # Extract text content with paragraph breaks
    text = soup.get_text(separator="\n", strip=True)

    # Clean redundant whitespace and multi-line breaks
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = "\n\n".join(lines)

    meta_attributes = {
        "description": meta_desc,
    }

    return title, cleaned_text, meta_attributes
