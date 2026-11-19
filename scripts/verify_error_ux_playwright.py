"""Playwright Browser Verification Script for OpenRouter Rate Limit & Error UX.

Validates in real headless Chromium browser:
1. Real live upstream OpenRouter 429 rate limit response.
2. Compact red error container inside the assistant bubble.
3. Complete suppression of citations, property cards, and demo notices on error.
4. Absence of duplicate global error banners.
5. Presence and functionality of the "Try again" button.
6. In-place message update upon retry without duplicate bubbles.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_url(url: str, timeout: float = 30.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status in (200, 304):
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    print("=" * 80)
    print("PROPINTEL AI - PLAYWRIGHT VERIFICATION: ERROR UX & RETRY INTERACTION")
    print("=" * 80)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_locked": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "scenarios": {},
    }

    # 1. Start FastAPI backend (without STREAM_TEST_MODE so it hits live OpenRouter)
    backend_port = 8008
    frontend_port = 5178
    print(f"\n[1/4] Starting FastAPI backend on port {backend_port} (Live OpenRouter Mode)...")
    backend_log = open(PROCESSED_DIR / "backend_playwright_error_ux.log", "w", encoding="utf-8")
    backend_env = {**os.environ, "PORT": str(backend_port)}
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", str(backend_port)],
        cwd=str(ROOT_DIR),
        stdout=backend_log,
        stderr=subprocess.STDOUT,
        env=backend_env,
    )

    # 2. Start Frontend Vite Preview
    print(f"[2/4] Starting Vite frontend preview on port {frontend_port}...")
    frontend_log = open(PROCESSED_DIR / "frontend_playwright_error_ux.log", "w", encoding="utf-8")
    frontend_env = {**os.environ, "BACKEND_PORT": str(backend_port)}
    frontend_proc = subprocess.Popen(
        ["npx.cmd" if sys.platform == "win32" else "npx", "vite", "preview", "--port", str(frontend_port), "--host", "127.0.0.1"],
        cwd=str(FRONTEND_DIR),
        stdout=frontend_log,
        stderr=subprocess.STDOUT,
        env=frontend_env,
    )

    try:
        print("Waiting for servers to become ready...")
        if not wait_for_url(f"http://127.0.0.1:{backend_port}/healthz", timeout=25.0):
            raise RuntimeError("Backend failed to start.")
        if not wait_for_url(f"http://127.0.0.1:{frontend_port}", timeout=25.0):
            raise RuntimeError("Frontend failed to start.")
        print("✓ Both servers are ready.")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 850})
            page = context.new_page()
            page.on("console", lambda msg: print(f"[BROWSER {msg.type.upper()}] {msg.text}"))
            page.on("pageerror", lambda err: print(f"[BROWSER ERROR] {err}"))

            # Navigate to frontend
            print("\n[3/4] Navigating to PropIntel AI UI...")
            page.goto(f"http://127.0.0.1:{frontend_port}")
            page.wait_for_selector("input[placeholder*='Ask anything']")
            print("✓ UI loaded successfully.")

            # Scenario A: Live upstream 429 error handling
            print("\nSubmitting user query against live backend: 'What is AIDA in Oman?'...")
            page.fill("input[placeholder*='Ask anything']", "What is AIDA in Oman?")
            page.click("button[aria-label='Send message']")

            # Wait for assistant bubble to appear (allowing time for initial embedding model load + live 429)
            print("Waiting for response from live OpenRouter...")
            page.wait_for_selector("text=The AI service is temporarily busy", timeout=45000)
            time.sleep(1.0)

            # Check 1: Single error treatment inside the message bubble
            error_elements = page.locator("text=The AI service is temporarily busy").all()
            print(f"Error text occurrence count: {len(error_elements)} (Expected: 1, no duplicate banners)")
            assert len(error_elements) == 1, f"Expected 1 error text, found {len(error_elements)}"

            # Check 2: Service Notice badge visible in message header
            service_notice = page.locator("text=Service Notice")
            assert service_notice.is_visible(), "Service Notice badge should be visible on failed generation"
            print("✓ Service Notice header badge is visible.")

            # Check 3: 'Try again' button is present
            retry_btn = page.locator("button:has-text('Try again')")
            assert retry_btn.is_visible(), "'Try again' button must be rendered inside error bubble"
            print("✓ 'Try again' button is present and visible.")

            # Check 4: No citations or sources displayed
            sources_header = page.locator("text=Sources & Evidence")
            assert not sources_header.is_visible(), "Citations must NOT be displayed on failed generation"
            print("✓ Citations and property cards are correctly suppressed.")

            # Check 5: Demonstration warning banner is suppressed on error screen
            demo_warning_banner = page.locator("text=Demonstration Dataset Notice")
            assert not demo_warning_banner.is_visible(), "Demonstration warning banner must be suppressed on error"
            print("✓ Demonstration Dataset Notice is correctly suppressed on error.")

            # Save screenshot of live 429 error treatment
            screenshot_429 = PROCESSED_DIR / "playwright_live_429_ux.png"
            page.screenshot(path=str(screenshot_429), full_page=True)
            print(f"✓ Screenshot saved: {screenshot_429}")

            report["scenarios"]["live_upstream_429"] = {
                "status": "PASSED",
                "single_error_container": True,
                "service_notice_badge": True,
                "try_again_button_present": True,
                "citations_suppressed": True,
                "demo_warning_suppressed": True,
                "screenshot": str(screenshot_429),
            }

            # Scenario B: In-place retry interaction test
            print("\n[4/4] Testing in-place retry interaction...")
            # Count current message bubbles before retry
            initial_user_bubbles = page.locator("p:has-text('What is AIDA in Oman?')").count()
            print(f"Initial user message count: {initial_user_bubbles}")

            # Intercept next request to simulate successful retried response
            mock_sse_response = (
                "event: status\n"
                'data: {"stage": "generating", "message": "Synthesizing answer..."}\n\n'
                "event: token\n"
                'data: {"token": "AIDA is a premier cliffside master community in Muscat, Oman."}\n\n'
                "event: citations\n"
                'data: {"citations": [{"title": "AIDA Project Overview", "url": "https://darglobal.co.uk/projects/aida", "source": "darglobal", "category": "project", "provenance": "demonstration_unverified"}], "grounded": true, "data_mode": "demonstration", "provenance_warning": "Notice: Generated from demonstration records."}\n\n'
                "event: complete\n"
                'data: {"latency_ms": 110, "grounded": true, "data_mode": "demonstration", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"}\n\n'
            )

            # Route next stream request to mock successful retry
            def handle_stream_retry(route):
                print(f"ROUTE INTERCEPTED: {route.request.url}")
                route.fulfill(
                    status=200,
                    headers={
                        "Content-Type": "text/event-stream; charset=utf-8",
                        "Cache-Control": "no-cache",
                    },
                    body=mock_sse_response,
                )

            page.route("**/api/v1/chat/stream", handle_stream_retry)

            # Click "Try again"
            print("Clicking 'Try again' button...")
            retry_btn.click()

            # Wait for successful retried content to render
            page.wait_for_selector("text=AIDA is a premier cliffside master community", timeout=10000)
            time.sleep(1.0)

            # Check 1: User messages count must NOT have increased (no duplicate message)
            final_user_bubbles = page.locator("p:has-text('What is AIDA in Oman?')").count()
            print(f"Final user message count after retry: {final_user_bubbles}")
            assert final_user_bubbles == initial_user_bubbles, "Retry should NOT add duplicate user messages"

            # Check 2: Error box is gone
            assert not page.locator("text=The AI service is temporarily busy").is_visible(), "Error box must be cleared"
            print("✓ Error box was replaced in-place by retried answer.")

            # Check 3: Validated citations and demonstration warning appear on SUCCESSFUL answer
            assert page.locator("text=AIDA Project Overview").is_visible(), "Citations must appear on successful answer"
            assert page.locator("text=Demonstration Dataset Notice").is_visible(), "Demonstration notice appears on successful answer"
            print("✓ Sources and demonstration notice correctly appear on successful retried response.")

            # Save screenshot of successful retry
            screenshot_retry = PROCESSED_DIR / "playwright_retry_success_ux.png"
            page.screenshot(path=str(screenshot_retry), full_page=True)
            print(f"✓ Screenshot saved: {screenshot_retry}")

            report["scenarios"]["in_place_retry"] = {
                "status": "PASSED",
                "in_place_replacement": True,
                "no_duplicate_messages": True,
                "error_cleared": True,
                "citations_rendered_on_success": True,
                "demo_warning_rendered_on_success": True,
                "screenshot": str(screenshot_retry),
            }

            browser.close()

        # Save report
        report_file = PROCESSED_DIR / "error_ux_verification_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Verification report written to: {report_file}")
        print("\nAll Playwright UI tests PASSED successfully!")

    finally:
        # Cleanup processes
        print("\nCleaning up processes...")
        try:
            backend_proc.terminate()
            backend_proc.wait(timeout=5)
        except Exception:
            backend_proc.kill()
        try:
            frontend_proc.terminate()
            frontend_proc.wait(timeout=5)
        except Exception:
            frontend_proc.kill()
        backend_log.close()
        frontend_log.close()
        print("✓ Cleanup complete.")


if __name__ == "__main__":
    main()
