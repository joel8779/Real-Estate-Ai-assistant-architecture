"""
UI Refinement Verification Script for DarGlobal & Wasalt AI Knowledge Engine.
Tests Markdown rendering, auto-scrolling behavior, and visual styling.
"""
import sys
import os
import json
import time
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def wait_for_url(url: str, timeout: float = 30.0) -> bool:
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status in (200, 304):
                    return True
        except Exception:
            time.sleep(0.5)
    return False

def verify_ui_refinement():
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "browser": "Chromium (Playwright)",
        "tests": []
    }

    print("=" * 70, flush=True)
    print("UI REFINEMENT VERIFICATION - PLAYWRIGHT CHROMIUM", flush=True)
    print("=" * 70, flush=True)

    # 1. Start Backend Subprocess
    print("\n[1/6] Starting FastAPI Backend on http://127.0.0.1:8000...", flush=True)
    backend_log_file = open(PROCESSED_DIR / "backend_refine.log", "w", encoding="utf-8")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT_DIR),
        stdout=backend_log_file,
        stderr=subprocess.STDOUT,
    )

    # 2. Build and Start Frontend Preview Subprocess
    print("[2/6] Starting Vite Frontend Preview on http://127.0.0.1:5173...", flush=True)
    frontend_log_file = open(PROCESSED_DIR / "frontend_refine.log", "w", encoding="utf-8")
    frontend_proc = subprocess.Popen(
        ["npx.cmd" if sys.platform == "win32" else "npx", "vite", "preview", "--port", "5173", "--host", "127.0.0.1"],
        cwd=str(FRONTEND_DIR),
        stdout=frontend_log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        print("Waiting for backend /healthz...", flush=True)
        if not wait_for_url("http://127.0.0.1:8000/healthz", timeout=25.0):
            raise RuntimeError("Backend failed to start within 25s.")
        print("✓ Backend is healthy.", flush=True)

        print("Waiting for frontend preview...", flush=True)
        if not wait_for_url("http://127.0.0.1:5173", timeout=25.0):
            raise RuntimeError("Frontend failed to start within 25s.")
        print("✓ Frontend preview reachable.", flush=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 850})

            # Check console errors
            page.on("console", lambda msg: print(f"  [Console] {msg.type}: {msg.text}", flush=True))

            page.goto("http://127.0.0.1:5173", wait_until="networkidle")
            time.sleep(1.5)

            # Test 1: Color palette and theme verification
            print("\n--- Test 1: Design System & Color Palette Verification ---", flush=True)
            bg_color = page.evaluate("() => window.getComputedStyle(document.body).backgroundColor")
            print(f"Body background color: {bg_color}")
            # rgb(11, 11, 11) is #0B0B0B
            assert "11, 11, 11" in bg_color or "rgb(11, 11, 11)" in bg_color, f"Unexpected body background: {bg_color}"
            print("✓ Primary background is confirmed near-black #0B0B0B (rgb(11, 11, 11)).")

            screenshot_welcome = PROCESSED_DIR / "ui_refinement_welcome.png"
            page.screenshot(path=str(screenshot_welcome))
            print(f"✓ Welcome screen screenshot: {screenshot_welcome.name}")
            report["tests"].append({
                "test": "Color Palette & Background",
                "status": "PASSED",
                "body_background": bg_color,
                "screenshot": screenshot_welcome.name
            })

            # Test 2: Live Query & Markdown Parsing Verification
            print("\n--- Test 2: Live Query & Markdown Rendering ---", flush=True)
            query = "What branded residences does DarGlobal develop in Dubai and Oman?"
            page.fill("input[type='text']", query)
            page.click("button[type='submit']")

            print("Waiting for assistant response...", flush=True)
            page.wait_for_selector(".prose", timeout=60000)
            time.sleep(1.0)

            # Inspect markdown elements inside .prose
            markdown_check = page.evaluate("""() => {
                const prose = document.querySelector('.prose');
                if (!prose) return { found: false };
                const text = prose.innerText;
                const hasRawAsterisks = text.includes('**');
                const strongTags = prose.querySelectorAll('strong').length;
                const listItems = prose.querySelectorAll('li').length;
                const headings = prose.querySelectorAll('h1, h2, h3, h4').length;
                const paragraphs = prose.querySelectorAll('p').length;
                const codeBlocks = prose.querySelectorAll('code').length;
                return {
                    found: true,
                    hasRawAsterisks,
                    strongTags,
                    listItems,
                    headings,
                    paragraphs,
                    codeBlocks,
                    sampleSnippet: text.slice(0, 180)
                };
            }""")

            print(f"Markdown parsing results: {markdown_check}")
            assert not markdown_check["hasRawAsterisks"], "Raw ** markdown asterisks detected in rendered text!"
            assert markdown_check["paragraphs"] > 0, "No paragraphs rendered in assistant response"
            print("✓ Markdown rendering confirmed: No raw '**' syntax; semantic HTML tags rendered cleanly.")

            report["tests"].append({
                "test": "Markdown Rendering",
                "status": "PASSED",
                "details": markdown_check
            })

            # Test 3: Auto-scrolling verification
            print("\n--- Test 3: Auto-scrolling Behavior ---", flush=True)
            scroll_check = page.evaluate("""() => {
                const scrollContainer = document.querySelector('.overflow-y-auto');
                const lastMessage = document.querySelectorAll('.group')[0]; // Assistant message wrapper
                if (!scrollContainer || !lastMessage) return { error: 'Containers not found' };

                const containerRect = scrollContainer.getBoundingClientRect();
                const msgRect = lastMessage.getBoundingClientRect();
                const citationsHeader = Array.from(document.querySelectorAll('h4')).find(el => el.textContent.includes('Sources'));
                const citationsRect = citationsHeader ? citationsHeader.getBoundingClientRect() : null;

                return {
                    containerScrollTop: scrollContainer.scrollTop,
                    containerHeight: containerRect.height,
                    messageTopRelativeToContainer: msgRect.top - containerRect.top,
                    isMessageTopVisible: (msgRect.top >= containerRect.top - 50) && (msgRect.top <= containerRect.bottom),
                    isScrolledToBottom: (scrollContainer.scrollTop + scrollContainer.clientHeight >= scrollContainer.scrollHeight - 20)
                };
            }""")
            print(f"Scroll metrics: {scroll_check}")
            assert scroll_check["isMessageTopVisible"], "Assistant message top is not visible in scroll container!"
            print("✓ Auto-scrolling confirmed: Assistant message top is visible to the user on arrival.")

            screenshot_response = PROCESSED_DIR / "ui_refinement_response.png"
            page.screenshot(path=str(screenshot_response))
            print(f"✓ Response screenshot: {screenshot_response.name}")

            report["tests"].append({
                "test": "Auto-scrolling Behavior",
                "status": "PASSED",
                "scroll_metrics": scroll_check,
                "screenshot": screenshot_response.name
            })

            # Test 4: Compact Citations and Demonstration Notice Layout
            print("\n--- Test 4: Compact Sources and Demonstration Notice Layout ---", flush=True)
            demo_notice = page.locator("text=Demonstration Dataset Notice")
            assert demo_notice.is_visible(), "Demonstration notice not visible"

            citations_heading = page.locator("h4:has-text('Sources')")
            assert citations_heading.is_visible(), "Sources heading not visible"
            
            # Check compact citation links
            citation_links = page.locator("a[target='_blank']")
            link_count = citation_links.count()
            assert link_count > 0, "No external citation links found"
            print(f"✓ Found {link_count} compact citation rows.")

            report["tests"].append({
                "test": "Compact Citations & Demo Notice",
                "status": "PASSED",
                "citation_links_count": link_count
            })

            # Test 5: Follow-up question & preservation of scroll stability
            print("\n--- Test 5: Follow-up Turn & No Errant Re-scrolling ---", flush=True)
            time.sleep(3.0)
            page.fill("input[type='text']", "Tell me about Trump International in Oman.")
            page.click("button[type='submit']")

            print("Waiting for follow-up response...", flush=True)
            page.wait_for_function("document.querySelectorAll('.prose').length >= 2", timeout=60000)
            time.sleep(1.0)

            followup_check = page.evaluate("""() => {
                const scrollContainer = document.querySelector('.overflow-y-auto');
                const messages = document.querySelectorAll('.prose');
                const secondMsg = messages[1];
                const containerRect = scrollContainer.getBoundingClientRect();
                const msgRect = secondMsg.getBoundingClientRect();
                return {
                    secondMsgTopRelativeToContainer: msgRect.top - containerRect.top,
                    isSecondMsgVisible: (msgRect.top >= containerRect.top - 60) && (msgRect.top <= containerRect.bottom)
                };
            }""")
            print(f"Follow-up scroll position: {followup_check}")
            assert followup_check["isSecondMsgVisible"], "Follow-up message top not in view!"
            print("✓ Follow-up message top correctly positioned in viewport.")

            screenshot_followup = PROCESSED_DIR / "ui_refinement_followup.png"
            page.screenshot(path=str(screenshot_followup))
            print(f"✓ Follow-up screenshot: {screenshot_followup.name}")

            report["tests"].append({
                "test": "Follow-up Scroll Position",
                "status": "PASSED",
                "screenshot": screenshot_followup.name
            })

            # Test 6: Clear Chat
            print("\n--- Test 6: Clear Chat ---", flush=True)
            page.click("button[title='Start fresh conversation']")
            time.sleep(0.5)
            assert page.locator("text=Real Estate Intelligence Assistant").is_visible()
            print("✓ Clear chat resets state properly.")

            report["tests"].append({
                "test": "Clear Chat Reset",
                "status": "PASSED"
            })

            browser.close()

        # Save Report
        report_path = PROCESSED_DIR / "ui_refinement_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nVerification report written to {report_path}")

        print("\n" + "=" * 70)
        print("ALL UI REFINEMENT VERIFICATIONS PASSED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        print("\nShutting down test servers...")
        try:
            if backend_proc.poll() is None:
                backend_proc.terminate()
                backend_proc.wait(timeout=3)
        except Exception:
            backend_proc.kill()

        try:
            if frontend_proc.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(frontend_proc.pid)], capture_output=True)
                else:
                    frontend_proc.terminate()
                    frontend_proc.wait(timeout=3)
        except Exception:
            pass
        try:
            backend_log_file.close()
            frontend_log_file.close()
        except Exception:
            pass

if __name__ == "__main__":
    verify_ui_refinement()
