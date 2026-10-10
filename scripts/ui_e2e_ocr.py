"""End-to-end browser test of OCR: a real scan is ingested, viewed, corrected and re-ingested.

Unlike ``ui_smoke_ocr.py`` (which only looks at the live app and cancels its job), this
starts its *own* throwaway server on a free port with a scratch corpus.db, jobs.db and
index in a temp folder, so real OCR jobs run without touching your library:

    python scripts/ui_e2e_ocr.py            # headless
    python scripts/ui_e2e_ocr.py --show     # watch it in a browser window

Covers: queue a PDF through the OCR page -> job runs Tesseract/Kraken -> document appears
in "Documents made from scans" -> scan viewer shows the page image beside the text ->
edit + Ctrl+S saves a correction -> "Re-ingest with corrections" makes a corrected
copy whose text contains the edit -> library audit button. Screenshots: data/ui_smoke/e2e_*.png
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
TEXT = ("Fides quaerens intellectum est. Omnes homines natura scire desiderant, "
        "et ea quae sunt in rerum natura cognoscere.")
EDIT = "CORRECTED-BY-E2E"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_job(base: str, job_id: int, timeout: int = 900) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = next((j for j in get_json(f"{base}/api/jobs?limit=50")["items"] if j["id"] == job_id), None)
        if job and job["status"] in ("done", "failed", "cancelled", "interrupted"):
            return job
        time.sleep(2)
    return {"status": "timeout"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--out", default=str(REPO / "data" / "ui_smoke"))
    args = ap.parse_args()

    import fitz
    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="latin_e2e_"))
    pdf = tmp / "e2e_latin.pdf"
    doc = fitz.open()
    doc.new_page().insert_textbox(fitz.Rect(60, 80, 540, 400), TEXT, fontsize=18, fontname="tiro")
    doc.save(str(pdf))

    sys.path.insert(0, str(REPO))
    from core.store import Store
    Store(str(tmp / "corpus.db")).close()                         # an empty library with the schema

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ, LATIN_CORPUS_DB=str(tmp / "corpus.db"), LATIN_JOBS_DB=str(tmp / "jobs.db"),
               LATIN_JOB_LOGS=str(tmp / "logs"), LATIN_INDEX=str(tmp / "index.faiss"),
               LATIN_AUDIT_CACHE=str(tmp / "audit.json"), PYTHONIOENCODING="utf-8")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "web.server:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO), env=env, stdout=open(tmp / "server.log", "w"), stderr=subprocess.STDOUT)

    results, problems = [], []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)

    finished = False
    try:
        for _ in range(60):
            try:
                get_json(f"{base}/api/facets")
                break
            except Exception:                                   # noqa: BLE001
                time.sleep(1)
        else:
            print("scratch server did not start; see", tmp / "server.log")
            return 2
        check("scratch server up (own corpus.db)", get_json(f"{base}/api/facets")["totals"]["documents"] == 0)

        with sync_playwright() as p:
            browser = None
            for channel in ("chrome", "msedge", None):
                try:
                    browser = p.chromium.launch(channel=channel, headless=not args.show)
                    break
                except Exception:                               # noqa: BLE001
                    continue
            if browser is None:
                print("No browser found.")
                return 2
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
            page.on("console", lambda m: problems.append(f"console: {m.text}") if m.type == "error" else None)
            page.goto(base)
            page.wait_for_selector("button[data-view=ocr]", timeout=20000)
            page.wait_for_selector("h2:has-text('Documents')", timeout=30000)

            # --- queue the ingest from the OCR page
            page.click("button[data-view=ocr]")
            page.wait_for_selector("h3:has-text('Ingest a file you downloaded')", timeout=20000)
            page.fill("input[data-ocr-path]", str(pdf))
            page.fill("input[data-ocr-title]", "E2E scan test")
            page.select_option("select[data-opt=ocr]", "print")
            page.click("button[data-ocr-ingest]")
            page.wait_for_selector("text=Queued #", timeout=10000)
            job_id = int(page.locator("text=Queued #").first.inner_text().split("#")[1].split(":")[0])
            job = wait_job(base, job_id)
            check("OCR ingest job finished", job["status"] == "done", job["status"])
            if job["status"] != "done":
                print((tmp / "logs").exists() and "log in " + str(tmp / "logs"))
                return 1

            # --- the document shows up under scans, with a Pages button
            page.reload()
            page.wait_for_selector("h2:has-text('Documents')", timeout=30000)
            page.click("button[data-view=ocr]")
            page.wait_for_selector("h3:has-text('Documents made from scans')", timeout=20000)
            page.wait_for_selector("button[data-scan-open]", timeout=20000)
            row = page.locator("table.grid tr", has_text="E2E scan test").first
            check("document listed under scans", row.count() == 1)
            check("source stamped with engine", page.locator(".badge.queued", has_text="tesseract").count()
                  + page.locator(".badge.queued", has_text="kraken-print").count() >= 1)

            # --- scan viewer: image + transcription side by side
            row.locator("button[data-scan-open]").click()
            page.wait_for_selector("#scan-text", timeout=20000)
            page.wait_for_function("document.querySelector('#scan-img') && document.querySelector('#scan-img').naturalWidth > 0",
                                   timeout=20000)
            check("page image loads", True)
            text = page.input_value("#scan-text")
            check("transcription beside the image", "Fides" in text and "natura" in text, text[:60])
            page.screenshot(path=str(out / "e2e_1_viewer.png"), full_page=True)

            # --- doubtful words (Tesseract word confidences)
            page.click("button[data-scan-analyze]")
            page.wait_for_selector(".checks", timeout=60000)
            check("'Find doubtful words' analyses a printed page", "words" in page.inner_text(".checks"),
                  page.inner_text(".checks")[:60].replace(chr(10), " "))

            # --- edit + save with Ctrl+S
            page.fill("#scan-text", text + "\n" + EDIT)
            check("unsaved state shown", "unsaved" in page.inner_text("#scan-state"))
            page.keyboard.press("Control+s")
            page.wait_for_selector("#scan-state:has-text('corrected')", timeout=10000)
            check("Ctrl+S saves a correction", True)
            check("page marked as corrected in selector", "✎" in page.locator("select[data-scan-page] option:checked").inner_text())
            page.screenshot(path=str(out / "e2e_2_corrected.png"), full_page=True)

            # --- correction survives a reload (stored on disk, not in the page)
            page.reload()
            page.wait_for_selector("h2:has-text('Documents')", timeout=30000)
            page.click("button[data-view=ocr]")
            page.wait_for_selector("button[data-scan-open]", timeout=20000)
            page.locator("table.grid tr", has_text="E2E scan test").first.locator("button[data-scan-open]").click()
            page.wait_for_selector("#scan-text", timeout=20000)
            check("correction persists after reload", EDIT in page.input_value("#scan-text"))

            # --- revert removes it
            page.click("button[data-scan-revert]")
            try:
                page.wait_for_selector("#scan-state:has-text('original')", timeout=10000)
            except Exception:                                   # noqa: BLE001
                page.screenshot(path=str(out / "e2e_fail_revert.png"), full_page=True)
                print("state:", page.inner_text("#scan-state"), "| problems:", problems[:3])
                raise
            check("revert restores original", EDIT not in page.input_value("#scan-text"))
            page.fill("#scan-text", page.input_value("#scan-text") + "\n" + EDIT)
            page.keyboard.press("Control+s")
            page.wait_for_selector("#scan-state:has-text('corrected')", timeout=10000)

            # --- re-ingest with corrections -> a corrected copy
            page.wait_for_selector("button[data-scan-reingest]", timeout=10000)
            page.click("button[data-scan-reingest]")
            page.wait_for_selector("text=Queued #", timeout=10000)
            job2 = int(page.locator("text=Queued #").last.inner_text().split("#")[1].split(":")[0])
            check("re-ingest job finished", wait_job(base, job2)["status"] == "done")
            docs = get_json(f"{base}/api/documents?q=corrected&limit=20")["items"]
            fixed = next((d for d in docs if "(corrected)" in d["title"]), None)
            check("corrected copy exists", fixed is not None)
            if fixed:
                segs = get_json(f"{base}/api/documents/{fixed['id']}/segments?limit=100")["items"]
                joined = " ".join(s["latin"] for s in segs)
                check("corrected copy contains the edit", EDIT in joined, joined[-60:])
                check("source records the corrections", "corrected: 1" in (fixed.get("source") or ""),
                      (fixed.get("source") or "")[-50:])

            # --- handwriting page with recorded confidences (St Gall 195 fixture, if present)
            fixture = REPO / "data" / "raw" / "iiif_f92189c0dd"
            lines_file = fixture / "htr_manicule-2026-latin_medieval.lines.json"
            if lines_file.exists():
                import sqlite3
                con = sqlite3.connect(str(tmp / "corpus.db"))
                con.execute("INSERT INTO documents(title, language, source) VALUES (?, 'la', ?)",
                            ("Fixture: St Gall 195",
                             "IIIF manifest x [OCR: htr] [scan: iiif_f92189c0dd|htr_manicule-2026-latin_medieval.json]"))
                con.commit()
                con.close()
                page.reload()
                page.wait_for_selector("h2:has-text('Documents')", timeout=30000)
                page.click("button[data-view=ocr]")
                page.wait_for_selector("button[data-scan-open]", timeout=20000)
                page.locator("table.grid tr", has_text="Fixture: St Gall").first.locator("button[data-scan-open]").click()
                page.wait_for_selector("#scan-text", timeout=20000)
                page.select_option("select[data-scan-page]", "page_0020.jpg")
                page.wait_for_selector(".checks", timeout=20000)
                page.wait_for_function("document.querySelectorAll('#scan-boxes .sbox').length > 0", timeout=20000)
                n_boxes = page.locator("#scan-boxes .sbox").count()
                check("handwriting page shows flagged words on the image", n_boxes >= 1, f"{n_boxes} boxes")
                page.locator(".chk").first.click()
                page.wait_for_selector("#scan-boxes .sbox.hot", timeout=5000)
                check("clicking a doubtful line highlights it on the image", True)
                page.screenshot(path=str(out / "e2e_4_doubtful.png"), full_page=True)
            else:
                print("  SKIP  handwriting fixture not present on this machine")

            # --- audit button
            page.click("button[data-view=ocr]")
            page.wait_for_selector("button[data-audit-run]", timeout=20000)
            page.click("button[data-audit-run]")
            page.wait_for_selector("text=Audit finished", timeout=120000)
            check("audit re-run completes", page.locator("h3:has-text('Library quality audit')").count() == 1)
            page.screenshot(path=str(out / "e2e_3_ocr_page.png"), full_page=True)

            check("no browser errors", not problems, "; ".join(problems)[:200])
            browser.close()

        finished = True
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        # the work dir for our temp PDF lives under data/raw/pdf_<hash of its path>
        import hashlib
        h = hashlib.sha1(str(pdf.resolve()).encode()).hexdigest()[:10]
        shutil.rmtree(REPO / "data" / "raw" / f"pdf_{h}", ignore_errors=True)
        if finished and all(ok for _, ok in results):
            shutil.rmtree(tmp, ignore_errors=True)

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed. Screenshots: {out}")
    if failed:
        print("Scratch files kept in", tmp)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
