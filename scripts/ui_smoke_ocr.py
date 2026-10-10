"""Browser smoke test for the web app's OCR features, with screenshots.

Drives the *running* app in a real Chrome (Playwright), clicks through the OCR
page, the Find-texts OCR controls and the Jobs page, and saves screenshots you
can open. Safe to run against your live app: the one job it queues is scheduled
a day ahead, then cancelled, so nothing is ingested.

    python scripts/ui_smoke_ocr.py                     # http://127.0.0.1:8000
    python scripts/ui_smoke_ocr.py --url http://127.0.0.1:9000 --show   # watch it

Output: data/ui_smoke/*.png and a pass/fail line per check. Needs
``pip install playwright`` and Chrome (or Edge) installed.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")


def make_pdf(path: Path) -> None:
    """A two-page PDF of typeset Latin, so the path field has something real."""
    import fitz
    doc = fitz.open()
    for text in ("Fides quaerens intellectum est. Omnes homines natura scire desiderant, "
                 "et ea quae sunt in rerum natura cognoscere.",
                 "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, "
                 "aliam Aquitani, tertiam qui ipsorum lingua Celtae appellantur."):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(60, 80, 540, 400), text, fontsize=18, fontname="tiro")
    doc.save(str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--show", action="store_true", help="open a visible browser window")
    ap.add_argument("--out", default=str(REPO / "data" / "ui_smoke"))
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / "smoke_latin.pdf"
    make_pdf(pdf)

    results, problems = [], []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    with sync_playwright() as p:
        browser = None
        for channel in ("chrome", "msedge", None):
            try:
                browser = p.chromium.launch(channel=channel, headless=not args.show)
                break
            except Exception:                                   # noqa: BLE001
                continue
        if browser is None:
            print("No browser found: install Chrome or Edge, or run `playwright install chromium`.")
            return 2
        page = browser.new_page(viewport={"width": 1300, "height": 1000})
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.on("console", lambda m: problems.append(f"console: {m.text}") if m.type == "error" else None)

        page.goto(args.url)
        page.wait_for_selector("nav button[data-view=ocr], button[data-view=ocr]", timeout=15000)
        # let boot() finish: it ends by opening Documents, which would override a click
        page.wait_for_selector("h2:has-text('Documents')", timeout=30000)
        check("OCR tab present", page.locator("button[data-view=ocr]").count() == 1)

        # --- OCR page
        page.click("button[data-view=ocr]")
        page.wait_for_selector("h3:has-text(\"Known gaps\")", timeout=20000)
        page.screenshot(path=str(out / "1_ocr_page.png"), full_page=True)
        headings = page.locator("h3").all_inner_texts()
        for want in ("Engines", "Ingest a file you downloaded", "Known gaps",
                     "Sources to download by hand"):
            check(f"section: {want}", any(want in h for h in headings))
        check("Tesseract reported ready", page.locator("text=Tesseract").first.is_visible()
              and page.locator(".badge.done", has_text="ready").count() >= 1)
        check("gap cards rendered", page.locator(".card", has_text="Next:").count() >= 8,
              f"{page.locator('.card', has_text='Next:').count()} cards")
        check("manual sources listed", page.locator("table.grid tbody tr", has_text="HathiTrust").count() == 1)
        check("scan-made documents listed",
              page.locator("h3", has_text="Documents made from scans").count() == 1)

        # --- queue a scheduled (never-running) ingest from a PDF path
        page.fill("input[data-ocr-path]", str(pdf))
        page.fill("input[data-ocr-title]", "UI smoke test (cancel me)")
        page.select_option("select[data-opt=ocr]", "print")
        page.fill("input[data-opt=pages]", "1-2")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M")
        page.fill("input[data-opt=not_before]", tomorrow)
        page.dispatch_event("input[data-opt=not_before]", "change")
        page.click("button[data-ocr-ingest]")
        page.wait_for_selector("text=Queued #", timeout=10000)
        toast = page.locator("text=Queued #").first.inner_text()
        check("job queued from the OCR page", "ocrimages" in toast, toast[:90])
        page.screenshot(path=str(out / "2_queued_toast.png"))

        # --- Jobs page shows it queued with the OCR options, then cancel it
        page.click("button[data-view=jobs]")
        page.wait_for_selector("#jobs-table", timeout=10000)
        row = page.locator("#jobs-table tr", has_text="UI smoke test").first
        row = row if row.count() else page.locator("#jobs-table tr", has_text="ocrimages").first
        check("job visible on Jobs page", row.count() == 1)
        if row.count():
            txt = row.inner_text()
            check("job label carries OCR engine and pages", "[OCR: print]" in txt and "1-2" in txt, txt[:100])
            page.screenshot(path=str(out / "3_jobs_queued.png"), full_page=True)
            cancel = row.locator("[data-cancel]")
            if cancel.count():
                cancel.first.click()
                page.wait_for_timeout(2500)
                page.reload()
                page.wait_for_selector("h2:has-text('Documents')", timeout=30000)
                page.click("button[data-view=jobs]")
                page.wait_for_selector("#jobs-table", timeout=10000)
                still = page.locator("#jobs-table tr", has_text="UI smoke test").locator(".badge.queued")
                check("test job cancelled", still.count() == 0)

        # --- Find texts: OCR controls + a scan catalogue with OCR badges
        page.click("button[data-view=catalog]")
        page.wait_for_selector("select[data-opt=ocr]", timeout=10000)
        check("OCR engine selector on Find texts", page.locator("select[data-opt=ocr] option").count() == 4)
        check("ingest-by-identifier box", page.locator("input[data-ident]").count() == 1)
        page.select_option("select[data-cat-source]", "vd")
        page.fill("input[data-cat-query]", "usuris")
        page.fill("input[data-cat-limit]", "12")
        page.click("button[data-cat-go]")
        page.wait_for_selector("table.grid tbody tr .badge", timeout=60000)
        badges = page.locator("table.grid tbody tr td .badge").all_inner_texts()
        check("catalogue rows carry OCR badges",
              any("scan" in b for b in badges), ", ".join(sorted(set(badges)))[:90])
        page.screenshot(path=str(out / "4_find_texts_vd.png"), full_page=True)

        check("no browser errors", not problems, "; ".join(problems)[:200])
        browser.close()

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed. Screenshots: {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
