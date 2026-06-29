"""One-time discovery crawl for Musa Medievalis (medieval Latin poetry, ~650-1250).

The site (musamedievalis.it, himeros JS app) renders its author list and per-author
work lists with JavaScript, so we use Playwright ONCE to build a static manifest of
work codes; the runtime connector (ingest/musamedievalis.py) then fetches each text
with plain requests (the text pages ARE server-rendered). This mirrors how the EDCS
connector was bootstrapped.

Flow: for each letter index page, read author rows (<tr id="authorsN"> with a name
and a date span), keep those whose date falls in the target window, click each to
populate its works panel, and collect the /testo/testo/codice/<CODE> links.

Output: data/musamedievalis_catalog.json — [{code, author, dates, century, title}].

Usage:
    python scripts/crawl_musamedievalis.py --max-year 1050           # full crawl
    python scripts/crawl_musamedievalis.py --letters a --max-year 3000  # test (all)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from playwright.sync_api import sync_playwright

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "musamedievalis_catalog.json")
LETTERS = "abcdefghijklmnopqrstuvwxyz"
INDEX = "https://www.musamedievalis.it/indici/autori/tipo/alpha/lettera/{L}"

_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}


def roman(s: str) -> int:
    s = s.lower().strip()
    if not s or any(c not in _ROMAN for c in s):
        return 0
    tot = 0
    for i, c in enumerate(s):
        v = _ROMAN[c]
        nxt = _ROMAN.get(s[i + 1]) if i + 1 < len(s) else 0
        tot += -v if (nxt and v < nxt) else v
    return tot


def earliest_century(dates: str):
    """Best-guess earliest century of activity from a date string like
    '940/45 - 1004', 'saec. IX', '† 939/40', 'saec. VII - VIII', '716 - 757'."""
    d = dates.lower()
    secs = [roman(r) for r in re.findall(r"saec\.?\s*([ivxl]+)", d)]
    secs = [c for c in secs if c]
    if secs:
        return min(secs)
    years = [int(y) for y in re.findall(r"\b(\d{3,4})\b", d)]
    if years:
        return (min(years) // 100) + 1
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-year", type=int, default=1050,
                    help="keep authors whose earliest activity is <= this year")
    ap.add_argument("--letters", default=LETTERS, help="letters to crawl (default a-z)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    max_cent = args.max_year // 100  # e.g. 1050 -> keep centuries <= 10

    catalog = []
    seen_codes = set()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for L in args.letters:
            page.goto(INDEX.format(L=L), wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(900)
            rows = page.eval_on_selector_all(
                'tr[id^="authors"]',
                """els => els.map(e => ({
                    id: e.id.replace('authors',''),
                    name: (e.querySelector('b') ? e.querySelector('b').innerText : '').trim(),
                    dates: (e.querySelector('.data') ? e.querySelector('.data').innerText : '').trim()
                }))""")
            kept = []
            for r in rows:
                cent = earliest_century(r["dates"])
                if cent is not None and cent <= max_cent:
                    r["century"] = cent
                    kept.append(r)
            print(f"[{L}] {len(rows)} authors, {len(kept)} in window", flush=True)

            for a in kept:
                try:
                    page.click(f'#authors{a["id"]}', timeout=8000)
                    page.wait_for_timeout(700)
                    works = page.eval_on_selector_all(
                        'a[href*="codice"]',
                        """els => els.map(e => ({
                            href: e.getAttribute('href'),
                            title: (e.innerText || '').trim()
                        }))""")
                except Exception as exc:
                    print(f"    ! {a['name']}: {str(exc)[:50]}", flush=True)
                    continue
                for w in works:
                    m = re.search(r"codice/(.+)$", w["href"])
                    if not m:
                        continue
                    code = m.group(1)
                    if code in seen_codes:
                        continue
                    seen_codes.add(code)
                    catalog.append({
                        "code": code, "author": a["name"], "dates": a["dates"],
                        "century": a["century"], "title": w["title"] or "carmina",
                    })
        browser.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=1)
    print(f"\nWrote {len(catalog)} works ({len(set(c['author'] for c in catalog))} authors) "
          f"to {args.out}")


if __name__ == "__main__":
    main()
