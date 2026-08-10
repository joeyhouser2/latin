"""Scan every document in the corpus for long-s (ſ misread as f) OCR corruption.

Uses ingest.ocr_fix.long_s_score: counts occurrences of a handful of common
Latin function words in both their clean form ("est", "sed", "si ", ...) and
their unambiguous long-s-corrupted form ("eft", "fed", "fi ", ...). A clean
digital edition should show ~0 corrupted hits against many clean hits; a
document with pervasive long-s corruption shows corrupted hits comparable to
(or exceeding) clean hits.

This is read-only and safe to run any time -- it doesn't touch the database.

Usage:
    python scripts/check_ocr_corruption.py
    python scripts/check_ocr_corruption.py --min-signature-hits 5
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.store import Store
from ingest.ocr_fix import long_s_score


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("--min-signature-hits", type=int, default=5,
                     help="ignore documents with fewer than this many total signature-pair "
                          "hits (clean+corrupted) -- too little text to judge reliably")
    ap.add_argument("--threshold", type=float, default=0.15,
                     help="flag documents where corrupted / (corrupted+clean) exceeds this")
    args = ap.parse_args()

    store = Store(args.db)
    docs = store.list_documents()
    print(f"=== Scanning {len(docs)} documents for long-s OCR corruption ===\n")

    results = []
    for d in docs:
        rows = store.conn.execute(
            """SELECT s.latin_text FROM segments s
               JOIN sections sec ON s.section_id = sec.id
               WHERE sec.doc_id = ?""",
            (d.id,),
        ).fetchall()
        corrupted = clean = 0
        for (text,) in rows:
            if not text:
                continue
            c, k = long_s_score(text)
            corrupted += c
            clean += k
        total = corrupted + clean
        if total < args.min_signature_hits:
            continue
        ratio = corrupted / total
        results.append((ratio, corrupted, clean, d))

    results.sort(key=lambda r: r[0], reverse=True)
    flagged = [r for r in results if r[0] > args.threshold]

    print(f"{'ratio':>6}  {'bad':>6}  {'good':>6}  id     source / title")
    print("-" * 100)
    for ratio, corrupted, clean, d in results[:40]:
        flag = " <-- SUSPECT" if ratio > args.threshold else ""
        src = (d.source or "")[:40]
        print(f"{ratio:>5.0%}  {corrupted:>6}  {clean:>6}  {d.id:<5}  {src:40} {d.title[:40]}{flag}")

    print(f"\n{len(flagged)} document(s) above {args.threshold:.0%} corruption ratio "
          f"out of {len(results)} scored (threshold min {args.min_signature_hits} signature hits).")
    if flagged:
        print("\nFlagged documents:")
        for ratio, corrupted, clean, d in flagged:
            print(f"  [{d.id}] {d.title} ({d.author}) -- {d.source} "
                  f"-- {ratio:.0%} corrupted ({corrupted} bad / {clean} good signature hits)")
    else:
        print("\nNo other documents show significant long-s corruption.")


if __name__ == "__main__":
    main()
