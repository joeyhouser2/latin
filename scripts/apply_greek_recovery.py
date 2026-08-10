"""Apply recovered text (from re-OCR'd page images) back to garbled segments.

Dry-run by default. Only touches segments whose english_text is currently
the untranslatable placeholder (see ingest/garble_detect.py) -- recovered
text still needs to go through the existing long-s/ct correction tiers and
then be retranslated (with mixed Latin/Greek routing), which are separate
follow-up steps.

Usage:
    python scripts/apply_greek_recovery.py --doc-id 376
    python scripts/apply_greek_recovery.py --doc-id 376 --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.store import Store
from ingest.garble_detect import UNTRANSLATABLE_PLACEHOLDER
from ingest.page_splice import recover_segment


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-id", type=int, required=True)
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("--orig-db", default="data/corpus.db.bak-preOcrFix-20260720201447",
                     help="pre-correction snapshot -- needed to match against the PDF's own (uncorrected) OCR layer")
    ap.add_argument("--scan-dir", default="data/raw/bub_gb_dSihnyKx6hgC_scan")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    store = Store(args.db)
    orig_store = Store(args.orig_db)

    with open(os.path.join(args.scan_dir, "page_texts.json"), encoding="utf-8") as f:
        old_pages = json.load(f)
    with open(os.path.join(args.scan_dir, "page_ocr_grclat.json"), encoding="utf-8") as f:
        new_pages = json.load(f)
    with open(os.path.join(args.scan_dir, "segment_pages.json"), encoding="utf-8") as f:
        seg_pages = json.load(f)

    rows = store.conn.execute(
        """SELECT s.id FROM segments s JOIN sections sec ON s.section_id = sec.id
           WHERE sec.doc_id = ? AND s.english_text = ?""",
        (args.doc_id, UNTRANSLATABLE_PLACEHOLDER),
    ).fetchall()
    garbled_ids = [r[0] for r in rows]
    print(f"=== Greek recovery for doc {args.doc_id}: {len(garbled_ids)} garbled segments ===")

    orig_texts = dict(orig_store.conn.execute(
        """SELECT s.id, s.latin_text FROM segments s JOIN sections sec ON s.section_id = sec.id
           WHERE sec.doc_id = ?""", (args.doc_id,)
    ).fetchall())

    recovered_count = 0
    to_write = []
    samples = []
    for sid in garbled_ids:
        page = seg_pages.get(str(sid), [None, False])[0]
        if page is None or str(page) not in new_pages:
            continue
        seg_text = orig_texts.get(sid)
        if not seg_text:
            continue
        recovered = recover_segment(old_pages[page], new_pages[str(page)], seg_text)
        if recovered:
            recovered_count += 1
            to_write.append((sid, recovered))
            if len(samples) < 25:
                samples.append((sid, page, seg_text, recovered))

    print(f"Pages re-OCR'd: {len(new_pages)}")
    print(f"Segments recovered: {recovered_count}/{len(garbled_ids)}")
    print("\n--- Samples ---")
    for sid, page, old, new in samples:
        print(f"[{sid}] page {page}")
        print("  OLD:", old[:120])
        print("  NEW:", new[:200].replace("\n", " "))
        print()

    if args.apply:
        store.conn.executemany(
            "UPDATE segments SET latin_text = ? WHERE id = ?",
            [(rec, sid) for sid, rec in to_write],
        )
        store.conn.executemany(
            "UPDATE segments SET english_text = NULL, english_styled = NULL, "
            "style_label = NULL, embed_text = NULL, scansion = NULL WHERE id = ?",
            [(sid,) for sid, _ in to_write],
        )
        store.conn.commit()
        print(f"\nApplied: {len(to_write)} segments updated, translations cleared for re-processing.")
    else:
        print("\nDry run only. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
