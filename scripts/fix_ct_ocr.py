"""Fix this document's c-before-t-misread-as-f OCR error (contraftus ->
contractus, diftum -> dictum, Leftor -> Lector). See ingest/ocr_fix_ct.py.

Dry-run by default. Usage:
    python scripts/fix_ct_ocr.py --doc-id 376
    python scripts/fix_ct_ocr.py --doc-id 376 --apply
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.store import Store
from ingest.ocr_fix import build_reference_vocab, build_prefix_index
from ingest.ocr_fix_model import load_model
from ingest.ocr_fix_ct import apply_ct_tier


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-id", type=int, required=True)
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    store = Store(args.db)
    doc = store.get_document(args.doc_id)
    print(f"=== ct-misread fix: doc {args.doc_id} — {doc.title!r} ({doc.author}) ===")
    vocab = build_reference_vocab(store, exclude_doc_ids=[args.doc_id])
    prefixes = build_prefix_index(vocab)
    pipeline = load_model()

    rows = store.conn.execute(
        """SELECT s.id, s.latin_text FROM segments s
           JOIN sections sec ON s.section_id = sec.id
           WHERE sec.doc_id = ? ORDER BY sec.ord, s.ord""",
        (args.doc_id,),
    ).fetchall()
    print(f"  segments: {len(rows):,}")

    change_pairs = Counter()
    to_write = []
    n_changed = 0
    for seg_id, text in rows:
        if not text:
            continue
        result = apply_ct_tier(text, vocab, prefixes, pipeline)
        for c in result.changes:
            change_pairs[(c.original, c.corrected)] += 1
        if result.changes:
            n_changed += 1
            to_write.append((seg_id, result.text))

    total = sum(change_pairs.values())
    print(f"\nSegments changed: {n_changed:,}/{len(rows):,}")
    print(f"Fixes: {total:,} ({len(change_pairs):,} distinct)")
    print("\n--- All corrections by frequency ---")
    for (orig, fixed), n in change_pairs.most_common(200):
        print(f"  {n:>4}x  {orig!r:25} -> {fixed!r}")

    if args.apply:
        print(f"\nApplying {len(to_write):,} segment updates and clearing stale translations...")
        store.set_latin_texts(to_write, reset_translation=True)
        print("Done.")
    else:
        print("\nDry run only. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
