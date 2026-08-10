"""Repair long-s (ſ misread as f) OCR corruption in a document's latin_text.

Builds a reference vocabulary from every *other* document in the corpus, then
runs ingest.ocr_fix.correct_long_s over each segment of the target document.

Dry-run by default: prints a change/unresolved summary and writes a full log,
but does not touch the database. Pass --apply to write corrected latin_text
back and null out stale english_text/embed_text/etc. for changed segments.

Usage:
    python scripts/fix_long_s_ocr.py --doc-id 376
    python scripts/fix_long_s_ocr.py --doc-id 376 --apply
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.store import Store
from ingest.ocr_fix import build_reference_vocab, build_prefix_index, correct_long_s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-id", type=int, required=True)
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("--apply", action="store_true", help="write changes to the DB (default: dry run)")
    ap.add_argument("--log", default=None, help="path to write the full change log (default: data/clean/<doc>_ocr_fix_log.txt)")
    args = ap.parse_args()

    store = Store(args.db)
    doc = store.get_document(args.doc_id)
    if doc is None:
        print(f"No document with id {args.doc_id}")
        return
    print(f"=== Long-s OCR fix: doc {args.doc_id} — {doc.title!r} ({doc.author}) ===")

    print("Building reference vocabulary from the rest of the corpus...")
    vocab = build_reference_vocab(store, exclude_doc_ids=[args.doc_id])
    prefixes = build_prefix_index(vocab)
    print(f"  vocab: {len(vocab):,} distinct forms")

    rows = store.conn.execute(
        """SELECT s.id, s.latin_text FROM segments s
           JOIN sections sec ON s.section_id = sec.id
           WHERE sec.doc_id = ? ORDER BY sec.ord, s.ord""",
        (args.doc_id,),
    ).fetchall()
    print(f"  segments: {len(rows):,}")

    change_pairs = Counter()      # (original, corrected) -> count -- auto-applied, exact vocab match
    suggestion_pairs = Counter()  # (original, corrected) -> count -- NOT applied, needs manual review
    unresolved = Counter()        # token -> count -- nothing found at all
    to_write = []                 # (segment_id, new_text)
    n_segments_changed = 0

    for seg_id, text in rows:
        if not text:
            continue
        result = correct_long_s(text, vocab, prefixes)
        for c in result.changes:
            change_pairs[(c.original, c.corrected)] += 1
        for c in result.suggestions:
            suggestion_pairs[(c.original, c.corrected)] += 1
        for u in result.unresolved:
            unresolved[u] += 1
        if result.changes:
            n_segments_changed += 1
            to_write.append((seg_id, result.text))

    total_changes = sum(change_pairs.values())
    total_suggestions = sum(suggestion_pairs.values())
    total_unresolved = sum(unresolved.values())
    print(f"\nSegments changed: {n_segments_changed:,}/{len(rows):,}")
    print(f"Applied fixes (exact vocab match): {total_changes:,} ({len(change_pairs):,} distinct)")
    print(f"Low-confidence suggestions (NOT applied, stem match only): "
          f"{total_suggestions:,} ({len(suggestion_pairs):,} distinct)")
    print(f"Fully unresolved f-tokens left untouched: {total_unresolved:,} ({len(unresolved):,} distinct)")

    print("\n--- Top 40 applied corrections by frequency ---")
    for (orig, fixed), n in change_pairs.most_common(40):
        print(f"  {n:>5}x  {orig!r:35} -> {fixed!r}")

    print("\n--- Top 40 low-confidence suggestions (NOT applied) ---")
    for (orig, fixed), n in suggestion_pairs.most_common(40):
        print(f"  {n:>5}x  {orig!r:35} -> {fixed!r}")

    print("\n--- Top 40 unresolved tokens by frequency ---")
    for tok, n in unresolved.most_common(40):
        print(f"  {n:>5}x  {tok!r}")

    log_path = args.log or f"data/clean/doc{args.doc_id}_ocr_fix_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Long-s OCR fix log for doc {args.doc_id} ({doc.title})\n")
        f.write(f"Segments changed: {n_segments_changed}/{len(rows)}\n")
        f.write(f"Applied fixes: {total_changes} ({len(change_pairs)} distinct)\n")
        f.write(f"Low-confidence suggestions (not applied): {total_suggestions} ({len(suggestion_pairs)} distinct)\n")
        f.write(f"Unresolved: {total_unresolved} ({len(unresolved)} distinct)\n\n")
        f.write("=== Applied corrections (original -> corrected, count) ===\n")
        for (orig, fixed), n in change_pairs.most_common():
            f.write(f"{n}\t{orig}\t->\t{fixed}\n")
        f.write("\n=== Low-confidence suggestions -- NOT applied, review manually ===\n")
        for (orig, fixed), n in suggestion_pairs.most_common():
            f.write(f"{n}\t{orig}\t->\t{fixed}\n")
        f.write("\n=== All unresolved tokens (count) ===\n")
        for tok, n in unresolved.most_common():
            f.write(f"{n}\t{tok}\n")
    print(f"\nFull log written to {log_path}")

    if args.apply:
        print(f"\nApplying {len(to_write):,} segment updates and clearing stale translations...")
        store.set_latin_texts(to_write, reset_translation=True)
        print("Done. Re-run scripts/translate_pending.py to regenerate translations for the changed segments.")
    else:
        print("\nDry run only -- no database changes made. Re-run with --apply to write them.")


if __name__ == "__main__":
    main()
