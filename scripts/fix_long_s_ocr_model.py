"""Second pass: apply the trained long-s context model to whatever
scripts/fix_long_s_ocr.py's dictionary tiers left behind (still-unresolved
tokens and the low-confidence "suggestions" it declined to apply).

Dry-run by default -- prints a change/decline summary and writes a log, but
does not touch the database. Pass --apply to write corrected latin_text back
(and null stale english_text/etc. for changed segments, same as the
dictionary-tier script).

Usage:
    python scripts/fix_long_s_ocr_model.py --doc-id 376
    python scripts/fix_long_s_ocr_model.py --doc-id 376 --apply
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.store import Store
from ingest.ocr_fix import build_reference_vocab, build_prefix_index
from ingest.ocr_fix_model import load_model, apply_model_tier, MODEL_PATH


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-id", type=int, required=True)
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--high", type=float, default=0.9, help="P(was-s) >= this -> flip to s")
    ap.add_argument("--low", type=float, default=0.1, help="P(was-s) <= this -> keep as f")
    ap.add_argument("--apply", action="store_true", help="write changes to the DB (default: dry run)")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    store = Store(args.db)
    doc = store.get_document(args.doc_id)
    if doc is None:
        print(f"No document with id {args.doc_id}")
        return
    print(f"=== Model-based long-s fix: doc {args.doc_id} — {doc.title!r} ({doc.author}) ===")
    print(f"Loading model from {args.model} (high={args.high}, low={args.low})...")
    pipeline = load_model(args.model)
    print("Building reference vocabulary (for multi-f corroboration)...")
    vocab = build_reference_vocab(store, exclude_doc_ids=[args.doc_id])
    vocab_prefixes = build_prefix_index(vocab)

    rows = store.conn.execute(
        """SELECT s.id, s.latin_text FROM segments s
           JOIN sections sec ON s.section_id = sec.id
           WHERE sec.doc_id = ? ORDER BY sec.ord, s.ord""",
        (args.doc_id,),
    ).fetchall()
    print(f"  segments: {len(rows):,}")

    change_pairs = Counter()
    unresolved = Counter()
    to_write = []
    n_segments_changed = 0

    for seg_id, text in rows:
        if not text:
            continue
        result = apply_model_tier(text, pipeline, args.high, args.low, vocab, vocab_prefixes)
        for c in result.changes:
            change_pairs[(c.original, c.corrected)] += 1
        for u in result.still_unresolved:
            unresolved[u] += 1
        if result.changes:
            n_segments_changed += 1
            to_write.append((seg_id, result.text))

    total_changes = sum(change_pairs.values())
    total_unresolved = sum(unresolved.values())
    print(f"\nSegments changed: {n_segments_changed:,}/{len(rows):,}")
    print(f"Model-applied fixes: {total_changes:,} ({len(change_pairs):,} distinct)")
    print(f"Still unresolved (ambiguous or no confident call): {total_unresolved:,} ({len(unresolved):,} distinct)")

    print("\n--- Top 50 model-applied corrections by frequency ---")
    for (orig, fixed), n in change_pairs.most_common(50):
        print(f"  {n:>5}x  {orig!r:35} -> {fixed!r}")

    print("\n--- Top 40 still-unresolved tokens by frequency ---")
    for tok, n in unresolved.most_common(40):
        print(f"  {n:>5}x  {tok!r}")

    log_path = args.log or f"data/clean/doc{args.doc_id}_ocr_fix_model_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Model-based long-s fix log for doc {args.doc_id} ({doc.title})\n")
        f.write(f"high={args.high} low={args.low}\n")
        f.write(f"Segments changed: {n_segments_changed}/{len(rows)}\n")
        f.write(f"Applied fixes: {total_changes} ({len(change_pairs)} distinct)\n")
        f.write(f"Still unresolved: {total_unresolved} ({len(unresolved)} distinct)\n\n")
        f.write("=== All model-applied corrections (original -> corrected, count) ===\n")
        for (orig, fixed), n in change_pairs.most_common():
            f.write(f"{n}\t{orig}\t->\t{fixed}\n")
        f.write("\n=== All still-unresolved tokens (count) ===\n")
        for tok, n in unresolved.most_common():
            f.write(f"{n}\t{tok}\n")
    print(f"\nFull log written to {log_path}")

    if args.apply:
        print(f"\nApplying {len(to_write):,} segment updates and clearing stale translations...")
        store.set_latin_texts(to_write, reset_translation=True)
        print("Done.")
    else:
        print("\nDry run only -- no database changes made. Re-run with --apply to write them.")


if __name__ == "__main__":
    main()
