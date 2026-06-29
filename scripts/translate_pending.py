"""Translate untranslated segments for library documents matching a filter,
using the pipeline's per-(language,stage) translator routing.

Resumable: only untranslated segments are touched, and translations are written
in small chunks, so an interrupted run (even mid-document) loses at most the
current chunk — just re-run to continue.

Usage:
    python scripts/translate_pending.py --source-prefix "ALIM ("
    python scripts/translate_pending.py --language la --chunk 200 --batch-size 16
    CUDA_VISIBLE_DEVICES=0 python scripts/translate_pending.py --source-prefix "ALIM ("
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import sentence_transformers  # noqa: F401  (import order: see harvest script)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import Library


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-prefix", default="",
                    help="only docs whose source starts with this (e.g. 'ALIM (')")
    ap.add_argument("--language", default="", help="only this language (la/grc)")
    ap.add_argument("--batch-size", type=int, default=16, help="model batch size")
    ap.add_argument("--chunk", type=int, default=200,
                    help="segments per DB commit (resume granularity)")
    args = ap.parse_args()

    lib = Library()
    docs = lib.store.list_documents()
    if args.source_prefix:
        docs = [d for d in docs if d.source and d.source.startswith(args.source_prefix)]
    if args.language:
        docs = [d for d in docs if d.language == args.language]

    # Count work up front.
    plan = []   # (doc, pending_segments)
    for d in docs:
        full = lib.store.get_document(d.id)
        pending = [s for s in full.iter_segments() if not s.is_translated]
        if pending:
            plan.append((d, pending))
    total = sum(len(p) for _, p in plan)
    print(f"=== Translate pending: {len(plan)} docs, {total:,} segments ===\n")

    grand = 0
    t_start = time.time()
    for d, pending in plan:
        tr = lib.translator_for(d.language, d.language_stage)
        n = len(pending)
        t0 = time.time()
        done = 0
        for i in range(0, n, args.chunk):
            batch = pending[i:i + args.chunk]
            eng = tr.translate_batch([s.latin_text for s in batch],
                                     batch_size=args.batch_size)
            lib.store.set_translations([(s.id, e) for s, e in zip(batch, eng)])
            done += len(batch)
            grand += len(batch)
            rate = grand / max(time.time() - t_start, 1e-6)
            eta = (total - grand) / rate / 3600
            print(f"  [{d.id}] {(d.source or '')[:22]:22} {done:>6,}/{n:<6,} "
                  f"| overall {grand:,}/{total:,} {rate:.1f} seg/s ETA {eta:.1f}h",
                  flush=True)
        print(f"  [{d.id}] DONE {n} segs in {time.time()-t0:.0f}s", flush=True)

    print(f"\nDone. Translated {grand:,} segments across {len(plan)} docs "
          f"in {(time.time()-t_start)/3600:.1f}h.")
    lib.close()


if __name__ == "__main__":
    main()
