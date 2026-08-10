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
from ingest.garble_detect import is_garbled, UNTRANSLATABLE_PLACEHOLDER
from ingest.mixed_lang_translate import translate_mixed_batch, _GREEK_CHAR


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-prefix", default="",
                    help="only docs whose source starts with this (e.g. 'ALIM (')")
    ap.add_argument("--language", default="", help="only this language (la/grc)")
    ap.add_argument("--skip-translated", action="store_true",
                    help="skip docs already known to have a published English "
                         "translation (translation_status == 'translated')")
    ap.add_argument("--batch-size", type=int, default=16, help="model batch size")
    ap.add_argument("--chunk", type=int, default=200,
                    help="segments per DB commit (resume granularity)")
    ap.add_argument("--max-length", type=int, default=256,
                    help="cap on tokenization/generation length (lower = less VRAM)")
    args = ap.parse_args()

    lib = Library()
    docs = lib.store.list_documents()
    if args.source_prefix:
        docs = [d for d in docs if d.source and d.source.startswith(args.source_prefix)]
    if args.language:
        docs = [d for d in docs if d.language == args.language]
    if args.skip_translated:
        docs = [d for d in docs if d.translation_status != "translated"]

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
        if hasattr(tr, "max_length"):
            tr.max_length = args.max_length
        n = len(pending)
        t0 = time.time()
        done = 0
        for i in range(0, n, args.chunk):
            batch = pending[i:i + args.chunk]
            # Segments whose source is too corrupted to be real Latin (e.g.
            # embedded quotations in a script the original OCR never
            # recognized, garbled into Latin-alphabet-lookalike noise) get an
            # honest placeholder instead of being sent through the
            # translator -- an NMT model doesn't fail loudly on garbage input,
            # it produces fluent, confident, entirely fabricated English,
            # which reads as real content and is worse than admitting the
            # source is unusable. See ingest/garble_detect.py.
            clean = [s for s in batch if not is_garbled(s.latin_text)]
            garbled = [s for s in batch if is_garbled(s.latin_text)]

            # Some clean-enough segments still carry genuine embedded Greek
            # (recovered by ingest/page_splice.py's re-OCR pipeline -- see
            # ingest/mixed_lang_translate.py for why those go through a
            # separate, safer path rather than the plain translator).
            mixed = [s for s in clean if d.language == "la" and _GREEK_CHAR.search(s.latin_text)]
            pure = [s for s in clean if s not in mixed]

            results = []
            if pure:
                eng = tr.translate_batch([s.latin_text for s in pure],
                                         batch_size=args.batch_size)
                results.extend(zip(pure, eng))
            if mixed:
                eng = translate_mixed_batch([s.latin_text for s in mixed], tr,
                                            batch_size=args.batch_size)
                results.extend(zip(mixed, eng))
            results.extend((s, UNTRANSLATABLE_PLACEHOLDER) for s in garbled)
            lib.store.set_translations([(s.id, e) for s, e in results])
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
