"""Run the Victorian-prose stylizer over translated segments across the library.

Resumable: only translated-but-not-yet-styled segments are touched, and results
are written to the DB after every passage-batch, so an interrupted run loses at
most the current batch -- just re-run to continue. Passages are chunked (rather
than styling a whole section in one LLM call) so a single request never blows
past the model's context window; on a CUDA OOM the batch backs off (halves,
down to one segment) before giving up on that segment for a later retry.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/stylize_library.py
    python scripts/stylize_library.py --source-prefix "ALIM (" --limit 2
    python scripts/stylize_library.py --doc-id 42 --batch-size 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import Library
from core.stylizer import StyleUnit
from ingest.garble_detect import UNTRANSLATABLE_PLACEHOLDER


def stylize_batch_safe(stylizer, units, preset, context, batch_size):
    """stylize_units with OOM backoff: halve the batch on CUDA OOM, down to a
    single unit; if even one unit OOMs, skip it (left unstyled, retry later)."""
    import torch

    if not units:
        return []
    try:
        return stylizer.stylize_units(units, preset=preset, context=context)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        if len(units) == 1:
            print(f"    ! OOM on a single segment, skipping (id will retry later)",
                  flush=True)
            return [""]
        mid = len(units) // 2
        return (stylize_batch_safe(stylizer, units[:mid], preset, context, batch_size)
                + stylize_batch_safe(stylizer, units[mid:], preset, context, batch_size))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-prefix", default="",
                    help="only docs whose source starts with this (e.g. 'ALIM (')")
    ap.add_argument("--language", default="", help="only this language (la/grc)")
    ap.add_argument("--doc-id", type=int, default=None, help="only this document")
    ap.add_argument("--limit", type=int, default=None, help="stop after N documents")
    ap.add_argument("--skip-poetry", action="store_true",
                    help="exclude genre='poetry' docs (verse needs a verse preset, "
                         "not victorian_prose -- see verse_blank/verse_couplet)")
    ap.add_argument("--preset", default="victorian_prose",
                    choices=["victorian_prose", "verse_blank", "verse_couplet"])
    ap.add_argument("--backend", default="llm", choices=["llm", "t5"])
    ap.add_argument("--batch-size", type=int, default=20,
                    help="segments per passage-level LLM call")
    ap.add_argument("--shard-count", type=int, default=1,
                    help="split the doc queue across N parallel workers (one per GPU)")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="which shard this process handles (0-based)")
    args = ap.parse_args()

    lib = Library()
    docs = lib.store.list_documents()
    if args.doc_id is not None:
        docs = [d for d in docs if d.id == args.doc_id]
    if args.source_prefix:
        docs = [d for d in docs if d.source and d.source.startswith(args.source_prefix)]
    if args.language:
        docs = [d for d in docs if d.language == args.language]
    if args.skip_poetry:
        docs = [d for d in docs if d.genre != "poetry"]
    if args.limit:
        docs = docs[:args.limit]

    # Count work up front (translated but not yet styled). Segments whose
    # "translation" is our own untranslatable-placeholder (garbled source,
    # see ingest/garble_detect.py) get the same placeholder copied straight
    # into english_styled -- no point asking the LLM to "Victorian-ize" a
    # sentence that says the source was unreadable.
    plan = []   # (doc, [(section, [segments])])
    placeholder_pairs = []  # (segment_id, placeholder) for direct writes
    for d in docs:
        full = lib.store.get_document(d.id)
        sections = []
        for section in sorted(full.sections, key=lambda s: s.order):
            pending = [s for s in sorted(section.segments, key=lambda x: x.order)
                       if s.is_translated and not s.is_styled]
            segs = []
            for s in pending:
                if s.english_text == UNTRANSLATABLE_PLACEHOLDER:
                    placeholder_pairs.append((s.id, UNTRANSLATABLE_PLACEHOLDER, args.preset))
                else:
                    segs.append(s)
            if segs:
                sections.append((section, segs))
        if sections:
            plan.append((full, sections))
    if placeholder_pairs:
        lib.store.set_styled(placeholder_pairs)
        print(f"Copied placeholder straight through for {len(placeholder_pairs):,} "
              f"untranslatable segments (skipped the LLM).", flush=True)

    if args.shard_count > 1:
        # Greedy load balance: largest docs first, round-robin across shards, so
        # each worker gets a roughly even segment count instead of an even doc count
        # (a handful of huge scholastic works would otherwise stack on one shard).
        plan.sort(key=lambda p: sum(len(segs) for _, segs in p[1]), reverse=True)
        plan = plan[args.shard_index::args.shard_count]

    total = sum(len(segs) for _, sections in plan for _, segs in sections)
    print(f"=== Stylize ({args.preset}/{args.backend}) shard {args.shard_index}/"
          f"{args.shard_count}: {len(plan)} docs, {total:,} segments pending ===\n",
          flush=True)
    if total == 0:
        lib.close()
        return

    stylizer = lib._stylizer_for(args.backend)
    grand = 0
    t_start = time.time()
    for doc, sections in plan:
        context = {
            "source_lang": doc.language_name,
            "author": doc.author,
            "era": doc.language_stage.replace("_", " ") if doc.language_stage else None,
        }
        n_doc = sum(len(segs) for _, segs in sections)
        done_doc = 0
        t0 = time.time()
        for section, segs in sections:
            for i in range(0, len(segs), args.batch_size):
                batch = segs[i:i + args.batch_size]
                units = [StyleUnit(literal=s.english_text, source=s.latin_text,
                                   scansion=s.scansion) for s in batch]
                styled = stylize_batch_safe(stylizer, units, args.preset, context,
                                            args.batch_size)
                label = "victorian_prose" if args.backend == "t5" else args.preset
                lib.store.set_styled(
                    [(s.id, text, label) for s, text in zip(batch, styled) if text]
                )
                done_doc += len(batch)
                grand += len(batch)
                rate = grand / max(time.time() - t_start, 1e-6)
                eta = (total - grand) / rate / 3600
                print(f"  [{doc.id}] {(doc.source or '')[:22]:22} "
                      f"{done_doc:>6,}/{n_doc:<6,} | overall {grand:,}/{total:,} "
                      f"{rate:.2f} seg/s ETA {eta:.1f}h", flush=True)
        print(f"  [{doc.id}] DONE {n_doc} segs in {time.time()-t0:.0f}s", flush=True)

    print(f"\nDone. Stylized {grand:,} segments across {len(plan)} docs "
          f"in {(time.time()-t_start)/3600:.1f}h.")
    stylizer.close()
    lib.close()


if __name__ == "__main__":
    main()
