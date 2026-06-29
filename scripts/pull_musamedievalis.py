"""Ingest the era-filtered Musa Medievalis batch (6th-10th c. medieval Latin
poetry) from the crawl manifest. Ingest only — these are POEMS, and stock NLLB
translates verse poorly (use the LLM verse-stylizer path later, not the MT).

Dedupes by work-stem (AUTHOR|work) so multi-section poems aren't ingested twice,
and skips works already present. Run scripts/crawl_musamedievalis.py first.

Usage:
    python scripts/pull_musamedievalis.py            # all unique works
    python scripts/pull_musamedievalis.py --target 50 --century 9
"""
from __future__ import annotations

import argparse
import os
import sys

import sentence_transformers  # noqa: F401  (import order: see harvest script)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import Library
from ingest.registry import get_connector
from ingest.base import Connector


def stem(code: str) -> str:
    return "|".join(code.split("|")[:2])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=0, help="max works (0 = all)")
    ap.add_argument("--century", type=int, default=0, help="only this century (0 = all)")
    ap.add_argument("--min-lines", type=int, default=2, help="skip tiny fragments")
    args = ap.parse_args()

    lib = Library()
    conn = get_connector("musamedievalis")
    entries = conn.catalogue()
    if args.century:
        entries = [e for e in entries if e.get("century") == args.century]

    # one entry per work-stem
    by_stem = {}
    for e in entries:
        by_stem.setdefault(stem(e["code"]), e)

    existing_stems = {stem(d.source[len("Musa Medievalis ("):-1])
                      for d in lib.store.list_documents()
                      if d.source and d.source.startswith("Musa Medievalis (")}

    print(f"=== Musa Medievalis pull: {len(by_stem)} unique works "
          f"({len(existing_stems)} already present) ===\n")
    ingested = 0
    for st, e in sorted(by_stem.items()):
        if st in existing_stems:
            continue
        try:
            meta, parts = conn.fetch(e["code"])
            doc = Connector.build_document(meta, parts)
            nseg = len(list(doc.iter_segments()))
            if nseg < args.min_lines:
                print(f"  thin  {e['code']} ({nseg} lines) — skip")
                continue
            lib.ingest(doc)
            ingested += 1
            print(f"  INGEST [{doc.id}] c{e.get('century')} {e['author'][:22]:22} "
                  f"{(meta.get('title') or '')[:30]:30} ({nseg} lines)")
            if args.target and ingested >= args.target:
                break
        except Exception as exc:
            print(f"  FAIL  {e['code']}: {str(exc)[:70]}")

    print(f"\nIngested {ingested} Musa Medievalis works.")
    lib.close()


if __name__ == "__main__":
    main()
