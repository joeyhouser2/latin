"""Pull a batch of ALIM (medieval Italian Latin) works into the library via the
ALIM connector (Corpus Corporum corpus 14009). Ingest only; translate later with
the normal pipeline. Skips Corpus Corporum's empty stub nodes and works already
present. Run: python scripts/pull_alim.py --target 30
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=30, help="how many works to ingest")
    ap.add_argument("--scan", type=int, default=0, help="ids to discover (default target*3)")
    ap.add_argument("--min-chars", type=int, default=300, help="skip stub/near-empty nodes")
    args = ap.parse_args()
    scan = args.scan or args.target * 3

    lib = Library()
    conn = get_connector("alim")
    existing = {d.source for d in lib.store.list_documents() if d.source}

    ids = conn.discover(limit=scan)
    print(f"=== ALIM pull: discovered {len(ids)} ids, target {args.target} works ===\n")
    ingested = skipped = 0
    for idno in ids:
        if f"ALIM ({idno})" in existing:
            skipped += 1
            continue
        try:
            meta, parts = conn.fetch(idno)
            nchars = sum(len(t) for _, t in parts)
            if nchars < args.min_chars:
                print(f"  stub  {idno}  ({nchars} chars) — skip")
                continue
            doc = Connector.build_document(meta, parts)
            nseg = len(list(doc.iter_segments()))
            if not nseg:
                continue
            lib.ingest(doc)
            ingested += 1
            print(f"  INGEST [{doc.id}] {idno}  {(meta.get('title') or '')[:46]:46} "
                  f"({nseg} segs, {nchars:,} chars)")
            if ingested >= args.target:
                break
        except Exception as exc:
            print(f"  FAIL  {idno}: {str(exc)[:70]}")

    print(f"\nIngested {ingested} ALIM works ({skipped} already present).")
    lib.close()


if __name__ == "__main__":
    main()
