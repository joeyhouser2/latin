"""Check that the FAISS index and the store agree. Read-only: writes nothing.

Reports:
  gaps    - segments that pass core.normalize.is_worth_indexing but have no vector
            (e.g. a document whose embedding failed after the store committed it)
  orphans - index ids with no matching segment (e.g. a deleted document)

Exits 1 if either count is non-zero, so it can gate a script or job.

Run:  python scripts/index_coverage.py [--db data/corpus.db] [--index data/index.faiss]
Fix gaps/orphans with scripts/reindex.py (full rebuild).
"""
import argparse
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np

from core.normalize import is_worth_indexing, _letters


def index_ids(path: str) -> np.ndarray:
    """Segment ids stored in an IndexIDMap (read fully: ~1.3 GB, a few seconds)."""
    # Keep the read_index result alive: it owns the C++ object, and a bare
    # downcast_index(read_index(...)) frees it, leaving id_map empty.
    owner = faiss.read_index(path)
    index = faiss.downcast_index(owner)
    if not hasattr(index, "id_map"):
        raise SystemExit(f"{path}: not an IndexIDMap ({type(index).__name__})")
    ids = faiss.vector_to_array(index.id_map).astype("int64")
    if len(ids) != index.ntotal:
        raise SystemExit(f"{path}: id_map has {len(ids)} ids but ntotal is {index.ntotal}")
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=os.environ.get("LATIN_CORPUS_DB") or "data/corpus.db")
    ap.add_argument("--index", default=os.environ.get("LATIN_INDEX") or "data/index.faiss")
    ap.add_argument("--show", type=int, default=20, help="documents/ids to list (default 20)")
    args = ap.parse_args(argv)

    for p in (args.db, args.index):
        if not Path(p).exists():
            raise SystemExit(f"missing: {p}")

    ids = index_ids(args.index)
    in_index = set(ids.tolist())
    dupes = len(ids) - len(in_index)

    conn = sqlite3.connect(Path(args.db).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    rows = conn.execute(
        "SELECT seg.id, sec.doc_id, COALESCE(seg.embed_text, seg.latin_text) "
        "FROM segments seg JOIN sections sec ON sec.id = seg.section_id"
    )
    n_seg = n_worth = 0
    all_seg_ids = set()
    gaps_by_doc = Counter()
    short_indexed = 0
    doc_has_vector, doc_has_letters = set(), set()
    for seg_id, doc_id, text in rows:
        n_seg += 1
        all_seg_ids.add(seg_id)
        if seg_id in in_index:
            doc_has_vector.add(doc_id)
        if _letters(text):
            doc_has_letters.add(doc_id)
        if is_worth_indexing(text):
            n_worth += 1
            if seg_id not in in_index:
                gaps_by_doc[doc_id] += 1
        elif seg_id in in_index:
            short_indexed += 1
    orphans = sorted(in_index - all_seg_ids)
    # Ingest/reindex give a short-only document one fallback vector.
    unfindable = sorted(doc_has_letters - doc_has_vector)
    gaps = sum(gaps_by_doc.values())

    print(f"segments: {n_seg:,}  worth indexing: {n_worth:,}  "
          f"skipped as short: {n_seg - n_worth:,}")
    print(f"index vectors: {len(ids):,}" + (f"  (duplicate ids: {dupes:,})" if dupes else ""))
    print(f"gaps (worth indexing, no vector): {gaps:,} in {len(gaps_by_doc):,} documents")
    if gaps_by_doc:
        titles = dict(conn.execute("SELECT id, title FROM documents"))
        for doc_id, n in gaps_by_doc.most_common(args.show):
            print(f"  doc {doc_id}: {n:,}  {titles.get(doc_id, '?')[:70]}")
    print(f"documents with text but no vector: {len(unfindable):,}")
    if unfindable:
        titles = dict(conn.execute("SELECT id, title FROM documents"))
        for doc_id in unfindable[:args.show]:
            print(f"  doc {doc_id}: {titles.get(doc_id, '?')[:70]}")
    print(f"orphans (index id, no segment): {len(orphans):,}")
    if orphans:
        print("  " + " ".join(map(str, orphans[:args.show])))
    if short_indexed:
        print(f"info: {short_indexed:,} short segments have a vector (per-document fallbacks, or "
              f"indexed before the filter); harmless, not counted as errors")
    conn.close()
    return 1 if (gaps or orphans or unfindable) else 0


if __name__ == "__main__":
    sys.exit(main())
