"""Audit OCR-derived documents with the known-word rate, to find ones worth re-doing.

    python -m ingest.ocr_audit                 # writes data/cache/ocr_audit.json
    python -m ingest.ocr_audit --below 0.6     # just print the weak ones

For each Latin document whose source mentions OCR, sample up to ``--sample``
segments spread across the text and score them with ``abbrev.vocab_hit_rate``
(share of 3+ letter words found in the corpus vocabulary; ~0.8 readable Latin,
~0.3 noise). Reads corpus.db read-only -- safe while the app is running.

Caveat: the vocabulary is built from the corpus itself, so a large junk document
partly vouches for its own mistakes; scores are an upper bound, and a low score
is reliable where a high one is only suggestive.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import abbrev

_REPO = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("LATIN_AUDIT_CACHE") or _REPO / "data" / "cache" / "ocr_audit.json")
_OCR_TAG = re.compile(r"\[OCR: ([^\]]+)\]")
WEAK = 0.60


def engine_of(source: Optional[str]) -> Optional[str]:
    """How a document's text was made from page images, or None if it was not.

    New scan connectors stamp ``[OCR: engine]`` into ``source``. Older documents
    carry no stamp: Internet Archive items are OCR dumps ("archive.org"), and a few
    others say OCR in words.
    """
    if not source:
        return None
    m = _OCR_TAG.search(source)
    if m:
        return m.group(1)
    if source.startswith("Internet Archive"):
        return "archive.org"
    return "ocr" if "OCR" in source else None


_engine = engine_of


def audit(db_path: Optional[str] = None, sample: int = 200, include_untagged: bool = False,
          min_segments: int = 150, log=lambda m: None) -> List[Dict[str, Any]]:
    """Score OCR-tagged documents; with ``include_untagged`` also every long Latin document
    (old scans were ingested without a tag -- e.g. doc 376, a local text file of an OCR)."""
    path = Path(db_path or _REPO / "data" / "corpus.db")
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    vocab = abbrev.build_vocab()
    n_seg = {r[0]: r[1] for r in conn.execute(
        "SELECT s.doc_id, COUNT(*) FROM sections s JOIN segments g ON g.section_id = s.id GROUP BY s.doc_id")}
    docs = [r for r in conn.execute(
        "SELECT id, title, author, language, source FROM documents WHERE language = 'la'")
        if _engine(r["source"]) or (include_untagged and n_seg.get(r["id"], 0) >= min_segments)]
    log(f"{len(docs)} Latin documents to score")
    out = []
    for r in docs:
        ids = [x[0] for x in conn.execute(
            """SELECT seg.id FROM segments seg JOIN sections s ON s.id = seg.section_id
               WHERE s.doc_id = ? ORDER BY s.ord, seg.ord""", (r["id"],))]
        if not ids:
            continue
        step = max(1, len(ids) // sample)
        pick = ids[::step][:sample]
        text = "\n".join(x[0] for x in conn.execute(
            f"SELECT latin_text FROM segments WHERE id IN ({','.join('?' * len(pick))})", pick))
        out.append({"id": r["id"], "title": r["title"], "author": r["author"],
                    "engine": _engine(r["source"]) or "untagged", "segments": len(ids),
                    "sampled": len(pick), "rate": round(abbrev.vocab_hit_rate(text, vocab), 3)})
    conn.close()
    out.sort(key=lambda d: d["rate"])
    return out


def save(results: List[Dict[str, Any]]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"at": time.time(), "weak_below": WEAK, "documents": results},
                                ensure_ascii=False, indent=1), encoding="utf-8")


def load() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--below", type=float, default=None, help="print only documents under this rate")
    ap.add_argument("--db", default=None)
    ap.add_argument("--all", action="store_true",
                    help="also score long Latin documents that carry no OCR tag")
    args = ap.parse_args()
    res = audit(args.db, args.sample, include_untagged=args.all, log=lambda m: print(m, file=sys.stderr))
    save(res)
    limit = args.below if args.below is not None else WEAK
    weak = [d for d in res if d["rate"] < limit]
    print(f"{len(res)} documents scored; {len(weak)} under {limit:.0%} known words")
    for d in weak[:60]:
        print(f"  {d['rate']:.0%}  #{d['id']:<6} {d['engine']:<10} {d['segments']:>6} seg  {d['title'][:60]}")


if __name__ == "__main__":
    main()
