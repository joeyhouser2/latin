"""Share what's been translated/summarized between computers, via git.

The ledger (data/ledger/*.jsonl, see core/ledger.py) is the one piece of
corpus state that *is* committed. Typical use:

  on any machine, before starting work:   git pull ; python scripts/ledger.py sync
  to see what's left (and what's been
  done on another machine already):       python scripts/ledger.py status
  after a work session:                   python scripts/ledger.py sync ; git add data/ledger ; git commit

Commands:
  status   progress here vs. the shared ledger; --list N shows documents
  export   fold this machine's state into the ledger files (never loses
           another machine's progress -- most work wins per document)
  import   pull summaries the ledger has and this machine lacks into summaries.db
  sync     import, then export (what you normally want)

Read-only on corpus.db. Writes only data/ledger/ and, for import/sync,
data/summaries.db -- so it is safe beside a running translation.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from core import ledger as L

DATA = L.REPO_ROOT / "data"


def _paths(args):
    return Path(args.corpus), Path(args.summaries), Path(args.ledger)


def do_export(args) -> None:
    corpus, summ, ledger = _paths(args)
    local = L.scan_local(corpus, summ)
    docs = L.merge_documents(L.load_documents(ledger), local)
    sums = L.merge_summaries(L.load_summaries(ledger), L.scan_summaries(corpus, summ))
    L.write_ledger(docs, sums, ledger)
    print(f"Ledger: {len(docs):,} documents, "
          f"{sum(len(v) for v in sums.values()):,} summaries "
          f"({len(sums)} documents) -> {ledger}")


def do_import(args) -> None:
    corpus, summ, ledger = _paths(args)
    local = L.scan_local(corpus, summ)
    n_docs, n_rows, skipped = L.import_summaries(
        corpus, summ, L.load_summaries(ledger), local, L.load_documents(ledger))
    print(f"Imported {n_rows:,} summaries for {n_docs} documents"
          + (f"; skipped {skipped} whose source text differs on this machine" if skipped else ""))


def do_sync(args) -> None:
    do_import(args)
    do_export(args)


def do_status(args) -> None:
    corpus, summ, ledger = _paths(args)
    local = L.scan_local(corpus, summ)
    shared = L.load_documents(ledger)
    import sqlite3
    c = sqlite3.connect(f"file:{corpus}?mode=ro", uri=True)
    all_docs = c.execute("SELECT source, title, language FROM documents "
                         "WHERE source IS NOT NULL").fetchall()
    c.close()

    buckets = Counter()
    by_bucket = {}
    for source, title, lang in all_docs:
        k = L.classify(local.get(source), shared.get(source))
        buckets[k] += 1
        by_bucket.setdefault(k, []).append((source, title, lang))
    unknown_here = [s for s in shared if s not in {d[0] for d in all_docs}]

    print(f"Documents in this corpus:        {len(all_docs):>7,}")
    print(f"  fully translated here:         {buckets['done']:>7,}")
    print(f"  done on another machine only:  {buckets['done_elsewhere']:>7,}   (English not on this machine)")
    print(f"  partially translated:          {buckets['partial']:>7,}")
    print(f"  not started anywhere:          {buckets['untouched']:>7,}")
    print(f"Ledger knows {len(shared):,} documents; {len(unknown_here)} are not ingested on this machine.")
    sums = L.load_summaries(ledger)
    here = sum(1 for r in local.values() if r.get("summarized_at_count") is not None)
    print(f"Summaries: {here} documents here, {len(sums)} in ledger.")
    if args.list:
        which = {"todo": ("untouched", "partial"), "elsewhere": ("done_elsewhere",),
                 "partial": ("partial",)}[args.which]
        print(f"\n--- {args.which} ---")
        n = 0
        for b in which:
            for source, title, lang in by_bucket.get(b, []):
                print(f"  [{lang}] {source[:48]:48} {title[:60]}")
                n += 1
                if n >= args.list:
                    return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["status", "export", "import", "sync"])
    ap.add_argument("--corpus", default=str(DATA / "corpus.db"))
    ap.add_argument("--summaries", default=str(DATA / "summaries.db"))
    ap.add_argument("--ledger", default=str(L.LEDGER_DIR))
    ap.add_argument("--list", type=int, default=0, metavar="N",
                    help="status: also list up to N documents")
    ap.add_argument("--which", choices=["todo", "elsewhere", "partial"], default="todo",
                    help="status --list: which group (default todo = untouched + partial)")
    args = ap.parse_args()
    {"status": do_status, "export": do_export, "import": do_import,
     "sync": do_sync}[args.command](args)


if __name__ == "__main__":
    main()
