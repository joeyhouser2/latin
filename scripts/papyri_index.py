"""Build the language index the papyri connector discovers from.

    python scripts/papyri_index.py --buckets DDbDP/69,DCLP/66    # part of it
    python scripts/papyri_index.py                               # all ~83,000 texts
    python scripts/papyri_index.py --stats                       # what is indexed

papyri.info marks a text's language only inside its EpiDoc file, so the only
way to find the Latin among ~83,000 texts is to read them. Files are mirrored
into data/raw/papyri as they are read (~530 MB for everything), and an entry
whose blob hash has not changed upstream is not read again -- so a full build
can be interrupted and resumed, and refreshing it later costs only what changed.
See ingest/papyri.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.papyri import TEXT_CORPORA, PapyriConnector   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append", choices=TEXT_CORPORA,
                    help="limit to DDbDP or DCLP (repeatable; default both)")
    ap.add_argument("--buckets", default="",
                    help="comma-separated corpus/bucket folders, e.g. DDbDP/69,DCLP/66")
    ap.add_argument("--stats", action="store_true", help="print what the index holds and exit")
    args = ap.parse_args()

    conn = PapyriConnector()
    if not args.stats:
        buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
        conn.build_index(args.corpus or TEXT_CORPORA, buckets or None,
                         progress=lambda s: print(s, flush=True))

    index = conn.load_index()
    by = Counter(((e.get("lang") or "?").split("-")[0], p.split("/")[0]) for p, e in index.items())
    readable = Counter(((e.get("lang") or "?").split("-")[0], p.split("/")[0])
                       for p, e in index.items() if e.get("letters", 0) >= conn.min_letters)
    print(f"\n{len(index):,} texts indexed. By language (readable = at least "
          f"{conn.min_letters} letters surviving):")
    for (lang, corpus), n in by.most_common(12):
        print(f"  {lang:8} {corpus:6} {n:6,}   readable {readable[(lang, corpus)]:6,}")


if __name__ == "__main__":
    main()
