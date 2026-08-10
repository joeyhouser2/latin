"""Enrich library documents' translation_status by checking, in priority
order, whether an English translation is already known to exist:

  1. Repertorium Fontium (ingest.repertorium)     - medieval Latin narrative
     sources, c. 750-1500. Most precise for that slice; Latin only.
  2. Series catalogs (ingest.series_catalogs)      - Loeb (paywalled), DOML
     (paywalled), ANF/NPNF (free). Covers classical/patristic/Greek.
  3. Wikidata (ingest.wikidata_translation)        - broadest coverage, any
     era/language, but sparsest per-item data. Tried last as a fallback.

Each source is tried in order per document; the first one to return a
confident (non-"unknown") result wins, so a source's specific scope doesn't
need to be re-checked here. All three use the same generic-title-guarded
fuzzy matcher (ingest.fuzzy_match): a bare title like "carmina" only counts
as a match if the author actually corroborates it, or the result stays
honestly "unknown" rather than guessing.

Statuses written:
  translated            - a free English translation is known to exist
  translated_paywalled  - an English translation exists but only in a
                           commercial edition (Loeb/DOML) -- our own MT
                           rendering still adds real value here
  untranslated           - high-confidence: none exists
  unknown                 - no source had a confident signal; left unchanged

Run --dry-run first to review proposed changes.

Usage:
    python scripts/enrich_translation_status.py --dry-run
    python scripts/enrich_translation_status.py                 # apply
    python scripts/enrich_translation_status.py --language la --only-unknown
    python scripts/enrich_translation_status.py --sources series,wikidata
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Latin/Greek titles are Unicode; the default Windows console is cp1252 and
# would raise UnicodeEncodeError on print.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from core.store import Store
from ingest.repertorium import RepertoriumLookup
from ingest.series_catalogs import SeriesCatalogLookup
from ingest.wikidata_translation import WikidataLookup

_SOURCE_BUILDERS = {
    "repertorium": RepertoriumLookup,
    "series": SeriesCatalogLookup,
    "wikidata": WikidataLookup,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write")
    ap.add_argument("--language", default="all",
                    help="restrict to a language ('la', 'grc', or 'all')")
    ap.add_argument("--source-prefix", default="",
                    help="only docs whose source starts with this (e.g. 'CroALa (')")
    ap.add_argument("--only-unknown", action="store_true",
                    help="skip docs whose translation_status is already set")
    ap.add_argument("--sources", default="repertorium,series,wikidata",
                    help="comma-separated source order to try "
                         "(repertorium,series,wikidata)")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="seconds between lookups (be kind to the servers)")
    args = ap.parse_args()

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [n for n in names if n not in _SOURCE_BUILDERS]
    if unknown:
        ap.error(f"unknown source(s) {unknown}; choose from {list(_SOURCE_BUILDERS)}")
    sources = [(n, _SOURCE_BUILDERS[n]()) for n in names]

    store = Store(args.db)
    docs = store.list_documents()
    if args.language != "all":
        docs = [d for d in docs if d.language == args.language]
    if args.source_prefix:
        docs = [d for d in docs if d.source and d.source.startswith(args.source_prefix)]
    if args.only_unknown:
        docs = [d for d in docs if d.translation_status == "unknown"]

    print(f"=== Enriching {len(docs)} docs via {'/'.join(names)} "
          f"({'dry-run' if args.dry_run else 'apply'}) ===\n")
    changed = matched = 0
    for d in docs:
        if not d.title:
            continue
        hit = None
        hit_source = None
        for src_name, lk in sources:
            res = lk.lookup(d.title, author=d.author)
            time.sleep(args.sleep)
            if res.status != "unknown":
                hit, hit_source = res, src_name
                break
        if hit is None:
            print(f"  --   [{d.id}] {d.title[:46]:46} no match")
            continue

        matched += 1
        flag = "" if hit.status == d.translation_status else "  <= CHANGE"
        ident = getattr(hit, "werk_id", None) or getattr(hit, "qid", None)
        mt = getattr(hit, "matched_title", None)
        detail = f"{ident}:{mt}" if ident and mt else (mt or ident or "?")
        print(f"  {hit.status:22} [{d.id}] {d.title[:40]:40} "
              f"via={hit_source:11} match={detail}{flag}")
        if hit.status != d.translation_status:
            changed += 1
            if not args.dry_run:
                store.set_translation_status(d.id, hit.status)

    verb = "would change" if args.dry_run else "changed"
    print(f"\n{matched} matched across sources; {verb} {changed} document(s).")
    store.close()


if __name__ == "__main__":
    main()
