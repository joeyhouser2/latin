"""Enrich library documents' translation_status by looking each work up in the
Geschichtsquellen/Repertorium Fontium (see ingest.repertorium).

The Repertorium records published translations per language, so it turns our
genre-proxy guess into an English-specific ground-truth signal:
  - English translation listed         -> translated
  - work found, translations but no Eng -> untranslated (into English)
  - work found, no translations at all  -> untranslated
  - no confident title match            -> left unchanged (honest "unknown")

SCOPE: the Repertorium covers Latin narrative sources of the medieval German
Reich (c. 750-1500), so this helps the MGH-style Latin material; Greek and
non-narrative genres (charters, exegesis) usually return no match and are left
as-is. Run --dry-run first to review proposed changes.

Usage:
    python scripts/enrich_translation_status.py --dry-run
    python scripts/enrich_translation_status.py                 # apply
    python scripts/enrich_translation_status.py --language la --only-unknown
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.store import Store
from ingest.repertorium import RepertoriumLookup


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write")
    ap.add_argument("--language", default="la",
                    help="restrict to a language ('la', 'grc', or 'all'); "
                         "default 'la' since the Repertorium is Latin-only")
    ap.add_argument("--only-unknown", action="store_true",
                    help="skip docs whose translation_status is already set")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="seconds between lookups (be kind to the server)")
    args = ap.parse_args()

    store = Store(args.db)
    rep = RepertoriumLookup()
    docs = store.list_documents()
    if args.language != "all":
        docs = [d for d in docs if d.language == args.language]
    if args.only_unknown:
        docs = [d for d in docs if d.translation_status == "unknown"]

    print(f"=== Enriching {len(docs)} docs via Repertorium "
          f"({'dry-run' if args.dry_run else 'apply'}) ===\n")
    changed = matched = 0
    for d in docs:
        if not d.title:
            continue
        res = rep.lookup(d.title, author=d.author)
        if res.status == "unknown":
            print(f"  --   [{d.id}] {d.title[:46]:46} no match")
        else:
            matched += 1
            flag = "" if res.status == d.translation_status else "  <= CHANGE"
            print(f"  {res.status:12} [{d.id}] {d.title[:46]:46} "
                  f"werk={res.werk_id} eng={res.has_english}{flag}")
            if res.status != d.translation_status:
                changed += 1
                if not args.dry_run:
                    store.set_translation_status(d.id, res.status)
        time.sleep(args.sleep)

    verb = "would change" if args.dry_run else "changed"
    print(f"\n{matched} matched in Repertorium; {verb} {changed} document(s).")
    store.close()


if __name__ == "__main__":
    main()
