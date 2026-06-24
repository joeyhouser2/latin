"""Unified ingestion CLI: pull texts from any registered source into the library.

Examples:
    # List available sources
    python scripts/ingest.py list

    # One work from a source
    python scripts/ingest.py wikisource "Confessiones (ed. Migne)/1" --author Augustinus --stage late_antique
    python scripts/ingest.py tei https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data/phi0448/phi001/phi0448.phi001.perseus-lat2.xml
    python scripts/ingest.py latinlibrary https://www.thelatinlibrary.com/ein.html --author Einhardus --stage medieval

    # Bulk: discover many works then ingest them
    python scripts/ingest.py wikisource "Beda" --discover --limit 5 --stage medieval
    python scripts/ingest.py latinlibrary https://www.thelatinlibrary.com/aug.html --discover --limit 10
    python scripts/ingest.py file ./my_texts --discover           # all .txt in a dir

    # Translate the first N segments of each ingested doc (slow on CPU)
    python scripts/ingest.py wikisource "..." --translate 20
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import Library
from ingest.registry import get_connector, available_sources
from ingest.base import Connector


def _meta_from_args(args) -> dict:
    meta = {}
    if args.author:          meta["author"] = args.author
    if args.title:           meta["title"] = args.title
    if args.century is not None: meta["century"] = args.century
    if args.genre:           meta["genre"] = args.genre
    if args.stage:           meta["language_stage"] = args.stage
    if args.language:        meta["language"] = args.language
    if args.has_translation: meta["has_existing_translation"] = True
    return meta


def _ingest_one(lib: Library, connector: Connector, identifier: str,
                meta: dict, translate: int) -> None:
    raw_meta, parts = connector.fetch(identifier, **meta)
    doc = Connector.build_document(raw_meta, parts)
    lib.ingest(doc)
    n_seg = len(list(doc.iter_segments()))
    print(f"  + [{doc.id}] {doc.author or 'Anon.'} — {doc.title}  ({n_seg} segments)")

    if translate:
        segs = [s for s in doc.iter_segments() if not s.is_translated][:translate]
        if segs:
            translator = lib._ensure_translator()
            english = translator.translate_batch([s.latin_text for s in segs])
            lib.store.set_translations([(s.id, e) for s, e in zip(segs, english)])
            print(f"      translated {len(segs)} segments")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="connector name, or 'list'")
    ap.add_argument("identifier", nargs="?", help="URL / page title / path / category")
    ap.add_argument("--discover", action="store_true",
                    help="treat identifier as a category/index/dir and ingest all found")
    ap.add_argument("--limit", type=int, default=25, help="max works when discovering")
    ap.add_argument("--translate", type=int, default=0,
                    help="translate first N segments of each doc now")
    # metadata overrides
    ap.add_argument("--author")
    ap.add_argument("--title")
    ap.add_argument("--century", type=int)
    ap.add_argument("--genre")
    ap.add_argument("--stage", help="classical|late_antique|medieval|early_modern|ancient|unknown")
    ap.add_argument("--language", help="primary language: la (Latin) or grc (Greek)")
    ap.add_argument("--has-translation", action="store_true",
                    help="flag the work as already having an English translation")
    args = ap.parse_args()

    if args.source == "list":
        print("Available sources:", ", ".join(available_sources()))
        return
    if not args.identifier:
        ap.error("identifier is required (or use 'list')")

    connector = get_connector(args.source)
    meta = _meta_from_args(args)
    lib = Library()

    try:
        if args.discover:
            ids = connector.discover(args.identifier, limit=args.limit)
            print(f"Discovered {len(ids)} item(s) from {args.source}.")
            for i, ident in enumerate(ids, 1):
                print(f"[{i}/{len(ids)}] {ident}")
                try:
                    _ingest_one(lib, connector, ident, meta, args.translate)
                except Exception as exc:
                    print(f"      ! skipped ({exc})")
        else:
            _ingest_one(lib, connector, args.identifier, meta, args.translate)
    finally:
        lib.close()
    print("Done.")


if __name__ == "__main__":
    main()
