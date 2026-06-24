"""Seed the library with one real medieval text for the Reader demo.

Ingests Einhard's *Vita Karoli Magni* (9th c.) from The Latin Library, then
translates the first N segments with NLLB so the Reader has aligned content to
show. The rest can be translated from the UI ("Translate untranslated" button).

Run:  python scripts/seed_demo.py [--max-translate N] [--url URL]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import Library
from ingest.latin_library import LatinLibraryConnector

DEFAULT_URL = "https://www.thelatinlibrary.com/ein.html"

# Offline fallback: the genuine opening of the Vita Karoli, in case fetch fails.
FALLBACK_META = {
    "title": "Vita Karoli Magni (excerpt)",
    "author": "Einhardus",
    "source": "fallback sample",
    "language_stage": "medieval",
}
FALLBACK_TEXT = (
    "Vitam et conversationem et ex parte non modica res gestas domini et "
    "nutritoris mei Karoli, excellentissimi et merito famosissimi regis, postquam "
    "scribere animus tulit, quanta potui brevitate complexus sum. Opusculum, ut "
    "nec prolixitate sua legentibus fastidium pareret, in ipsa fronte temperavi. "
    "Gens Francorum sub regibus Meroingis degener esse coeperat, et a regni "
    "potestate ad ministerii curam translata est."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--max-translate", type=int, default=30,
                    help="Translate at most this many segments now (0 = none).")
    args = ap.parse_args()

    lib = Library()

    # Skip if Einhard is already loaded.
    for d in lib.list_documents():
        if d.author == "Einhardus":
            print(f"Already seeded: {d.title} (doc {d.id}). Nothing to do.")
            lib.close()
            return

    meta = {
        "author": "Einhardus",
        "century": 9,
        "genre": "biography",
        "language_stage": "medieval",
        "has_existing_translation": True,  # famous; flagged honestly
    }
    try:
        print(f"Fetching {args.url} ...")
        raw_meta, parts = LatinLibraryConnector().fetch(args.url, **meta)
        chars = sum(len(t) for _, t in parts)
        print(f"Fetched {chars} chars across {len(parts)} section(s).")
    except Exception as exc:
        print(f"Fetch failed ({exc}); using offline fallback sample.")
        raw_meta = {**FALLBACK_META, **meta, "has_existing_translation": False}
        parts = [("Text", FALLBACK_TEXT)]

    doc = LatinLibraryConnector.build_document(raw_meta, parts)
    lib.ingest(doc)
    segs = list(doc.iter_segments())
    print(f"Ingested doc {doc.id}: '{doc.title}' with {len(segs)} segments.")

    n = args.max_translate
    if n:
        to_do = [s for s in segs if not s.is_translated][:n]
        if to_do:
            print(f"Translating first {len(to_do)} segments with NLLB "
                  "(slow on CPU; the rest can be done from the UI)...")
            translator = lib._ensure_translator()
            english = translator.translate_batch([s.latin_text for s in to_do])
            lib.store.set_translations([(s.id, e) for s, e in zip(to_do, english)])
            print(f"Translated {len(to_do)} segments.")

    lib.close()
    print("Done. Launch the reader with:  python app.py")


if __name__ == "__main__":
    main()
