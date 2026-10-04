"""Stylize an explicit list of segment IDs -- unlike stylize_library.py's
--doc-id (which picks up *every* pending segment in that document, including
any large pre-existing backlog unrelated to whatever you're actually trying
to fix), this touches only the IDs you give it.

Usage:
    python scripts/stylize_segment_ids.py --ids-file data/clean/some_ids.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import Library
from core.stylizer import StyleUnit


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids-file", required=True, help="JSON list of segment ids")
    ap.add_argument("--preset", default="victorian_prose")
    ap.add_argument("--backend", default="llm", choices=["llm", "t5"])
    ap.add_argument("--batch-size", type=int, default=20)
    args = ap.parse_args()

    with open(args.ids_file, encoding="utf-8") as f:
        ids = json.load(f)

    lib = Library()
    conn = lib.store.conn
    c = conn.cursor()
    placeholders = ",".join("?" * len(ids))
    c.execute(f"""SELECT s.id, sec.doc_id, s.latin_text, s.english_text, s.scansion
                  FROM segments s JOIN sections sec ON s.section_id = sec.id
                  WHERE s.id IN ({placeholders}) AND s.english_text IS NOT NULL""", ids)
    rows = c.fetchall()
    print(f"=== Stylizing {len(rows)}/{len(ids)} requested segments (rest missing/untranslated) ===")

    # group by doc for correct author/era context
    by_doc: dict = {}
    for sid, doc_id, lat, eng, scan in rows:
        by_doc.setdefault(doc_id, []).append((sid, lat, eng, scan))

    stylizer = lib._stylizer_for(args.backend)
    doc_cache = {}
    total = 0
    for doc_id, segs in by_doc.items():
        if doc_id not in doc_cache:
            doc_cache[doc_id] = lib.store.get_document(doc_id)
        doc = doc_cache[doc_id]
        context = {
            "source_lang": doc.language_name,
            "author": doc.author,
            "era": doc.language_stage.replace("_", " ") if doc.language_stage else None,
        }
        for i in range(0, len(segs), args.batch_size):
            batch = segs[i:i + args.batch_size]
            units = [StyleUnit(literal=eng, source=lat, scansion=scan) for _, lat, eng, scan in batch]
            label = "victorian_prose" if args.backend == "t5" else args.preset
            styled = stylizer.stylize_units(units, preset=args.preset, context=context)
            lib.store.set_styled([(sid, text, label) for (sid, *_), text in zip(batch, styled) if text])
            total += len(batch)
            print(f"  [{doc_id}] {total}/{len(rows)} done", flush=True)

    print(f"Done. Styled {total} segments across {len(by_doc)} docs.")
    lib.close()


if __name__ == "__main__":
    main()
