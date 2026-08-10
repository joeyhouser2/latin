"""Retroactively replace fabricated translations of garbled source text with
an honest placeholder (see ingest/garble_detect.py for why: an NMT model
doesn't fail loudly on non-Latin noise, it produces fluent fake English).

For segments already translated/styled before the garble gate existed, this
overwrites both english_text and english_styled with the placeholder. Dry-run
by default.

Usage:
    python scripts/mark_garbled_segments.py --doc-id 376
    python scripts/mark_garbled_segments.py --doc-id 376 --apply
"""
from __future__ import annotations

import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.store import Store
from ingest.garble_detect import is_garbled, garble_score, UNTRANSLATABLE_PLACEHOLDER


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-id", type=int, required=True)
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("--threshold", type=float, default=10.0)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    store = Store(args.db)
    rows = store.conn.execute(
        """SELECT s.id, s.latin_text, s.english_text, s.english_styled
           FROM segments s JOIN sections sec ON s.section_id = sec.id
           WHERE sec.doc_id = ?""",
        (args.doc_id,),
    ).fetchall()

    already = sum(1 for _, _, en, _ in rows if en == UNTRANSLATABLE_PLACEHOLDER)
    to_mark = [(sid, lat, en, sty) for sid, lat, en, sty in rows
               if en != UNTRANSLATABLE_PLACEHOLDER and is_garbled(lat, args.threshold)]

    print(f"=== Marking garbled segments for doc {args.doc_id} (threshold={args.threshold}) ===")
    print(f"Total segments: {len(rows):,}")
    print(f"Already marked (from garble gate in translate_pending.py): {already:,}")
    print(f"Newly identified as garbled, currently have a fabricated translation: {len(to_mark):,}")

    print("\n--- Sample of what will be overwritten ---")
    for sid, lat, en, sty in to_mark[:10]:
        print(f"  [{sid}] score={garble_score(lat):.1f}")
        print(f"    LAT: {lat[:120]}")
        print(f"    old ENG: {(en or '')[:120]}")

    if args.apply:
        pairs_text = [(sid, UNTRANSLATABLE_PLACEHOLDER) for sid, *_ in to_mark]
        pairs_styled = [(sid, UNTRANSLATABLE_PLACEHOLDER, "victorian_prose") for sid, *_ in to_mark]
        store.conn.executemany("UPDATE segments SET english_text = ? WHERE id = ?",
                               [(UNTRANSLATABLE_PLACEHOLDER, sid) for sid, *_ in to_mark])
        store.conn.executemany("UPDATE segments SET english_styled = ?, style_label = ? WHERE id = ?",
                               [(UNTRANSLATABLE_PLACEHOLDER, "placeholder", sid) for sid, *_ in to_mark])
        store.conn.commit()
        print(f"\nApplied: overwrote {len(to_mark):,} segments' english_text and english_styled.")
    else:
        print("\nDry run only -- no database changes made. Re-run with --apply to write them.")


if __name__ == "__main__":
    main()
