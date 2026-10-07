"""Transcribe a whole IIIF manuscript and save every stage for inspection.

    python scripts/transcribe_manuscript.py ecodices:csg-0226 --out data/manuscripts/csg-0226

Writes: raw HTR per page (htr_raw.json), expanded text (expanded.txt), the
sectioned text the library would ingest (sections.json) and meta.json. Does not
touch corpus.db -- ingest afterwards with scripts/ingest.py if the text is good.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from ingest.registry import get_connector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("identifier", help="iiif identifier, e.g. ecodices:csg-0226")
    ap.add_argument("--out", required=True)
    ap.add_argument("--options", default="mode=htr", help="#options for the connector")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ident = args.identifier + ("#" + args.options if args.options else "")
    meta, parts = get_connector("iiif").fetch(ident)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "sections.json").write_text(json.dumps(parts, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "expanded.txt").write_text("\n\n".join(t for _, t in parts), encoding="utf-8")
    words = sum(len(t.split()) for _, t in parts)
    print(f"{meta.get('title')}: {len(parts)} sections, {words} words -> {out}")


if __name__ == "__main__":
    main()
