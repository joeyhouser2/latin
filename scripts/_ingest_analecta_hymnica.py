"""One-off bulk ingest of the deduped Analecta Hymnica Medii Aevi volume list
(data/clean/analecta_hymnica_final_ids.json) via the archiveorg connector."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from pipeline import Library
from ingest.registry import get_connector
from ingest.base import Connector

with open("data/clean/analecta_hymnica_final_ids.json", encoding="utf-8") as f:
    ids = json.load(f)

connector = get_connector("archiveorg")
lib = Library()
meta_overrides = {
    "author": "Guido Maria Dreves, Clemens Blume, Henry Marriott Bannister (eds.)",
    "language": "la",
    "language_stage": "medieval",
    "genre": "poetry",
    "translation_status": "untranslated",
}

try:
    for i, ident in enumerate(ids, 1):
        try:
            raw_meta, parts = connector.fetch(ident, **meta_overrides)
            doc = Connector.build_document(raw_meta, parts, use_cltk=False)
            lib.ingest(doc)
            n_seg = len(list(doc.iter_segments()))
            print(f"[{i}/{len(ids)}] + [{doc.id}] {doc.title}  ({n_seg:,} segments)",
                  flush=True)
        except Exception as exc:
            print(f"[{i}/{len(ids)}] ! skipped {ident} ({exc})", flush=True)
finally:
    lib.close()
print("Done.")
