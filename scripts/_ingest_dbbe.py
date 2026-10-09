"""Bulk ingest of DBBE (Database of Byzantine Book Epigrams) occurrences.

Resumable: skips occurrence ids already present in a `documents.source` (so an
interrupted run can just be re-launched). Rate-limited (small delay per
fetch) to be polite to DBBE's server for a personal-research bulk pull.
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from pipeline import Library
from ingest.registry import get_connector
from ingest.base import Connector

SLEEP = 0.25

lib = Library()
already = set()
for row in lib.store.conn.execute(
    "SELECT source FROM documents WHERE source LIKE 'DBBE (%'"
):
    m = re.match(r"DBBE \((\d+),", row[0])
    if m:
        already.add(m.group(1))
print(f"{len(already)} DBBE occurrences already ingested; skipping those.", flush=True)

connector = get_connector("dbbe")
print("Discovering all occurrence ids (date-partitioned, past the ~10K ceiling)...",
      flush=True)
ids = connector.discover_all(year_from=1, year_to=1600, ceiling=9000)
todo = [i for i in ids if i not in already]
print(f"{len(ids)} total ids, {len(todo)} to fetch.", flush=True)

ok = skipped = 0
t0 = time.time()
try:
    for n, occ_id in enumerate(todo, 1):
        try:
            meta, parts = connector.fetch(occ_id)
            if not parts[0][1].strip():
                skipped += 1
                continue
            doc = Connector.build_document(meta, parts, use_cltk=False)
            lib.ingest(doc)
            ok += 1
        except Exception as exc:
            skipped += 1
            print(f"  ! {occ_id} failed: {exc}", flush=True)
        if n % 50 == 0:
            rate = n / max(time.time() - t0, 1e-6)
            eta = (len(todo) - n) / max(rate, 1e-6) / 3600
            print(f"[{n}/{len(todo)}] ok={ok} skipped={skipped} "
                  f"{rate:.2f}/s ETA {eta:.1f}h", flush=True)
        time.sleep(SLEEP)
finally:
    lib.close()

print(f"Done. {ok} ingested, {skipped} skipped/empty, "
      f"{(time.time()-t0)/3600:.1f}h.", flush=True)
