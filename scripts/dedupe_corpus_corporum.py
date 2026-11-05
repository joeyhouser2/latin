"""Remove the repeated text that old Corpus Corporum / ALIM ingests stored.

The old connector emitted the itertext() of every nested <div>: first the whole
work, then each section again at every nesting level, then the site's paratext.
So a document's first segments are the complete work, and everything after them
is repetition. This keeps that first pass, with its translations, and deletes
the rest.

The cut is never guessed. Each work is re-fetched, the old connector's first
chunk is rebuilt and segmented, and a document is changed only if its stored
segments begin with exactly those sentences. Anything else is reported and skipped.
One exception: if the first pass ended without punctuation, its last sentence was
stored joined to the start of the repeat. That segment is trimmed and its English
cleared for re-translation, or dropped if only punctuation is left.

    python scripts/dedupe_corpus_corporum.py            # dry run: writes nothing
    python scripts/dedupe_corpus_corporum.py --apply    # stop all corpus jobs first

--apply backs up corpus.db and summaries.db with SQLite's backup API, then updates
corpus.db, summaries.db (part summaries of removed text go; kept rows are re-counted,
so nothing looks stale), the FAISS index and the ledger records of changed documents.
Run it on each machine that ingested these documents.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8")

from core import ledger as L
from core.segmenter import segment_text
from core.summaries import SummaryStore
from core.normalize import embedding_text_for
from ingest.corpus_corporum import CorpusCorporumConnector, MIN_BODY_WORDS, _local


@dataclass
class Plan:
    doc_id: int
    source: str
    title: str
    keep: int
    total: int
    remove_ids: List[int]
    translated_kept: int
    trim_id: Optional[int] = None    # boundary segment to shorten
    trim_text: Optional[str] = None


def legacy_first_pass(root) -> List[str]:
    """The sentences the old connector produced from its first <div>: the whole work.

    Where that <div> is a stub (apparatus-only records), the text that followed it
    is the first <div> holding anything, so it is included."""
    chunks: List[str] = []
    for el in root.iter():
        if _local(el.tag) == "div":
            text = re.sub(r"\s+", " ", " ".join(el.itertext())).strip()
            if text:
                chunks.append(text)
                if len(" ".join(chunks).split()) >= MIN_BODY_WORDS:
                    break
    return segment_text("\n".join(chunks), lang="la") if chunks else []


def is_joined_boundary(stored: str, expected: str) -> bool:
    """True if the stored segment is the expected sentence plus text run on after it."""
    return stored.startswith(expected) and len(stored) > len(expected)


def plan_document(conn: sqlite3.Connection, connector, doc_id: int, source: str,
                  title: str) -> Optional[Plan]:
    rows = conn.execute(
        """SELECT s.id, s.latin_text, s.english_text FROM segments s
           JOIN sections x ON x.id = s.section_id
           WHERE x.doc_id = ? ORDER BY x.ord, s.ord""", (doc_id,)).fetchall()
    match = re.search(r"\((\d+)\)$", source)
    if not match:
        print(f"  SKIP [{doc_id}] {source}: no text id in source")
        return None
    expected = legacy_first_pass(connector._get_xml(
        "display_text.php", {"ajax": "true", "path": match.group(1)}))
    keep = len(expected)
    stored = [r[1] for r in rows[:keep]]
    trim_id, trim_text = None, None
    if keep and len(stored) == keep and stored[:-1] == expected[:-1] and stored[-1] != expected[-1] \
            and is_joined_boundary(stored[-1], expected[-1]):
        if any(ch.isalpha() for ch in expected[-1]):
            trim_id, trim_text = rows[keep - 1][0], expected[-1]
        else:
            keep -= 1    # a stray closing quote: nothing worth keeping
    elif keep == 0 or stored != expected:
        where = next((i for i, (a, b) in enumerate(zip(stored, expected)) if a != b),
                     min(len(stored), keep))
        print(f"  SKIP [{doc_id}] {source}: stored text differs from the source's first "
              f"pass at segment {where} (stored {len(rows)}, first pass {keep})")
        return None
    if len(rows) == keep:
        print(f"  ok   [{doc_id}] {source}: already clean ({keep} segments)")
        return None
    translated = sum(1 for r in rows[:keep] if r[2] and r[0] != trim_id)
    return Plan(doc_id, source, title, keep, len(rows), [r[0] for r in rows[keep:]],
                translated, trim_id, trim_text)


def backup(path: Path, stamp: str) -> Path:
    dest = path.with_name(f"{path.name}.bak-dedupe-{stamp}")
    src, dst = sqlite3.connect(path), sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    return dest


def update_summaries(store: SummaryStore, plan: Plan) -> int:
    """Drop parts wholly in the removed text; make the rest describe the kept text.
    Keeps the existing summaries valid, so nothing has to be re-summarized."""
    removed = store._delete_where("doc_id=? AND level='part' AND seg_offset_first>=?",
                                  (plan.doc_id, plan.keep))
    parts = store.conn.execute(
        "SELECT COUNT(*) FROM summaries WHERE doc_id=? AND level='part'",
        (plan.doc_id,)).fetchone()[0]
    store.conn.execute(
        """UPDATE summaries SET translated_count=?,
               seg_offset_last=MIN(seg_offset_last, ?),
               part_count=CASE WHEN part_count IS NULL THEN NULL ELSE ? END
           WHERE doc_id=?""",
        (plan.translated_kept, plan.keep - 1, parts, plan.doc_id))
    store.conn.commit()
    return removed


def remove_from_index(path: Path, ids: List[int]) -> int:
    import faiss
    import numpy as np
    index = faiss.read_index(str(path))
    removed = index.remove_ids(np.asarray(ids, dtype="int64"))
    tmp = path.with_name(path.name + ".tmp")
    faiss.write_index(index, str(tmp))
    tmp.replace(path)
    print(f"Index: removed {removed:,} vectors, {index.ntotal:,} remain")
    return removed


def update_ledger(corpus: Path, summaries: Path, ledger: Path, plans: List[Plan]) -> None:
    """Overwrite these records directly: the ledger's merge keeps the larger count,
    so an ordinary sync would keep the inflated numbers."""
    docs = L.load_documents(ledger)
    local = L.scan_local(corpus, summaries)
    for plan in plans:
        if plan.source in docs and plan.source in local:
            docs[plan.source] = local[plan.source]
    L.write_ledger(docs, L.load_summaries(ledger), ledger)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    ap.add_argument("--corpus", default=str(REPO / "data" / "corpus.db"))
    ap.add_argument("--summaries", default=str(REPO / "data" / "summaries.db"))
    ap.add_argument("--index", default=str(REPO / "data" / "index.faiss"))
    ap.add_argument("--ledger", default=str(L.LEDGER_DIR))
    args = ap.parse_args()
    corpus, summaries, index, ledger = map(
        Path, (args.corpus, args.summaries, args.index, args.ledger))

    ro = sqlite3.connect(f"file:{corpus}?mode=ro", uri=True, timeout=30)
    docs = ro.execute(
        """SELECT id, source, title FROM documents
           WHERE source LIKE 'Corpus Corporum (%' OR source LIKE 'ALIM (%' ORDER BY id"""
    ).fetchall()
    connector = CorpusCorporumConnector()
    plans: List[Plan] = []
    errors = 0
    for doc_id, source, title in docs:
        try:
            plan = plan_document(ro, connector, doc_id, source, title)
        except Exception as exc:
            print(f"  SKIP [{doc_id}] {source}: fetch failed ({exc})")
            errors += 1
            continue
        if plan:
            plans.append(plan)
            print(f"  cut  [{doc_id}] {source}: keep {plan.keep:,} of {plan.total:,} "
                  f"({plan.translated_kept:,} translated)"
                  f"{' +1 to re-translate' if plan.trim_id else ''} -- {title[:50]}")
        time.sleep(0.3)
    ro.close()

    remove = [i for p in plans for i in p.remove_ids]
    print(f"\n{len(docs)} documents checked; {len(plans)} to clean; "
          f"{len(remove):,} segments to remove; {errors} fetch errors")
    if not args.apply:
        print("Dry run: nothing written. Use --apply to make these changes.")
        return 0
    if not plans:
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print("Backups:", backup(corpus, stamp), backup(summaries, stamp))

    conn = sqlite3.connect(corpus, timeout=60)
    conn.execute("PRAGMA foreign_keys = ON")
    with conn:
        conn.executemany("DELETE FROM segments WHERE id = ?", [(i,) for i in remove])
        for p in plans:
            if p.trim_id is not None:
                conn.execute(
                    """UPDATE segments SET latin_text=?, embed_text=?, english_text=NULL,
                           english_styled=NULL, style_label=NULL WHERE id=?""",
                    (p.trim_text, embedding_text_for(p.trim_text), p.trim_id))
    check = conn.execute("PRAGMA integrity_check").fetchone()[0]
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    print("corpus.db integrity:", check)
    if check != "ok":
        print("Integrity check failed: stopping before summaries, index and ledger.")
        return 1

    store = SummaryStore(summaries)
    gone = sum(update_summaries(store, p) for p in plans)
    store.close()
    print(f"Summaries: removed {gone:,} part summaries")

    if index.exists():
        remove_from_index(index, remove)
    update_ledger(corpus, summaries, ledger, plans)
    print("Ledger updated. Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
