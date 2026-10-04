"""A portable ledger of what has been translated and summarized.

corpus.db and summaries.db are rebuildable local state and far too big (and
too binary to merge) for git. What *is* worth sharing between computers is the
record of work already paid for: which documents have English, how much of
each, which have summaries. That lives in two line-per-record JSON files under
``data/ledger/``, committed to git:

* ``documents.jsonl`` -- one line per document that has any work on it;
* ``summaries.jsonl`` -- the LLM summaries themselves (small, expensive).

Records are keyed by ``documents.source`` (unique per document, identical on
every machine that ingested it) -- never by ``documents.id``, which is a local
autoincrement. Lines are sorted by key, so two computers touching different
documents merge in git without conflicts; if the same line does conflict, take
either side and run ``scripts/ledger.py sync``, which merges by "most work
wins" and rewrites the files.

The ledger shares *knowledge of* the work, not the translated text: another
machine learns "doc X is fully translated on DESKTOP-A" and can skip it, but
does not receive the English. Summaries are the exception -- they travel whole.
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_DIR = REPO_ROOT / "data" / "ledger"
DOCS_FILE = "documents.jsonl"
SUMMARIES_FILE = "summaries.jsonl"

MACHINE = socket.gethostname()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -- reading the local databases ----------------------------------------------

def _ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def scan_local(corpus_path: Path, summaries_path: Optional[Path] = None
               ) -> Dict[str, dict]:
    """Every document's current state, keyed by source. Read-only."""
    corpus = _ro(corpus_path)
    try:
        rows = corpus.execute(
            """SELECT d.id, d.source, d.title, d.author, d.language,
                      d.translation_status,
                      COUNT(seg.id) AS seg_total,
                      COALESCE(SUM(seg.english_text IS NOT NULL AND seg.english_text <> ''), 0)
                          AS seg_translated,
                      COALESCE(SUM(seg.english_styled IS NOT NULL), 0) AS seg_styled
               FROM documents d
               LEFT JOIN sections sec ON sec.doc_id = d.id
               LEFT JOIN segments seg ON seg.section_id = sec.id
               WHERE d.source IS NOT NULL
               GROUP BY d.id""").fetchall()
        summarized = {}
        if summaries_path and summaries_path.exists():
            s = _ro(summaries_path)
            try:
                summarized = {r["doc_id"]: r["translated_count"] for r in s.execute(
                    "SELECT doc_id, translated_count FROM summaries WHERE level='document'")}
            finally:
                s.close()

        out: Dict[str, dict] = {}
        for r in rows:
            touched = (r["seg_translated"] or r["seg_styled"]
                       or r["translation_status"] != "unknown"
                       or r["id"] in summarized)
            if not touched:
                continue
            rec = {
                "source": r["source"], "title": r["title"], "author": r["author"],
                "language": r["language"],
                "seg_total": r["seg_total"], "seg_translated": r["seg_translated"],
                "seg_styled": r["seg_styled"],
                "translation_status": r["translation_status"],
                "latin_hash": _latin_hash(corpus, r["id"]),
                "summarized_at_count": summarized.get(r["id"]),
                "machine": MACHINE, "updated_at": _now(),
            }
            out[r["source"]] = rec
        return out
    finally:
        corpus.close()


def _latin_hash(corpus: sqlite3.Connection, doc_id: int) -> str:
    """Fingerprint of a document's source text, in reading order. Two machines
    whose hashes differ ingested different text under the same source, so
    segment-level offsets (summaries) must not be trusted across them."""
    h = hashlib.sha1()
    for (t,) in corpus.execute(
            """SELECT seg.latin_text FROM segments seg
               JOIN sections sec ON sec.id = seg.section_id
               WHERE sec.doc_id = ? ORDER BY sec.ord, seg.ord""", (doc_id,)):
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def scan_summaries(corpus_path: Path, summaries_path: Path) -> Dict[str, List[dict]]:
    """Every local summary row, grouped by the document's source."""
    if not summaries_path.exists():
        return {}
    corpus, s = _ro(corpus_path), _ro(summaries_path)
    try:
        src = {r["id"]: r["source"] for r in corpus.execute(
            "SELECT id, source FROM documents WHERE source IS NOT NULL")}
        out: Dict[str, List[dict]] = {}
        for r in s.execute("SELECT * FROM summaries ORDER BY doc_id, level, part_index"):
            source = src.get(r["doc_id"])
            if source is None:
                continue
            emb = r["embedding"]
            out.setdefault(source, []).append({
                "source": source, "level": r["level"], "part_index": r["part_index"],
                "part_count": r["part_count"],
                "seg_offset_first": r["seg_offset_first"],
                "seg_offset_last": r["seg_offset_last"],
                "section_first": r["section_first"], "section_last": r["section_last"],
                "summary": r["summary"], "topics": json.loads(r["topics"] or "[]"),
                "model": r["model"], "source_mode": r["source_mode"],
                "translated_count": r["translated_count"],
                "embed_model": r["embed_model"],
                "embedding": base64.b64encode(emb).decode("ascii") if emb else None,
                "created_at": r["created_at"],
            })
        return out
    finally:
        corpus.close()
        s.close()


# -- the ledger files -----------------------------------------------------------

def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")) + "\n")


def load_documents(ledger_dir: Path = LEDGER_DIR) -> Dict[str, dict]:
    return {r["source"]: r for r in _read_jsonl(ledger_dir / DOCS_FILE)}


def load_summaries(ledger_dir: Path = LEDGER_DIR) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for r in _read_jsonl(ledger_dir / SUMMARIES_FILE):
        out.setdefault(r["source"], []).append(r)
    return out


def _progress(rec: dict) -> Tuple[int, int, str]:
    return (rec.get("seg_translated") or 0, rec.get("seg_styled") or 0,
            rec.get("updated_at") or "")


def merge_documents(a: Dict[str, dict], b: Dict[str, dict]) -> Dict[str, dict]:
    """Union by source; where both know a document, the one with more
    translated segments wins (then more styled, then newer). A summary count
    is carried from whichever side has one, since it is independent of who
    holds the translation."""
    out = dict(a)
    for key, rec in b.items():
        mine = out.get(key)
        if mine is None:
            out[key] = rec
            continue
        win, lose = (rec, mine) if _progress(rec) > _progress(mine) else (mine, rec)
        win = dict(win)
        if win.get("summarized_at_count") is None and lose.get("summarized_at_count") is not None:
            win["summarized_at_count"] = lose["summarized_at_count"]
        if win.get("translation_status", "unknown") == "unknown":
            win["translation_status"] = lose.get("translation_status", "unknown")
        out[key] = win
    return out


def merge_summaries(a: Dict[str, List[dict]], b: Dict[str, List[dict]]
                    ) -> Dict[str, List[dict]]:
    """Per document, keep the summary set made from more translated text
    (newest on a tie). Sets are never mixed: a document summary and its part
    summaries have to describe the same state of the translation."""
    def rank(rows: List[dict]) -> Tuple[int, str]:
        return (max(r["translated_count"] for r in rows),
                max(r.get("created_at") or "" for r in rows))
    out = dict(a)
    for key, rows in b.items():
        if key not in out or rank(rows) > rank(out[key]):
            out[key] = rows
    return out


def write_ledger(docs: Dict[str, dict], summaries: Dict[str, List[dict]],
                 ledger_dir: Path = LEDGER_DIR) -> None:
    _write_jsonl(ledger_dir / DOCS_FILE, (docs[k] for k in sorted(docs)))
    _write_jsonl(ledger_dir / SUMMARIES_FILE,
                 (r for k in sorted(summaries)
                  for r in sorted(summaries[k], key=lambda r: (r["level"], r["part_index"] or 0))))


# -- applying the ledger to a local database -------------------------------------

def import_summaries(corpus_path: Path, summaries_path: Path,
                     ledger: Dict[str, List[dict]], local_docs: Dict[str, dict],
                     ledger_docs: Dict[str, dict]) -> Tuple[int, int, int]:
    """Copy summaries this machine lacks (or has from less translated text)
    into summaries.db. Returns (documents_imported, rows_imported, skipped).

    A document whose source text differs from the one the summary was written
    from is skipped: part summaries carry segment offsets into that text."""
    from core.summaries import SummaryStore

    corpus = _ro(corpus_path)
    ids = {r["source"]: (r["id"], r["title"], r["author"])
           for r in corpus.execute("SELECT id, source, title, author FROM documents "
                                   "WHERE source IS NOT NULL")}
    corpus.close()
    store = SummaryStore(summaries_path)
    have = store.summarized_doc_ids()
    docs_n = rows_n = skipped = 0
    try:
        for source, rows in ledger.items():
            if source not in ids:
                continue
            doc_id, title, author = ids[source]
            lh = (ledger_docs.get(source) or {}).get("latin_hash")
            mine = (local_docs.get(source) or {}).get("latin_hash")
            if lh and mine and lh != mine:
                skipped += 1
                continue
            incoming = max(r["translated_count"] for r in rows)
            if have.get(doc_id) is not None and have[doc_id] >= incoming:
                continue
            store.clear_doc(doc_id)
            for r in rows:
                sid = store.add(
                    doc_id=doc_id, level=r["level"], summary=r["summary"],
                    topics=r["topics"], model=r["model"], source_mode=r["source_mode"],
                    translated_count=r["translated_count"], title=title, author=author,
                    part_index=r["part_index"], part_count=r["part_count"],
                    seg_offset_first=r["seg_offset_first"],
                    seg_offset_last=r["seg_offset_last"],
                    section_first=r["section_first"], section_last=r["section_last"])
                if r.get("embedding"):
                    store.conn.execute(
                        "UPDATE summaries SET embed_model=?, embedding=?, created_at=? WHERE id=?",
                        (r["embed_model"], base64.b64decode(r["embedding"]),
                         r["created_at"], sid))
                    store.conn.commit()
                rows_n += 1
            docs_n += 1
    finally:
        store.close()
    return docs_n, rows_n, skipped


# -- questions the ledger answers -------------------------------------------------

def classify(local: Optional[dict], shared: Optional[dict]) -> str:
    """'done' (here), 'done_elsewhere', 'partial', or 'untouched'."""
    def frac(rec):
        if not rec or not rec.get("seg_total"):
            return 0.0
        return rec["seg_translated"] / rec["seg_total"]
    here, there = frac(local), frac(shared)
    if here >= 1.0:
        return "done"
    if there >= 1.0:
        return "done_elsewhere"
    if here > 0 or there > 0:
        return "partial"
    return "untouched"


def done_elsewhere_sources(ledger_dir: Path = LEDGER_DIR) -> set:
    """Sources the shared ledger records as fully translated somewhere."""
    return {k for k, r in load_documents(ledger_dir).items()
            if r.get("seg_total") and r["seg_translated"] >= r["seg_total"]}
