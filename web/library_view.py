"""Read-only views over corpus.db for the web app.

Deliberately separate from ``core.store.Store``: the browser needs paginated,
filtered *lists* with per-document progress counts, which is a different shape
from the store's "load one document with every segment" API -- calling that for
a 13k-document index page would read the whole corpus.

Nothing here writes. Writes to corpus.db happen only in job subprocesses (see
web.jobs), which keeps a single writer on the WAL database at any time.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "corpus.db"

# How long the per-document progress counts stay cached. The aggregate is a
# ~0.3s scan of the segments table; re-running it on every keystroke in the
# filter box would be the slowest thing on the page, and the numbers only move
# when a job commits a chunk.
COUNTS_TTL = 20.0


class LibraryView:
    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self._counts: Dict[int, Dict[str, int]] = {}
        self._counts_at = 0.0

    # -- documents -----------------------------------------------------------

    def counts(self, force: bool = False) -> Dict[int, Dict[str, int]]:
        """Per-document {segments, translated, styled}, cached for COUNTS_TTL."""
        with self._lock:
            if not force and self._counts and time.time() - self._counts_at < COUNTS_TTL:
                return self._counts
            rows = self.conn.execute(
                """SELECT s.doc_id AS doc_id, COUNT(*) AS n,
                          SUM(CASE WHEN seg.english_text IS NOT NULL
                                    AND seg.english_text <> '' THEN 1 ELSE 0 END) AS n_tr,
                          SUM(CASE WHEN seg.english_styled IS NOT NULL
                                    AND seg.english_styled <> '' THEN 1 ELSE 0 END) AS n_st
                   FROM sections s JOIN segments seg ON seg.section_id = s.id
                   GROUP BY s.doc_id"""
            ).fetchall()
            self._counts = {r["doc_id"]: {"segments": r["n"], "translated": r["n_tr"] or 0,
                                          "styled": r["n_st"] or 0} for r in rows}
            self._counts_at = time.time()
            return self._counts

    def documents(self, q: str = "", language: str = "", stage: str = "",
                  source_prefix: str = "", status: str = "",
                  translation_status: str = "", sort: str = "author",
                  offset: int = 0, limit: int = 50) -> Dict[str, Any]:
        """A filtered, paginated page of documents with progress counts.

        ``status`` filters on *our* progress, not the bibliographic one:
        untranslated / partial / translated / unstyled. It is applied after the
        SQL because it depends on the cached segment counts.
        """
        where, args = [], []
        if q:
            where.append("(title LIKE ? OR IFNULL(author,'') LIKE ? OR IFNULL(source,'') LIKE ?)")
            args += [f"%{q}%"] * 3
        if language:
            where.append("language = ?")
            args.append(language)
        if stage:
            where.append("language_stage = ?")
            args.append(stage)
        if source_prefix:
            where.append("source LIKE ?")
            args.append(f"{source_prefix}%")
        if translation_status:
            where.append("translation_status = ?")
            args.append(translation_status)
        sql = "SELECT * FROM documents"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += {
            "author": " ORDER BY IFNULL(author,'zzz'), title",
            "title": " ORDER BY title",
            "id": " ORDER BY id",
            "newest": " ORDER BY id DESC",
        }.get(sort, " ORDER BY IFNULL(author,'zzz'), title")

        with self._lock:
            rows = self.conn.execute(sql, args).fetchall()
        counts = self.counts()
        items = [_doc_dict(r, counts.get(r["id"], {})) for r in rows]
        if status:
            items = [d for d in items if _matches_status(d, status)]
        total = len(items)
        return {"total": total, "offset": offset, "limit": limit,
                "items": items[offset:offset + limit]}

    def document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM documents WHERE id = ?",
                                    (doc_id,)).fetchone()
            if row is None:
                return None
            secs = self.conn.execute(
                """SELECT sec.id, sec.label, sec.ord, COUNT(seg.id) AS n,
                          SUM(CASE WHEN seg.english_text IS NOT NULL
                                    AND seg.english_text <> '' THEN 1 ELSE 0 END) AS n_tr
                   FROM sections sec LEFT JOIN segments seg ON seg.section_id = sec.id
                   WHERE sec.doc_id = ? GROUP BY sec.id ORDER BY sec.ord""",
                (doc_id,),
            ).fetchall()
        doc = _doc_dict(row, self.counts().get(doc_id, {}))
        doc["sections"] = [
            {"id": s["id"], "label": s["label"], "order": s["ord"],
             "segments": s["n"], "translated": s["n_tr"] or 0,
             # 1-based position, which is what --section-range on the scripts means
             "number": i + 1}
            for i, s in enumerate(secs)
        ]
        return doc

    def titles(self, doc_ids) -> Dict[int, str]:
        """{doc_id: "Author — Title"} for the given ids (the job chip's headline)."""
        ids = sorted({int(i) for i in doc_ids})
        if not ids:
            return {}
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id, title, author FROM documents WHERE id IN ({','.join('?' * len(ids))})",
                ids).fetchall()
        return {r["id"]: (f"{r['author']} — {r['title']}" if r["author"] else r["title"])
                for r in rows}

    def segments(self, doc_id: int, offset: int = 0, limit: int = 300,
                 section_id: Optional[int] = None) -> Dict[str, Any]:
        """A page of segments in reading order, for the side-by-side reader.

        Paginated because single documents here run to tens of thousands of
        segments; the reader fetches the next page as you scroll.
        """
        where = "sec.doc_id = ?"
        args: List[Any] = [doc_id]
        if section_id:
            where += " AND sec.id = ?"
            args.append(section_id)
        with self._lock:
            total = self.conn.execute(
                f"SELECT COUNT(*) FROM segments seg JOIN sections sec "
                f"ON seg.section_id = sec.id WHERE {where}", args
            ).fetchone()[0]
            rows = self.conn.execute(
                f"""SELECT seg.*, sec.label AS section_label, sec.ord AS section_ord
                    FROM segments seg JOIN sections sec ON seg.section_id = sec.id
                    WHERE {where} ORDER BY sec.ord, seg.ord LIMIT ? OFFSET ?""",
                [*args, limit, offset],
            ).fetchall()
        return {
            "total": total, "offset": offset, "limit": limit,
            "items": [{
                "id": r["id"], "latin": r["latin_text"], "english": r["english_text"],
                "styled": r["english_styled"], "style_label": r["style_label"],
                "scansion": r["scansion"], "section": r["section_label"],
                "source_loc": r["source_loc"],
            } for r in rows],
        }

    # -- facets & totals -----------------------------------------------------

    def facets(self) -> Dict[str, Any]:
        """Distinct values for the filter controls, plus library-wide totals."""
        with self._lock:
            langs = [dict(r) for r in self.conn.execute(
                "SELECT language AS value, COUNT(*) AS n FROM documents "
                "GROUP BY language ORDER BY n DESC")]
            stages = [dict(r) for r in self.conn.execute(
                "SELECT language_stage AS value, COUNT(*) AS n FROM documents "
                "GROUP BY language_stage ORDER BY n DESC")]
            # Sources carry a per-work suffix ("Perseus (tlg2042.tlg008)"), so
            # group on the part before the parenthesis to get one row per corpus.
            sources = [dict(r) for r in self.conn.execute(
                """SELECT TRIM(CASE WHEN INSTR(source,'(') > 0
                             THEN SUBSTR(source, 1, INSTR(source,'(') - 1)
                             ELSE source END) AS value, COUNT(*) AS n
                   FROM documents WHERE source IS NOT NULL
                   GROUP BY value ORDER BY n DESC LIMIT 60""")]
            n_docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        counts = self.counts()
        segs = sum(c["segments"] for c in counts.values())
        tr = sum(c["translated"] for c in counts.values())
        st = sum(c["styled"] for c in counts.values())
        return {
            "languages": langs, "stages": stages, "sources": sources,
            "totals": {"documents": n_docs, "segments": segs, "translated": tr,
                       "styled": st, "untranslated": segs - tr},
        }

    def close(self) -> None:
        with self._lock:
            self.conn.close()


_OCR_TAG = re.compile(r"\[OCR: ([^\]]+)\]")


def ocr_of(source: Optional[str]) -> Optional[str]:
    """How a document's text was made from page images, or None if it was not.

    New scan connectors stamp ``[OCR: engine]`` into ``source``; older documents
    (archive.org dumps, hand-fixed scans) are recognised by the word OCR.
    """
    if not source:
        return None
    m = _OCR_TAG.search(source)
    if m:
        return m.group(1)
    return "ocr" if "OCR" in source else None


def _doc_dict(row: sqlite3.Row, counts: Dict[str, int]) -> Dict[str, Any]:
    n = counts.get("segments", 0)
    tr = counts.get("translated", 0)
    return {
        "id": row["id"], "title": row["title"], "author": row["author"],
        "century": row["century"], "genre": row["genre"], "language": row["language"],
        "language_stage": row["language_stage"], "source": row["source"],
        "ocr": ocr_of(row["source"]),
        "shelfmark": row["shelfmark"], "license": row["license"],
        "has_existing_translation": bool(row["has_existing_translation"]),
        "translation_status": row["translation_status"],
        "translation_evidence": (row["translation_evidence"]
                                 if "translation_evidence" in row.keys() else None),
        "segments": n, "translated": tr, "styled": counts.get("styled", 0),
        "percent": (100.0 * tr / n) if n else 0.0,
    }


def _matches_status(doc: Dict[str, Any], status: str) -> bool:
    n, tr, st = doc["segments"], doc["translated"], doc["styled"]
    if status == "untranslated":
        return tr == 0 and n > 0
    if status == "partial":
        return 0 < tr < n
    if status == "translated":
        return n > 0 and tr >= n
    if status == "unstyled":
        return tr > 0 and st == 0
    if status == "empty":
        return n == 0
    return True
