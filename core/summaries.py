"""Storage and search for LLM-written document summaries.

Summaries live in their own SQLite file, ``data/summaries.db``, never in
corpus.db. That is what lets a summarize job run on one GPU while a translation
job writes corpus.db on the other: the summarizer only *reads* the corpus, so
the corpus keeps exactly one writer.

Two levels per document:

* ``document`` -- one paragraph for the whole work;
* ``part`` -- one per chunk of the text, with the segment offsets it covers,
  so a search hit can open the reader at the right place. For a 21,000-segment
  Analecta Hymnica volume that is the difference between "somewhere in this
  book" and "these forty hymns".

Search is hybrid, because the two halves fail differently. SQLite FTS5 (BM25,
Porter-stemmed) is exact about names -- *Ambrosius*, *Novalesa* -- and knows
nothing about meaning. Embeddings (``nomic-embed-text`` through Ollama) find
"lending money at interest" when the summary says "usury", and blur proper
names together. Results from both are merged by reciprocal rank fusion, which
needs no score calibration between two incomparable scales. If Ollama is not
reachable at query time the search quietly degrades to keyword-only and says
so.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
# LATIN_SUMMARIES_DB: point the web app at a copy (testing without the live file).
DEFAULT_DB = Path(os.environ.get("LATIN_SUMMARIES_DB") or REPO_ROOT / "data" / "summaries.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS summaries (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id           INTEGER NOT NULL,
    level            TEXT    NOT NULL CHECK (level IN ('document', 'part')),
    part_index       INTEGER,          -- 0-based; NULL for the document summary
    part_count       INTEGER,
    seg_offset_first INTEGER,          -- position in reading order (0-based), for the reader
    seg_offset_last  INTEGER,
    section_first    INTEGER,          -- 1-based section numbers covered
    section_last     INTEGER,
    summary          TEXT    NOT NULL,
    topics           TEXT    NOT NULL DEFAULT '[]',   -- JSON list of strings
    model            TEXT    NOT NULL,
    source_mode      TEXT    NOT NULL,                -- 'both' | 'english'
    translated_count INTEGER NOT NULL,                -- doc's translated segments when written
    embed_model      TEXT,
    embedding        BLOB,                            -- float32, L2-normalised
    created_at       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_doc ON summaries(doc_id, level, part_index);

-- rowid = summaries.id. Title/author are copied in so a search for an author's
-- name finds their summaries even when the summary text never says it.
CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
    summary, topics, title, author,
    tokenize = 'porter unicode61 remove_diacritics 2'
);

-- How interesting a work's document summary makes it look (core/interest.py).
-- Keyed to the summary it rated: once a document is re-summarized the old
-- rating no longer joins and the work counts as unrated again.
CREATE TABLE IF NOT EXISTS interest (
    doc_id     INTEGER PRIMARY KEY,
    summary_id INTEGER NOT NULL,
    score      REAL    NOT NULL,      -- 1.0-10.0, computed from the axes
    subject INTEGER, story INTEGER, history INTEGER, neglect INTEGER,
    garbled    INTEGER NOT NULL DEFAULT 0,
    hook       TEXT,
    model      TEXT    NOT NULL,
    rated_at   TEXT    NOT NULL
);
"""

# Browse orders for the Summaries tab. "distinct" is computed, not stored.
SORTS = ("interest", "distinct", "recent", "title")
DISTINCT_K = 10     # neighbours averaged for the distinctiveness score

# Highlight sentinels for FTS5 snippet(): control characters that cannot occur
# in summary text, so the UI can HTML-escape everything and only then turn
# these into <mark> tags -- no path for markup to leak through.
HL_OPEN, HL_CLOSE = "\x02", "\x03"

RRF_K = 60          # the standard reciprocal-rank-fusion constant
SEM_MARGIN = 0.08   # a meaning-only hit must beat the query's median similarity by this


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_blob(vec: Sequence[float]) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm if norm else arr).tobytes()


class SummaryStore:
    def __init__(self, path: str | Path = DEFAULT_DB, readonly: bool = False):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        if not readonly:
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        # Cache of (ids, matrix) per embedding model, rebuilt when the row
        # count changes. A few thousand 768-d vectors is a few MB: brute-force
        # cosine in numpy is faster than any index at this size.
        self._vec_cache: Dict[str, Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._distinct_cache: Dict[str, Tuple[int, Dict[int, Tuple[float, int]]]] = {}

    # -- writing (summarize job) --------------------------------------------

    def find_part(self, doc_id: int, offset_first: int, offset_last: int,
                  model: str, source_mode: str, translated_count: int) -> Optional[sqlite3.Row]:
        """An existing part summary over exactly this span, made the same way.

        Chunk boundaries are deterministic for a given text, so a job that was
        cancelled half-way through a 300-part volume can pick up where it left
        off instead of paying for the first 150 parts again.
        """
        with self._lock:
            return self.conn.execute(
                """SELECT * FROM summaries WHERE doc_id=? AND level='part'
                   AND seg_offset_first=? AND seg_offset_last=? AND model=?
                   AND source_mode=? AND translated_count=?""",
                (doc_id, offset_first, offset_last, model, source_mode, translated_count),
            ).fetchone()

    def add(self, *, doc_id: int, level: str, summary: str, topics: List[str],
            model: str, source_mode: str, translated_count: int,
            title: str, author: Optional[str], part_index: Optional[int] = None,
            part_count: Optional[int] = None, seg_offset_first: Optional[int] = None,
            seg_offset_last: Optional[int] = None, section_first: Optional[int] = None,
            section_last: Optional[int] = None) -> int:
        with self._lock:
            if level == "document":
                self._delete_where("doc_id=? AND level='document'", (doc_id,))
            cur = self.conn.execute(
                """INSERT INTO summaries (doc_id, level, part_index, part_count,
                       seg_offset_first, seg_offset_last, section_first, section_last,
                       summary, topics, model, source_mode, translated_count, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (doc_id, level, part_index, part_count, seg_offset_first, seg_offset_last,
                 section_first, section_last, summary, json.dumps(topics), model,
                 source_mode, translated_count, _now()),
            )
            sid = cur.lastrowid
            self.conn.execute(
                "INSERT INTO summaries_fts (rowid, summary, topics, title, author) "
                "VALUES (?,?,?,?,?)",
                (sid, summary, " ; ".join(topics), title or "", author or ""),
            )
            self.conn.commit()
            return sid

    def prune_stale(self, doc_id: int, keep_translated_count: int, model: str,
                    source_mode: str) -> int:
        """Drop a document's summaries that no longer match how it was just summarized
        (older translation state, another model) -- so search never mixes them."""
        with self._lock:
            n = self._delete_where(
                "doc_id=? AND NOT (translated_count=? AND model=? AND source_mode=?)",
                (doc_id, keep_translated_count, model, source_mode))
            self.conn.commit()
            return n

    def clear_doc(self, doc_id: int) -> int:
        with self._lock:
            n = self._delete_where("doc_id=?", (doc_id,))
            self.conn.execute("DELETE FROM interest WHERE doc_id=?", (doc_id,))
            self.conn.commit()
            return n

    def _delete_where(self, where: str, args: tuple) -> int:
        ids = [r[0] for r in self.conn.execute(f"SELECT id FROM summaries WHERE {where}", args)]
        if ids:
            marks = ",".join("?" * len(ids))
            self.conn.execute(f"DELETE FROM summaries_fts WHERE rowid IN ({marks})", ids)
            self.conn.execute(f"DELETE FROM summaries WHERE id IN ({marks})", ids)
        return len(ids)

    def set_embeddings(self, embed_model: str, pairs: Iterable[Tuple[int, Sequence[float]]]) -> None:
        with self._lock:
            self.conn.executemany(
                "UPDATE summaries SET embed_model=?, embedding=? WHERE id=?",
                [(embed_model, _to_blob(v), sid) for sid, v in pairs])
            self.conn.commit()

    def set_interest(self, doc_id: int, summary_id: int, rating: Dict[str, Any],
                     model: str) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO interest (doc_id, summary_id, score, subject,
                       story, history, neglect, garbled, hook, model, rated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (doc_id, summary_id, rating["score"], rating["subject"], rating["story"],
                 rating["history"], rating["neglect"], int(rating["garbled"]),
                 rating["hook"], model, _now()))
            self.conn.commit()

    def documents_to_rate(self, force: bool = False,
                          doc_ids: Sequence[int] = ()) -> List[sqlite3.Row]:
        """Document summaries with no current interest rating (all of them with force)."""
        where = ["s.level='document'"]
        args: List[Any] = []
        if not force:
            where.append("(i.summary_id IS NULL OR i.summary_id <> s.id)")
        if doc_ids:
            where.append(f"s.doc_id IN ({','.join('?' * len(doc_ids))})")
            args.extend(doc_ids)
        with self._lock:
            return self.conn.execute(
                f"""SELECT s.id, s.doc_id, s.summary, s.topics, f.title, f.author
                    FROM summaries s JOIN summaries_fts f ON f.rowid = s.id
                    LEFT JOIN interest i ON i.doc_id = s.doc_id
                    WHERE {' AND '.join(where)} ORDER BY s.doc_id""", args).fetchall()

    def document_row(self, doc_id: int) -> Optional[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM summaries WHERE doc_id=? AND level='document'", (doc_id,)
            ).fetchone()

    # -- reading (web app) ---------------------------------------------------

    def for_document(self, doc_id: int) -> Dict[str, Any]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM summaries WHERE doc_id=? ORDER BY level, part_index",
                (doc_id,)).fetchall()
        doc = next((r for r in rows if r["level"] == "document"), None)
        return {"document": _row_dict(doc) if doc else None,
                "parts": [_row_dict(r) for r in rows if r["level"] == "part"]}

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            r = self.conn.execute(
                """SELECT COUNT(DISTINCT doc_id) docs,
                          SUM(level='document') documents, SUM(level='part') parts,
                          SUM(embedding IS NOT NULL) embedded
                   FROM summaries""").fetchone()
            models = [x[0] for x in self.conn.execute(
                "SELECT embed_model FROM summaries WHERE embed_model IS NOT NULL "
                "GROUP BY embed_model ORDER BY COUNT(*) DESC")]
            rated = self.conn.execute(
                """SELECT COUNT(*) FROM interest i JOIN summaries s
                   ON s.id = i.summary_id AND s.level='document'""").fetchone()[0]
        return {"docs": r["docs"] or 0, "documents": r["documents"] or 0,
                "parts": r["parts"] or 0, "embedded": r["embedded"] or 0,
                "rated": rated, "embed_model": models[0] if models else None}

    def summarized_doc_ids(self) -> Dict[int, int]:
        """doc_id -> translated_count at the time its document summary was written."""
        with self._lock:
            return {r[0]: r[1] for r in self.conn.execute(
                "SELECT doc_id, translated_count FROM summaries WHERE level='document'")}

    def browse(self, sort: str = "recent", limit: int = 30, offset: int = 0,
               embed_model: Optional[str] = None) -> Dict[str, Any]:
        """Whole-work summaries in one of SORTS, a page at a time."""
        order = {"interest": "i.score IS NULL, i.score DESC, s.created_at DESC",
                 "title": "f.title COLLATE NOCASE",
                 }.get(sort, "s.created_at DESC")
        select = """SELECT s.*, f.title, f.author FROM summaries s
                    JOIN summaries_fts f ON f.rowid = s.id"""
        with self._lock:
            total = self.conn.execute(
                "SELECT COUNT(*) FROM summaries WHERE level='document'").fetchone()[0]
            if sort == "distinct":
                ranks = self._distinct(embed_model)
                ids = sorted((r[0] for r in self.conn.execute(
                                 "SELECT id FROM summaries WHERE level='document'")),
                             key=lambda sid: -ranks.get(sid, (-1.0, 0))[0])[offset:offset + limit]
                rows = {r["id"]: r for r in self.conn.execute(
                    f"{select} WHERE s.id IN ({','.join('?' * len(ids))})", ids)}
                page = [rows[i] for i in ids if i in rows]
            else:
                page = self.conn.execute(
                    f"""{select}
                        LEFT JOIN interest i ON i.doc_id = s.doc_id AND i.summary_id = s.id
                        WHERE s.level='document' ORDER BY {order} LIMIT ? OFFSET ?""",
                    (limit, offset)).fetchall()
        items = self._annotate([_row_dict(r) for r in page], embed_model)
        return {"items": items, "total": total}

    def _annotate(self, items: List[Dict[str, Any]],
                  embed_model: Optional[str]) -> List[Dict[str, Any]]:
        """Attach each item's work-level interest rating and distinctiveness.

        A part inherits its work's: the rating is of the work, and a part of a
        dull volume is no more interesting for matching a query."""
        doc_ids = sorted({d["doc_id"] for d in items})
        if not doc_ids:
            return items
        with self._lock:
            rated = {r["doc_id"]: dict(r) for r in self.conn.execute(
                f"""SELECT i.doc_id, i.score, i.subject, i.story, i.history, i.neglect,
                           i.garbled, i.hook, s.id AS work_id
                    FROM interest i JOIN summaries s
                      ON s.id = i.summary_id AND s.level='document'
                    WHERE i.doc_id IN ({','.join('?' * len(doc_ids))})""", doc_ids)}
            works = {r[0]: r[1] for r in self.conn.execute(
                f"""SELECT doc_id, id FROM summaries WHERE level='document'
                    AND doc_id IN ({','.join('?' * len(doc_ids))})""", doc_ids)}
            ranks = self._distinct(embed_model)
        for d in items:
            r = rated.get(d["doc_id"])
            d["interest"] = ({k: r[k] for k in ("score", "subject", "story", "history",
                                                 "neglect", "hook")} | {"garbled": bool(r["garbled"])}
                             if r else None)
            # Percentile: "more distinctive than N% of the library's works".
            d["distinct_pct"] = ranks.get(works.get(d["doc_id"]), (None, None))[1]
        return items

    def _distinct(self, embed_model: Optional[str]) -> Dict[int, Tuple[float, int]]:
        """Work summary id -> (distinctiveness, percentile 0-100). Caller holds the lock.

        Distinctiveness is 1 minus the mean cosine similarity to the work's
        DISTINCT_K nearest other works: a forty-first hymn volume sits close to
        forty siblings, a travel account to India close to nothing. It measures
        "unusual for this library", which is not the same as interesting --
        Cicero is an outlier here too -- so it is a separate sort, not a factor
        in the interest score.
        """
        if not embed_model:
            return {}
        n = self.conn.execute(
            "SELECT COUNT(*) FROM summaries WHERE level='document' AND embed_model=?",
            (embed_model,)).fetchone()[0]
        cached = self._distinct_cache.get(embed_model)
        if cached and cached[0] == n:
            return cached[1]
        rows = self.conn.execute(
            "SELECT id, embedding FROM summaries WHERE level='document' AND embed_model=?",
            (embed_model,)).fetchall()
        out: Dict[int, Tuple[float, int]] = {}
        if len(rows) > DISTINCT_K:
            mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            sims = mat @ mat.T
            np.fill_diagonal(sims, -np.inf)
            near = -np.sort(-sims, axis=1)[:, :DISTINCT_K].mean(axis=1)
            dist = 1.0 - near
            pct = (np.argsort(np.argsort(dist)) * 100) // len(dist)
            out = {int(r[0]): (float(dist[i]), int(pct[i])) for i, r in enumerate(rows)}
        self._distinct_cache[embed_model] = (n, out)
        return out

    def search(self, q: str, *, level: str = "", limit: int = 20,
               query_vec: Optional[Sequence[float]] = None,
               embed_model: Optional[str] = None, sort: str = "") -> Dict[str, Any]:
        """Hybrid search. Returns {"mode": "hybrid"|"keyword", "items": [...]}.

        ``sort`` (one of SORTS) reorders the top ``limit`` matches; empty keeps
        relevance order. Reordering only the matches, not the whole pool, keeps
        a weak match from climbing to the top just for being well rated."""
        pool = max(limit * 3, 50)
        # "hybrid" means meaning-search *ran*, not that it found something: a
        # query whose closest summaries all fall under the floor must not be
        # reported as "keyword only", which reads as meaning-search being down.
        mode = "hybrid" if query_vec is not None and embed_model else "keyword"
        kw = self._keyword(q, level, pool)
        sem: List[Tuple[int, float]] = []
        if query_vec is not None and embed_model:
            sem = self._semantic(query_vec, embed_model, level, pool)

        scores: Dict[int, float] = {}
        for rank, (sid, _) in enumerate(kw):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (sid, _) in enumerate(sem):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (RRF_K + rank + 1)
        ranked = sorted(scores, key=scores.get, reverse=True)[:limit]
        if not ranked:
            return {"mode": mode, "items": []}

        snippets = dict(kw)       # keyword hits carry a highlighted snippet
        marks = ",".join("?" * len(ranked))
        with self._lock:
            rows = {r["id"]: r for r in self.conn.execute(
                f"""SELECT s.*, f.title, f.author FROM summaries s
                    JOIN summaries_fts f ON f.rowid = s.id WHERE s.id IN ({marks})""",
                ranked)}
        items = []
        for sid in ranked:
            if sid not in rows:
                continue
            d = _row_dict(rows[sid])
            d["score"] = round(scores[sid], 5)
            d["match"] = ("both" if sid in snippets and any(s == sid for s, _ in sem)
                          else "keyword" if sid in snippets else "meaning")
            d["snippet"] = snippets.get(sid)
            items.append(d)
        items = self._annotate(items, embed_model)
        # Stable sorts (reverse=True too), so ties keep their relevance order.
        if sort == "interest":
            items.sort(key=lambda d: (d["interest"] or {}).get("score", 0), reverse=True)
        elif sort == "distinct":
            items.sort(key=lambda d: d["distinct_pct"] if d["distinct_pct"] is not None else -1,
                       reverse=True)
        elif sort == "recent":
            items.sort(key=lambda d: d["created_at"], reverse=True)
        elif sort == "title":
            items.sort(key=lambda d: (d["title"] or "").lower())
        return {"mode": mode, "items": items}

    def _keyword(self, q: str, level: str, limit: int) -> List[Tuple[int, str]]:
        match = _fts_query(q)
        if not match:
            return []
        sql = (f"""SELECT f.rowid, snippet(summaries_fts, 0, '{HL_OPEN}', '{HL_CLOSE}', '…', 40)
                   FROM summaries_fts f JOIN summaries s ON s.id = f.rowid
                   WHERE summaries_fts MATCH ? {"AND s.level = ?" if level else ""}
                   ORDER BY bm25(summaries_fts, 1.0, 1.5, 2.0, 1.5) LIMIT ?""")
        args: List[Any] = [match] + ([level] if level else []) + [limit]
        with self._lock:
            try:
                rows = self.conn.execute(sql, args).fetchall()
                if not rows and '* "' in match:
                    # Every term required found nothing: retry with any term, so
                    # a four-word query does not fail on one unlucky word.
                    args[0] = match.replace('* "', '* OR "')
                    rows = self.conn.execute(sql, args).fetchall()
            except sqlite3.OperationalError:
                return []
        return [(r[0], r[1]) for r in rows]

    def _semantic(self, query_vec: Sequence[float], embed_model: str, level: str,
                  limit: int) -> List[Tuple[int, float]]:
        ids, levels, mat = self._matrix(embed_model)
        if mat.size == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q /= (np.linalg.norm(q) or 1.0)
        sims = mat @ q
        valid = (levels == level) if level else np.ones(len(sims), dtype=bool)
        if not valid.any():
            return []
        # Keep only hits that stand out from *this query's* own baseline. An
        # absolute cutoff cannot work with these embeddings: similarity levels
        # shift per query, so "steam engines and railways" (nothing relevant
        # here) peaked at 0.604 while the correct hit for "Chrysogon" was 0.548.
        # What separates them is the margin over the query's median: measured
        # on this library, true matches sat +0.098 to +0.192 above it, junk
        # +0.038 to +0.061. Revisit SEM_MARGIN once there are thousands of
        # summaries and the median is steadier.
        floor = float(np.median(sims[valid])) + SEM_MARGIN
        sims = np.where(valid, sims, -1.0)
        top = np.argsort(-sims)[:limit]
        return [(int(ids[i]), float(sims[i])) for i in top if sims[i] >= floor]

    def _matrix(self, embed_model: str):
        with self._lock:
            n = self.conn.execute(
                "SELECT COUNT(*) FROM summaries WHERE embed_model=?", (embed_model,)
            ).fetchone()[0]
            cached = self._vec_cache.get(embed_model)
            if cached and cached[0] == n:
                return cached[1], cached[2], cached[3]
            rows = self.conn.execute(
                "SELECT id, level, embedding FROM summaries WHERE embed_model=?",
                (embed_model,)).fetchall()
        if not rows:
            return np.zeros(0), np.zeros(0), np.zeros((0, 0), dtype=np.float32)
        ids = np.array([r[0] for r in rows])
        levels = np.array([r[1] for r in rows])
        mat = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        self._vec_cache[embed_model] = (n, ids, levels, mat)
        return ids, levels, mat

    def close(self) -> None:
        with self._lock:
            self.conn.close()


def _fts_query(q: str) -> str:
    """User text -> a safe FTS5 query: every word required, each prefix-matched.

    Quoting each token neutralises FTS5 syntax (a stray quote, AND/NEAR/column
    filters) so no input can raise a query error; the trailing ``*`` lets
    "usur" find usury/usurer/usurious.
    """
    words = re.findall(r"[\w'-]+", q, flags=re.UNICODE)
    words = [w.strip("'-") for w in words if len(w.strip("'-")) > 1]
    # Function words carry no subject, and in the any-word fallback they match
    # nearly every summary ("funeral elegy for a friend" hit on "for").
    content = [w for w in words if w.lower() not in _STOPWORDS]
    return " ".join(f'"{w}"*' for w in (content or words)[:12])


_STOPWORDS = frozenset("""a an and are as at be by for from has have in into is it its of
on or that the their this to was were which with about who whom what""".split())


def _row_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    d = {k: row[k] for k in row.keys() if k != "embedding"}
    d["topics"] = json.loads(d.get("topics") or "[]")
    return d
