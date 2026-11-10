"""Quotation leads: what the mining job found, what the local model made of it, and your verdict.

Kept in data/leads.db, apart from corpus.db, so mining and sorting jobs can run
beside a translation. Three layers per lead, each written by a different step:

  found     scripts/mine_quotations.py run   (the matcher; re-runs refresh these)
  tag       scripts/triage_leads.py          (local model; a re-run skips tagged leads)
  verdict   the Leads tab                    (yours; nothing overwrites it)

Two kinds of lead:

  citation  a text names a DFHG author beside a citing formula, and the nearby
            words match no DFHG fragment
  reuse     a DFHG fragment's wording appears in a text DFHG doesn't name as
            its source
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "leads.db")
VERDICTS = ("new", "collected", "not_quotation", "unsure")
SORTS = {
    "default": "CASE WHEN tag_json IS NULL THEN 1 ELSE 0 END, rank_score DESC, id",
    "witness": "witness, id",
    "author": "author, id",
    "recent": "id DESC",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    key          TEXT    NOT NULL UNIQUE,
    kind         TEXT    NOT NULL,          -- citation | reuse
    witness      TEXT    NOT NULL,          -- registry name, ingest.quotations.WITNESSES
    loc          TEXT,
    author       TEXT,                      -- DFHG author
    fragment     TEXT,                      -- DFHG fragment number (reuse)
    name         TEXT,                      -- the name as written (citation)
    formula      TEXT,                      -- verb | book (citation)
    score        REAL,                      -- dfhg_share (citation) or coverage (reuse)
    context      TEXT    NOT NULL,
    fragment_text TEXT,                     -- reuse: the DFHG fragment
    dfhg_source  TEXT,                      -- reuse: where DFHG says it comes from
    found_at     TEXT    NOT NULL,
    tag_json     TEXT,                      -- the model's tags, see core/lead_triage.py
    tag_model    TEXT,
    tagged_at    TEXT,
    rank_score   REAL    NOT NULL DEFAULT 0,
    verdict      TEXT,
    verdict_note TEXT,
    verdict_at   TEXT,
    stale        INTEGER NOT NULL DEFAULT 0  -- reviewed, but the latest run no longer finds it
);
CREATE INDEX IF NOT EXISTS idx_leads_witness ON leads(witness);
CREATE INDEX IF NOT EXISTS idx_leads_untagged ON leads(tag_json) WHERE tag_json IS NULL;
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def lead_key(row: Dict[str, Any]) -> str:
    if row["kind"] == "citation":
        return f"citation|{row['witness']}|{row['pos']}|{row['author']}"
    return f"reuse|{row['witness']}|{row['author']}|{row['fragment']}"


class LeadStore:
    def __init__(self, path: str = DEFAULT_DB):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        columns = {r[1] for r in self.conn.execute("PRAGMA table_info(leads)")}
        if "stale" not in columns:   # databases made before the column existed
            self.conn.execute("ALTER TABLE leads ADD COLUMN stale INTEGER NOT NULL DEFAULT 0")
            self.conn.commit()
        self._lock = threading.Lock()

    def close(self) -> None:
        self.conn.close()

    # -- written by the mining job ---------------------------------------------

    def upsert(self, rows: Iterable[Dict[str, Any]]) -> int:
        """Add or refresh leads. Tags and verdicts on existing leads are kept."""
        n = 0
        with self._lock:
            for r in rows:
                self.conn.execute(
                    """INSERT INTO leads (key, kind, witness, loc, author, fragment, name, formula, score,
                                          context, fragment_text, dfhg_source, found_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(key) DO UPDATE SET loc=excluded.loc, score=excluded.score,
                           context=excluded.context, fragment_text=excluded.fragment_text,
                           dfhg_source=excluded.dfhg_source""",
                    (lead_key(r), r["kind"], r["witness"], r.get("loc"), r.get("author"), r.get("fragment"),
                     r.get("name"), r.get("formula"), r.get("score"), r["context"], r.get("fragment_text"),
                     r.get("dfhg_source"), _now()))
                n += 1
            self.conn.commit()
        return n

    def replace_witness(self, witness: str, rows: List[Dict[str, Any]]) -> Dict[str, int]:
        """Store one text's fresh results. Leads it no longer produces are dropped,
        unless you gave them a verdict: those are kept, marked stale."""
        keys = {lead_key(r) for r in rows}
        self.upsert(rows)
        with self._lock:
            old = self.conn.execute("SELECT id, key, verdict FROM leads WHERE witness=?", (witness,)).fetchall()
            gone = [r for r in old if r["key"] not in keys]
            dropped = [r["id"] for r in gone if r["verdict"] is None]
            self.conn.executemany("DELETE FROM leads WHERE id=?", [(i,) for i in dropped])
            self.conn.execute("UPDATE leads SET stale=1 WHERE witness=?", (witness,))
            self.conn.executemany("UPDATE leads SET stale=0 WHERE key=?", [(k,) for k in keys])
            self.conn.commit()
        return {"stored": len(rows), "dropped": len(dropped), "kept_reviewed": len(gone) - len(dropped)}

    # -- written by the sorting job --------------------------------------------

    def to_tag(self, force: bool = False, witness: Optional[str] = None, kind: Optional[str] = None,
               limit: Optional[int] = None) -> List[sqlite3.Row]:
        sql, args = "SELECT * FROM leads WHERE 1=1", []
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        if not force:
            sql += " AND tag_json IS NULL"
        if witness:
            sql += " AND witness = ?"
            args.append(witness)
        sql += " ORDER BY id"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        with self._lock:
            return self.conn.execute(sql, args).fetchall()

    def set_tag(self, lead_id: int, tag: Dict[str, Any], model: str, rank_score: float) -> None:
        with self._lock:
            self.conn.execute("UPDATE leads SET tag_json=?, tag_model=?, tagged_at=?, rank_score=? WHERE id=?",
                              (json.dumps(tag, ensure_ascii=False), model, _now(), rank_score, lead_id))
            self.conn.commit()

    # -- written by you ----------------------------------------------------------

    def set_verdict(self, lead_id: int, verdict: str, note: str = "") -> Optional[Dict[str, Any]]:
        if verdict and verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS} or empty")
        with self._lock:
            cur = self.conn.execute("UPDATE leads SET verdict=?, verdict_note=?, verdict_at=? WHERE id=?",
                                    (verdict or None, note or None, _now() if verdict else None, lead_id))
            self.conn.commit()
        return self.get(lead_id) if cur.rowcount else None

    # -- read by the Leads tab ---------------------------------------------------

    def get(self, lead_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM leads WHERE id=?", (lead_id,)).fetchone()
        return _to_dict(row) if row else None

    def search(self, kind: str = "", witness: str = "", author: str = "", verdict: str = "", tag: str = "",
               sort: str = "default", limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        where, args = ["1=1"], []
        if kind:
            where.append("kind = ?"); args.append(kind)
        if witness:
            where.append("witness = ?"); args.append(witness)
        if author:
            where.append("author LIKE ?"); args.append(f"%{author}%")
        if verdict == "none":
            where.append("verdict IS NULL")
        elif verdict:
            where.append("verdict = ?"); args.append(verdict)
        if tag == "untagged":
            where.append("tag_json IS NULL")
        elif tag:   # e.g. "same_person:no" or "reference:quotation"
            field, _, value = tag.partition(":")
            if field not in ("same_person", "reference", "relation"):
                raise ValueError(f"unknown tag filter {tag!r}")
            where.append(f"json_extract(tag_json, '$.{field}') = ?"); args.append(value)
        clause = " AND ".join(where)
        with self._lock:
            total = self.conn.execute(f"SELECT COUNT(*) FROM leads WHERE {clause}", args).fetchone()[0]
            rows = self.conn.execute(f"SELECT * FROM leads WHERE {clause} ORDER BY {SORTS.get(sort, SORTS['default'])} "
                                     "LIMIT ? OFFSET ?", args + [limit, offset]).fetchall()
        return {"total": total, "items": [_to_dict(r) for r in rows]}

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            c = self.conn
            one = lambda sql: c.execute(sql).fetchone()[0]                   # noqa: E731
            return {
                "total": one("SELECT COUNT(*) FROM leads"),
                "untagged": one("SELECT COUNT(*) FROM leads WHERE tag_json IS NULL"),
                "reviewed": one("SELECT COUNT(*) FROM leads WHERE verdict IS NOT NULL"),
                "by_witness": dict(c.execute("SELECT witness, COUNT(*) FROM leads GROUP BY witness").fetchall()),
                "by_kind": dict(c.execute("SELECT kind, COUNT(*) FROM leads GROUP BY kind").fetchall()),
                "by_verdict": dict(c.execute("SELECT COALESCE(verdict,'none'), COUNT(*) FROM leads "
                                             "GROUP BY 1").fetchall()),
            }


def _to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["tag"] = json.loads(d.pop("tag_json")) if d.get("tag_json") else None
    return d
