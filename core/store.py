"""SQLite persistence for the corpus.

The store owns the canonical text + metadata. The FAISS index (see vectorstore)
holds only vectors keyed by segment id, so the store is the source of truth and
the vector index can always be rebuilt from it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional, Iterable

from .models import Document, Section, Segment


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    title                    TEXT NOT NULL,
    author                   TEXT,
    century                  INTEGER,
    genre                    TEXT,
    language                 TEXT NOT NULL DEFAULT 'la',
    language_stage           TEXT NOT NULL DEFAULT 'unknown',
    source                   TEXT,
    shelfmark                TEXT,
    license                  TEXT,
    has_existing_translation INTEGER NOT NULL DEFAULT 0,
    translation_status       TEXT NOT NULL DEFAULT 'unknown'
);

CREATE TABLE IF NOT EXISTS sections (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    label  TEXT NOT NULL,
    ord    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id     INTEGER NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    ord            INTEGER NOT NULL,
    latin_text     TEXT NOT NULL,
    english_text   TEXT,
    english_styled TEXT,
    style_label    TEXT,
    source_loc     TEXT,
    image_region   TEXT,
    embed_text     TEXT,
    scansion       TEXT
);

CREATE INDEX IF NOT EXISTS idx_sections_doc ON sections(doc_id);
CREATE INDEX IF NOT EXISTS idx_segments_section ON segments(section_id);
"""


class Store:
    """Thin SQLite wrapper. Use as a context manager or call close()."""

    def __init__(self, path: str = "data/corpus.db"):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Additive migrations for databases created by an older schema."""
        seg_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(segments)")}
        if "embed_text" not in seg_cols:
            self.conn.execute("ALTER TABLE segments ADD COLUMN embed_text TEXT")
        for col in ("english_styled", "style_label", "scansion"):
            if col not in seg_cols:
                self.conn.execute(f"ALTER TABLE segments ADD COLUMN {col} TEXT")
        doc_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(documents)")}
        if "language" not in doc_cols:
            self.conn.execute(
                "ALTER TABLE documents ADD COLUMN language TEXT NOT NULL DEFAULT 'la'")
        if "translation_status" not in doc_cols:
            self.conn.execute(
                "ALTER TABLE documents ADD COLUMN translation_status "
                "TEXT NOT NULL DEFAULT 'unknown'")

    # -- write ---------------------------------------------------------------

    def add_document(self, doc: Document) -> Document:
        """Insert a document with all its sections and segments. Assigns ids."""
        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO documents
               (title, author, century, genre, language, language_stage, source,
                shelfmark, license, has_existing_translation, translation_status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (doc.title, doc.author, doc.century, doc.genre, doc.language,
             doc.language_stage, doc.source, doc.shelfmark, doc.license,
             int(doc.has_existing_translation), doc.translation_status),
        )
        doc.id = cur.lastrowid

        for section in sorted(doc.sections, key=lambda s: s.order):
            cur.execute(
                "INSERT INTO sections (doc_id, label, ord) VALUES (?,?,?)",
                (doc.id, section.label, section.order),
            )
            section.id = cur.lastrowid
            section.doc_id = doc.id

            for seg in sorted(section.segments, key=lambda s: s.order):
                cur.execute(
                    """INSERT INTO segments
                       (section_id, ord, latin_text, english_text, english_styled,
                        style_label, source_loc, image_region, embed_text, scansion)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (section.id, seg.order, seg.latin_text, seg.english_text,
                     seg.english_styled, seg.style_label, seg.source_loc,
                     seg.image_region, seg.embed_text, seg.scansion),
                )
                seg.id = cur.lastrowid
                seg.section_id = section.id

        self.conn.commit()
        return doc

    def set_translation(self, segment_id: int, english_text: str) -> None:
        self.conn.execute(
            "UPDATE segments SET english_text = ? WHERE id = ?",
            (english_text, segment_id),
        )
        self.conn.commit()

    def set_translations(self, pairs: Iterable[tuple]) -> None:
        """Bulk-write (segment_id, english_text) pairs in one transaction."""
        self.conn.executemany(
            "UPDATE segments SET english_text = ? WHERE id = ?",
            [(en, sid) for sid, en in pairs],
        )
        self.conn.commit()

    def set_styled(self, triples: Iterable[tuple]) -> None:
        """Bulk-write (segment_id, english_styled, style_label) in one transaction."""
        self.conn.executemany(
            "UPDATE segments SET english_styled = ?, style_label = ? WHERE id = ?",
            [(styled, label, sid) for sid, styled, label in triples],
        )
        self.conn.commit()

    def set_scansions(self, pairs: Iterable[tuple]) -> None:
        """Bulk-write (segment_id, scansion) pairs in one transaction."""
        self.conn.executemany(
            "UPDATE segments SET scansion = ? WHERE id = ?",
            [(sc, sid) for sid, sc in pairs],
        )
        self.conn.commit()

    def set_embed_texts(self, pairs: Iterable[tuple]) -> None:
        """Bulk-write (segment_id, embed_text) pairs in one transaction."""
        self.conn.executemany(
            "UPDATE segments SET embed_text = ? WHERE id = ?",
            [(et, sid) for sid, et in pairs],
        )
        self.conn.commit()

    # -- read ----------------------------------------------------------------

    def get_document(self, doc_id: int) -> Optional[Document]:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        doc = self._row_to_document(row)

        sec_rows = self.conn.execute(
            "SELECT * FROM sections WHERE doc_id = ? ORDER BY ord", (doc_id,)
        ).fetchall()
        for sec_row in sec_rows:
            section = Section(
                label=sec_row["label"], order=sec_row["ord"],
                id=sec_row["id"], doc_id=doc_id,
            )
            seg_rows = self.conn.execute(
                "SELECT * FROM segments WHERE section_id = ? ORDER BY ord",
                (section.id,),
            ).fetchall()
            section.segments = [self._row_to_segment(r) for r in seg_rows]
            doc.sections.append(section)
        return doc

    def list_documents(self) -> List[Document]:
        """All documents as metadata-only stubs (no sections loaded)."""
        rows = self.conn.execute(
            "SELECT * FROM documents ORDER BY author, title"
        ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def get_segment(self, segment_id: int) -> Optional[Segment]:
        row = self.conn.execute(
            "SELECT * FROM segments WHERE id = ?", (segment_id,)
        ).fetchone()
        return self._row_to_segment(row) if row else None

    def iter_untranslated_segments(self) -> List[Segment]:
        rows = self.conn.execute(
            "SELECT * FROM segments WHERE english_text IS NULL OR english_text = ''"
        ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def iter_all_segments(self) -> List[Segment]:
        rows = self.conn.execute("SELECT * FROM segments").fetchall()
        return [self._row_to_segment(r) for r in rows]

    def document_id_for_segment(self, segment_id: int) -> Optional[int]:
        row = self.conn.execute(
            """SELECT d.id FROM documents d
               JOIN sections s ON s.doc_id = d.id
               JOIN segments seg ON seg.section_id = s.id
               WHERE seg.id = ?""",
            (segment_id,),
        ).fetchone()
        return row["id"] if row else None

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        keys = row.keys()
        return Document(
            id=row["id"], title=row["title"], author=row["author"],
            century=row["century"], genre=row["genre"],
            language=row["language"] if "language" in keys else "la",
            language_stage=row["language_stage"], source=row["source"],
            shelfmark=row["shelfmark"], license=row["license"],
            has_existing_translation=bool(row["has_existing_translation"]),
            translation_status=(row["translation_status"]
                                if "translation_status" in keys else "unknown"),
        )

    @staticmethod
    def _row_to_segment(row: sqlite3.Row) -> Segment:
        keys = row.keys()
        return Segment(
            id=row["id"], section_id=row["section_id"], order=row["ord"],
            latin_text=row["latin_text"], english_text=row["english_text"],
            english_styled=row["english_styled"] if "english_styled" in keys else None,
            style_label=row["style_label"] if "style_label" in keys else None,
            source_loc=row["source_loc"], image_region=row["image_region"],
            embed_text=row["embed_text"] if "embed_text" in keys else None,
            scansion=row["scansion"] if "scansion" in keys else None,
        )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
