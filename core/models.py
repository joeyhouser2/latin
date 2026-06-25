"""Data model for the Latin corpus.

The spine is Document -> Section -> Segment. A Segment is one Latin sentence and
is the unit of embedding, translation, and side-by-side alignment all at once:
because we translate per-segment, Latin segment *i* always renders next to
English segment *i* with no re-alignment needed.

Ids are assigned by the store (SQLite autoincrement). An id of None means the
object has not been persisted yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


# Coarse language stage, used both as metadata and as a discovery filter.
# "archaic" = Homeric/early Greek (and archaic Latin), a dialect our Greek models
# never trained on; see docs/roadmap-verse-stylizer.md.
LANGUAGE_STAGES = ("archaic", "classical", "late_antique", "medieval",
                   "early_modern", "ancient", "unknown")

# Primary language of a work's text (ISO-ish). "la" = Latin, "grc" = ancient Greek.
LANGUAGES = {"la": "Latin", "grc": "Greek"}


@dataclass
class Segment:
    """One Latin sentence plus its (optional) translation and alignment refs."""

    latin_text: str
    order: int                      # position within the parent section
    english_text: Optional[str] = None
    english_styled: Optional[str] = None    # optional stylized variant (Victorian prose, verse...)
    style_label: Optional[str] = None       # which Stylizer preset produced english_styled
    source_loc: Optional[str] = None        # e.g. "Book 1, sect. 3" or "f. 12r"
    image_region: Optional[str] = None      # IIIF region for a manuscript page (Phase 5)
    embed_text: Optional[str] = None        # markup-stripped copy for embedding (None => use latin_text)
    scansion: Optional[str] = None          # metrical pattern of this verse line (None => not scanned/prose)
    id: Optional[int] = None
    section_id: Optional[int] = None

    @property
    def is_translated(self) -> bool:
        return bool(self.english_text and self.english_text.strip())

    @property
    def is_styled(self) -> bool:
        return bool(self.english_styled and self.english_styled.strip())

    @property
    def text_for_embedding(self) -> str:
        return self.embed_text or self.latin_text


@dataclass
class Section:
    """A structural unit of a work: a book, chapter, folio, etc."""

    label: str                      # e.g. "Book 1", "Chapter 3", "f. 12r"
    order: int
    segments: List[Segment] = field(default_factory=list)
    id: Optional[int] = None
    doc_id: Optional[int] = None


@dataclass
class Document:
    """A complete work with bibliographic metadata."""

    title: str
    author: Optional[str] = None
    century: Optional[int] = None           # negative = BCE; e.g. -1, 4, 13
    genre: Optional[str] = None             # e.g. "history", "sermon", "letter"
    language: str = "la"                    # primary language: "la" or "grc"
    language_stage: str = "unknown"         # one of LANGUAGE_STAGES
    source: Optional[str] = None            # where the text came from
    shelfmark: Optional[str] = None         # manuscript shelfmark, if any
    license: Optional[str] = None
    has_existing_translation: bool = False  # is a known English translation already published?
    sections: List[Section] = field(default_factory=list)
    id: Optional[int] = None

    @property
    def language_name(self) -> str:
        return LANGUAGES.get(self.language, "Latin")

    def iter_segments(self):
        """Yield every segment in reading order across all sections."""
        for section in sorted(self.sections, key=lambda s: s.order):
            for segment in sorted(section.segments, key=lambda seg: seg.order):
                yield segment
