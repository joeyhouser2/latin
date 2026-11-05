"""Connector interface.

A connector knows how to fetch and structure one kind of source. It produces an
unsegmented Document (sections with raw text), or a list of (section_label,
raw_text) pairs that the Library segments and embeds on ingest.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from core.models import Document, Section, Segment
from core.segmenter import segment_text
from core.normalize import embedding_text_for


# A "raw work": metadata plus a list of (section_label, raw_latin_text) parts.
RawWork = Tuple[dict, List[Tuple[str, str]]]

PAGE_MARKER = re.compile(r"\bp\. ?(\d+[A-Da-d]?) ?\|")


def split_page_markers(sentences: List[str]) -> List[Tuple[str, Optional[str]]]:
    """Remove inline page markers, returning (clean_text, page) per sentence.

    ``page`` is the page the sentence starts on, or None before the first marker.
    A sentence holding nothing but markers is dropped, but still moves the page on.
    """
    out: List[Tuple[str, Optional[str]]] = []
    current: Optional[str] = None
    for sentence in sentences:
        start = PAGE_MARKER.match(sentence)
        page = f"p. {start.group(1)}" if start else current
        for match in PAGE_MARKER.finditer(sentence):
            current = f"p. {match.group(1)}"
        clean = " ".join(PAGE_MARKER.sub(" ", sentence).split())
        if clean:
            out.append((clean, page))
    return out


class Connector(ABC):
    """Base class for source connectors."""

    # Short name used by the registry / CLI. Override in subclasses.
    name: str = "base"

    @abstractmethod
    def fetch(self, identifier: str) -> RawWork:
        """Return (metadata_dict, [(section_label, raw_text), ...]) for a work."""
        ...

    def discover(self, query: str, limit: int = 50) -> List[str]:
        """Return a list of identifiers that fetch() accepts (bulk ingestion).

        Optional: connectors that can enumerate a source (a category, an index
        page, a directory) override this. Others raise NotImplementedError.
        """
        raise NotImplementedError(f"{self.name} does not support discover()")

    @staticmethod
    def build_document(meta: dict, parts: List[Tuple[str, str]],
                       use_cltk: bool = False) -> Document:
        """Segment raw parts into a fully-structured Document (not yet persisted).

        If meta carries ``_pre_segmented=True``, each part is treated as a single
        already-complete unit (one segment, no sentence splitting) — used for
        inscriptions and other short, self-contained texts.

        If meta carries ``_page_markers=True``, inline "p. 0719B |" markers are
        removed from the text and the page goes in each segment's source_loc.
        """
        meta = dict(meta)
        pre_segmented = meta.pop("_pre_segmented", False)
        verse = meta.pop("_verse", False)   # one segment per verse line (poetry)
        page_markers = meta.pop("_page_markers", False)
        lang = meta.get("language", "la")
        sections: List[Section] = []
        for s_order, (label, raw_text) in enumerate(parts):
            if pre_segmented:
                sentences = [raw_text.strip()] if raw_text.strip() else []
            else:
                sentences = segment_text(raw_text, use_cltk=use_cltk,
                                         lang=lang, verse=verse)
            located = (split_page_markers(sentences) if page_markers
                       else [(s, None) for s in sentences])
            section = Section(label=label, order=s_order)
            section.segments = [
                Segment(latin_text=s, order=i, source_loc=page or label,
                        embed_text=embedding_text_for(s))
                for i, (s, page) in enumerate(located)
            ]
            sections.append(section)
        return Document(
            title=meta.get("title", "Untitled"),
            author=meta.get("author"),
            century=meta.get("century"),
            genre=meta.get("genre"),
            language=meta.get("language", "la"),
            language_stage=meta.get("language_stage", "unknown"),
            source=meta.get("source"),
            shelfmark=meta.get("shelfmark"),
            license=meta.get("license"),
            has_existing_translation=meta.get("has_existing_translation", False),
            translation_status=meta.get("translation_status", "unknown"),
            translation_evidence=meta.get("translation_evidence"),
            sections=sections,
        )
