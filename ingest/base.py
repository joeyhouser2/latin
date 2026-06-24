"""Connector interface.

A connector knows how to fetch and structure one kind of source. It produces an
unsegmented Document (sections with raw text), or a list of (section_label,
raw_text) pairs that the Library segments and embeds on ingest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from core.models import Document, Section, Segment
from core.segmenter import segment_text
from core.normalize import embedding_text_for


# A "raw work": metadata plus a list of (section_label, raw_latin_text) parts.
RawWork = Tuple[dict, List[Tuple[str, str]]]


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
        """
        meta = dict(meta)
        pre_segmented = meta.pop("_pre_segmented", False)
        lang = meta.get("language", "la")
        sections: List[Section] = []
        for s_order, (label, raw_text) in enumerate(parts):
            if pre_segmented:
                sentences = [raw_text.strip()] if raw_text.strip() else []
            else:
                sentences = segment_text(raw_text, use_cltk=use_cltk, lang=lang)
            section = Section(label=label, order=s_order)
            section.segments = [
                Segment(latin_text=s, order=i, source_loc=label,
                        embed_text=embedding_text_for(s))
                for i, s in enumerate(sentences)
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
            sections=sections,
        )
