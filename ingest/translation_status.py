"""Shared inference of whether a work already has a published English translation.

The reading/discovery mission is to surface *truly untranslated* Latin and Greek.
Knowing a work lacks an English translation is harder than finding the text, so
this module centralises the heuristics every connector can reuse instead of each
re-implementing its own ``has_existing_translation`` guess.

Three signals, strongest first:

1. **Parallel edition** — a source ships an English (`-eng`) edition next to the
   original (Perseus / First1KGreek). Presence is near-certain proof a
   translation exists; absence is only weak evidence it does not.
2. **Documentary-genre proxy** — a few genres are *overwhelmingly* untranslated
   into English (charters/diplomata, formulae, necrologia, inscriptions). For
   those we assume untranslated with high precision.
3. **Default unknown** — no signal. We do NOT guess; ``unknown`` is honest and
   keeps false "untranslated" claims out of a discovery library.

The result carries a 3-state ``status`` so the library can distinguish "known
translated" from "high-confidence untranslated" from "no idea" — a distinction
the bare ``has_existing_translation`` boolean cannot make.

Deliberately conservative on hagiography: *Vitae* / *Passiones* are mostly
untranslated in bulk, but enough famous ones are Englished (Vita Severini, Vita
Karoli, ...) that a blanket tag would mislabel them. Hagiography therefore stays
``unknown`` rather than ``untranslated``; promote specific series via a curated
allow-list when one is verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import re


# Status vocabulary kept in sync with the documents.translation_status column.
TRANSLATED = "translated"     # a published English translation is known to exist
UNTRANSLATED = "untranslated"  # high-confidence: none exists (documentary genres)
UNKNOWN = "unknown"           # no signal — do not claim either way

STATUSES = (TRANSLATED, UNTRANSLATED, UNKNOWN)


@dataclass
class TranslationStatus:
    """Outcome of inference. ``has_existing_translation`` mirrors the legacy
    boolean (True iff status == TRANSLATED) so existing call-sites keep working."""

    status: str
    reason: str

    @property
    def has_existing_translation(self) -> bool:
        return self.status == TRANSLATED


# Substrings (matched case-insensitively against title/genre/series) for genres
# that are documentary and almost never translated into English. High precision
# by design — only add a marker here if mislabelling its works "untranslated"
# would be rare. See the module docstring on why hagiography is excluded.
_UNTRANSLATED_MARKERS = (
    "diplom",      # Diplomata — royal/imperial charters
    "urkunden",    # German for charters (MGH Diplomata volume titles)
    "formul",      # Formulae — legal formularies
    "necrolog",    # Necrologia / libri memoriales
    "inscript",    # epigraphy
    "epigraph",
    "papyr",       # papyri
)


def has_parallel_english(filenames: Iterable[str]) -> bool:
    """True if a sibling English edition (``*-engN.xml``) sits beside the text.

    This is the Perseus/First1KGreek convention: ``...-eng1.xml`` next to
    ``...-lat2.xml`` / ``...-grc1.xml`` in the same work directory.
    """
    return any(re.search(r"-eng\d*\.xml$", f) for f in filenames)


def infer_translation_status(
    *,
    title: Optional[str] = None,
    genre: Optional[str] = None,
    series: Optional[str] = None,
    edition_files: Optional[Iterable[str]] = None,
    default: str = UNKNOWN,
) -> TranslationStatus:
    """Infer translation status from whatever signals a connector can supply.

    Pass ``edition_files`` (a directory listing) when the source ships parallel
    editions; pass ``title``/``genre``/``series`` for the documentary-genre
    proxy. ``default`` is returned when nothing matches (usually ``unknown``,
    but a connector for a genuinely-untranslated corpus may pass ``untranslated``).
    """
    if edition_files is not None and has_parallel_english(edition_files):
        return TranslationStatus(TRANSLATED, "parallel English (-eng) edition present")

    haystack = " ".join(p for p in (title, genre, series) if p).lower()
    for marker in _UNTRANSLATED_MARKERS:
        if marker in haystack:
            return TranslationStatus(
                UNTRANSLATED, f"documentary genre proxy (matched {marker!r})"
            )

    if default == UNKNOWN:
        return TranslationStatus(UNKNOWN, "no translation signal")
    return TranslationStatus(default, f"connector default ({default})")
