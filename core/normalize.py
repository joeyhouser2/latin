"""Normalize Latin text for embedding (not for display).

Epigraphic and critical editions wrap letters in editorial sigla:
    Imp(erator)      expansion of an abbreviation
    [Aug]ustus       editorial restoration of lost letters
    <e>              editorial addition
    {sic}            deletion / error
    / //             line / page breaks
    --- [3] +        lacunae / illegible

For semantic search we want the *letters* but not the marks, so "Imp(erator)
Caes]ar [Aug]ustus" should embed as "Imperator Caesar Augustus". The original,
marked-up text is kept untouched for display; only the embedding copy is cleaned.
"""

from __future__ import annotations

import re
import unicodedata

_SIGLA = str.maketrans({c: None for c in "()[]<>{}"})   # drop brackets, keep content
_BREAKS = re.compile(r"/+")                              # line/page breaks -> space
_NOISE = re.compile(r"[+*=~^|]+")                        # gap/illegible markers
_DASHES = re.compile(r"-{2,}")                           # lacuna dashes
_DOTS = re.compile(r"\.{2,}")                            # ellipses / uncertain
_WS = re.compile(r"\s+")


def normalize_for_embedding(text: str) -> str:
    """Strip editorial markup, keeping the underlying letters. Safe for plain prose."""
    if not text:
        return ""
    t = text.translate(_SIGLA)
    t = _BREAKS.sub(" ", t)
    t = _DASHES.sub(" ", t)
    t = _DOTS.sub(" ", t)
    t = _NOISE.sub(" ", t)
    return _WS.sub(" ", t).strip()


def strip_greek_diacritics(text: str) -> str:
    """Reduce polytonic Greek to base letters (drop accents, breathings, iota
    subscript). NLLB's tokenizer was trained on monotonic modern Greek, so the
    polytonic marks of ancient Greek fragment it into poor tokens. Case is kept.

    e.g. "μῆνιν ἄειδε" -> "μηνιν αειδε".  Must be applied identically at training
    and inference for a model trained on normalized Greek.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def embedding_text_for(latin_text: str) -> "str | None":
    """Return the cleaned embedding text, or None if cleaning changes nothing.

    None means "just embed the display text" — saves storing a duplicate string
    for the common case of clean prose.
    """
    cleaned = normalize_for_embedding(latin_text)
    return cleaned if cleaned and cleaned != latin_text.strip() else None
