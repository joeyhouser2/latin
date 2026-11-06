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


_WORD = re.compile(r"[^\W\d_]+")


def is_worth_indexing(text: str) -> bool:
    """True if a segment should get a search vector: 3+ words and 5+ letters.

    Shorter segments are mostly verse numbers and apparatus sigla ("15.", "Cod.",
    "Vatican."). In a 16-query trial they took 16.6% of the top-20 slots, mostly
    with weak hits, while adding about 0.8 GB to the flat index. They stay in the
    store and in the reader; they just aren't reachable through vector search.
    """
    words = _WORD.findall(text or "")
    return len(words) >= 3 and sum(map(len, words)) >= 5


def _letters(text: str) -> int:
    return sum(map(len, _WORD.findall(text or "")))


def select_for_index(segments, text_of, doc_of) -> list:
    """Segments that should get a vector: every one passing is_worth_indexing,
    plus, for a document with none that pass, its segment with the most letters.

    The fallback keeps very short texts (epigrams, ostraca, one-line capitularies)
    reachable through vector search. A document with no letters at all gets
    nothing. Input order is kept.
    """
    chosen, best = [], {}
    has_long = set()
    for s in segments:
        text, doc = text_of(s), doc_of(s)
        if is_worth_indexing(text):
            chosen.append(s)
            has_long.add(doc)
        else:
            n = _letters(text)
            if n and n > best.get(doc, (0, None))[0]:
                best[doc] = (n, s)
    fallback = {id(s) for d, (_, s) in best.items() if d not in has_long}
    if not fallback:
        return chosen
    return [s for s in segments if id(s) in fallback or is_worth_indexing(text_of(s))]
