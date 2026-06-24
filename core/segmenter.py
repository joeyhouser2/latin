"""Split raw Latin text into sentences (segments).

The segment is our alignment unit, so good sentence boundaries matter. We default
to a lightweight, dependency-free splitter that works offline. CLTK's trained
Latin tokenizer is available as an optional upgrade via use_cltk=True, but it
requires downloading a model the first time, so it is not the default.
"""

from __future__ import annotations

import re
from typing import List


# Common Latin/Roman abbreviations whose trailing period is NOT a sentence end.
_ABBREVIATIONS = {
    "a", "c", "cn", "d", "f", "k", "l", "m", "n", "p", "q", "s", "t", "v",  # praenomina
    "kal", "non", "id", "coss", "cos", "imp", "trib", "pl", "sext",
    "vol", "lib", "cap", "fol", "fig", "no", "cf", "ed", "etc", "i.e", "e.g",
}

_SENTENCE_END = re.compile(r"([.;:?!])\s+")
# Greek: full stop and the Greek question mark ";"/"·" end sentences; "·" is the
# ano teleia. Latin praenomina abbreviations don't apply.
_SENTENCE_END_GRC = re.compile(r"([.;?·])\s+")


def _looks_like_abbrev(chunk: str) -> bool:
    """True if the token just before the break is a known abbreviation."""
    word = re.split(r"[\s(]", chunk.strip())[-1].rstrip(".;:?!").lower()
    return word in _ABBREVIATIONS


def segment_text(
    text: str, use_cltk: bool = False, lang: str = "la", verse: bool = False
) -> List[str]:
    """Return a list of segment strings.

    Prose (default) is split into sentences. Verse is split one line per segment
    (``verse=True``): the line is the alignment unit for poetry, so the
    side-by-side reader renders verse line-for-line and scansion has something to
    scan. Without this, the prose path collapses newlines and re-chops the poem on
    sentence punctuation, destroying the lineation.
    """
    if verse:
        return _segment_verse(text)

    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    if use_cltk and lang == "la":
        try:
            return _segment_with_cltk(text)
        except Exception as exc:  # pragma: no cover - optional path
            print(f"CLTK segmentation unavailable ({exc}); using regex splitter.")

    return _segment_regex(text, lang=lang)


def _segment_regex(text: str, lang: str = "la") -> List[str]:
    is_greek = lang == "grc"
    pattern = _SENTENCE_END_GRC if is_greek else _SENTENCE_END
    sentences: List[str] = []
    buffer = ""
    last = 0
    for match in pattern.finditer(text):
        candidate = text[last:match.end()]
        buffer += candidate
        # Don't break after a Latin abbreviation like "Cn." or "Kal."
        if not is_greek and _looks_like_abbrev(text[last:match.start() + 1]):
            last = match.end()
            continue
        sentences.append(buffer.strip())
        buffer = ""
        last = match.end()
    tail = (buffer + text[last:]).strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def _segment_verse(text: str) -> List[str]:
    """One verse line per segment. Drops blank (stanza-break) lines and trims, but
    preserves the line itself; internal runs of spaces/tabs are collapsed so
    indentation and alignment whitespace don't leak into the text."""
    lines = []
    for raw in (text or "").splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if line:
            lines.append(line)
    return lines


def _segment_with_cltk(text: str) -> List[str]:  # pragma: no cover - optional path
    from cltk.sentence.lat import LatinPunktSentenceTokenizer
    tokenizer = LatinPunktSentenceTokenizer()
    return [s.strip() for s in tokenizer.tokenize(text) if s.strip()]
