"""Cheap automatic checks on a machine translation, run as it is written.

The flags are *heuristics that point a reviewer at a line*, not a verdict on quality:
a clean line can still be wrong (a rare name read as another one is invisible to all of
these), and a flagged line is often fine. Each flag is a short token stored, comma-
separated, in segments.translation_flags ("" = checked and clean, NULL = never checked).

    empty         no English at all
    copied        the English is mostly the Latin back again (names excepted)
    too_short     far fewer English words than the Latin warrants
    too_long      far more -- the model padded or looped
    repeated      a phrase or word repeated, the loop NLLB falls into
    uncertain     the translator itself marked the line ([?] or [uncertain])
    new_name      a capitalised English word that nothing in the Latin line (or the
                  document glossary) accounts for -- the signature of an invented name
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_CAP = re.compile(r"(?<![.!?:;]\s)(?<!^)\b([A-Z][a-z]{3,})\b")

# Words that are capitalised in English prose and need no Latin source.
_ENGLISH_CAPS = {
    "Latin", "Roman", "Romans", "Greek", "Greeks", "Italian", "Italians", "Italy",
    "Latins", "Muse", "Muses", "God", "Gods", "Lord", "Heaven", "Hell", "Fate", "Fates",
    "Jove", "Jupiter", "Mars", "Apollo", "Venus", "Juno", "Minerva", "Neptune", "Pluto",
    "Father", "Rome", "Fortune", "Fame", "Rumour", "Discord", "Virtue", "Faith", "Peace", "War",
}

# Words whose gloss differs enough from the Latin spelling to need a hint.
SUSPICIOUS_RATIO = (0.45, 3.2)     # English words per Latin word


def _words(text: str) -> List[str]:
    return _WORD.findall(text or "")


def _stem(word: str) -> str:
    w = word.lower().replace("j", "i").replace("v", "u")
    return w[:3]


def load_glossary_names(text: str) -> Set[str]:
    """Capitalised words in a glossary file: names the English may use freely."""
    return {w for w in _WORD.findall(text or "") if w[:1].isupper() and len(w) > 3}


def flag_segment(latin: str, english: str, allowed_names: Optional[Iterable[str]] = None) -> List[str]:
    """The flags for one translated line (empty list = nothing suspicious)."""
    en = (english or "").strip()
    if not en:
        return ["empty"]
    flags: List[str] = []
    lat_words = _words(latin)
    en_words = _words(en)
    if "[?]" in en or "[uncertain]" in en:
        flags.append("uncertain")

    if len(lat_words) >= 4:
        src = {w.lower() for w in lat_words if len(w) >= 3}
        out = [w.lower() for w in en_words if len(w) >= 3]
        if out and sum(w in src for w in out) / len(out) >= 0.6:
            flags.append("copied")
        ratio = len(en_words) / len(lat_words)
        if ratio < SUSPICIOUS_RATIO[0]:
            flags.append("too_short")
        elif ratio > SUSPICIOUS_RATIO[1]:
            flags.append("too_long")

    low = [w.lower() for w in en_words]
    if any(low[i] == low[i + 1] == low[i + 2] for i in range(len(low) - 2)) or any(
            low[i:i + 3] == low[j:j + 3] and len(set(low[i:i + 3])) > 1
            for i in range(len(low) - 5) for j in range(i + 3, len(low) - 2)
            if low[i:i + 3] == low[j:j + 3]):
        flags.append("repeated")

    stems = {_stem(w) for w in lat_words}
    ok = set(_ENGLISH_CAPS) | set(allowed_names or ())
    for name in _CAP.findall(en):
        if name in ok or _stem(name) in stems:
            continue
        flags.append("new_name")
        break
    return flags


def flag_batch(latins: List[str], englishes: List[str],
               allowed_names: Optional[Iterable[str]] = None) -> List[str]:
    """flag_segment over a batch, each result joined with commas ("" = clean)."""
    return [",".join(flag_segment(l, e, allowed_names)) for l, e in zip(latins, englishes)]


def summarize(flag_strings: Iterable[str]) -> Dict[str, int]:
    """Counts per flag, plus 'checked' and 'flagged' totals, for a progress line."""
    counts: Dict[str, int] = {"checked": 0, "flagged": 0}
    for s in flag_strings:
        counts["checked"] += 1
        if s:
            counts["flagged"] += 1
            for f in s.split(","):
                counts[f] = counts.get(f, 0) + 1
    return counts
