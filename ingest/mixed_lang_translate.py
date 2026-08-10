"""Translate segments whose text mixes Latin and Greek script (the product of
the Greek-recovery pipeline in ingest/page_splice.py: a segment that used to
be pure garbled-Greek-as-Latin-noise now has real embedded Greek alongside
its surrounding Latin).

Tried translating each script-homogeneous run separately (Greek runs to the
Greek NLLB checkpoint, Latin runs to the Latin one) and it was a disaster:
even with the *good* re-OCR, embedded Greek quotations still come out
fragmented into many 1-4 word runs (residual OCR noise breaks up what should
be one continuous quotation), and translating an isolated word or two as its
own "sentence" sent this fine-tuned NLLB model into severe repetition-loop
degeneration -- one segment's English output repeated "the son of Aeschines"
seventeen times. That's not a minor quality hit, it's actively worse than
the honest untranslatable placeholder it replaced.

So: translate only the Latin portions (concatenated back into one coherent
string per segment -- full sentence context, one model call, no fragment
degeneration), and preserve the recovered Greek verbatim, untranslated, as a
bracketed note. Proper nouns and technical terms (which most of these short
embedded quotations turn out to be -- personal names, single Greek
technical/legal vocabulary) generally shouldn't be machine-"translated" word
by word anyway; showing the real Greek is more useful than a fabricated
gloss, and a human reader who knows Greek can interpret it directly.
"""
from __future__ import annotations

import re
from typing import List, Tuple

_GREEK_CHAR = re.compile(r"[Ͱ-Ͽἀ-῿]")
_RUN_RE = re.compile(r"[Ͱ-Ͽἀ-῿]+|[^Ͱ-Ͽἀ-῿]+")


def split_runs(text: str) -> List[Tuple[str, str]]:
    """Returns [(script, run_text), ...] in order, script in {"grc","la"}."""
    runs = []
    for m in _RUN_RE.finditer(text):
        chunk = m.group()
        script = "grc" if _GREEK_CHAR.search(chunk) else "la"
        runs.append((script, chunk))
    return runs


def split_latin_and_greek(text: str) -> Tuple[str, List[str]]:
    """Returns (latin_only_text, [greek_fragment, ...]) -- Greek runs removed
    from the Latin text and collected separately, in order of appearance."""
    latin_parts = []
    greek_parts = []
    for script, chunk in split_runs(text):
        if script == "grc":
            stripped = chunk.strip()
            if stripped:
                greek_parts.append(stripped)
        else:
            latin_parts.append(chunk)
    return "".join(latin_parts), greek_parts


def translate_mixed_batch(texts: List[str], lat_translator, batch_size: int = 16) -> List[str]:
    """One Latin-translator call for the whole batch (proper sentence
    context, no fragment degeneration); recovered Greek is preserved
    verbatim as a bracketed note rather than machine-"translated" word by
    word. See module docstring for why -- the run-by-run approach this
    replaced caused severe repetition-loop degeneration."""
    latin_and_greek = [split_latin_and_greek(t) for t in texts]
    latin_only = [lo for lo, _ in latin_and_greek]
    translated_latin = (lat_translator.translate_batch(latin_only, batch_size=batch_size)
                         if any(s.strip() for s in latin_only) else [""] * len(latin_only))

    results = []
    for (_, greek_parts), eng in zip(latin_and_greek, translated_latin):
        eng = (eng or "").strip()
        if greek_parts:
            note = "; ".join(dict.fromkeys(greek_parts))  # de-dupe, keep order
            eng = f"{eng} [Greek in source: {note}]".strip()
        results.append(eng)
    return results
