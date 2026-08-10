"""Detect segments whose latin_text is too corrupted to translate reliably.

Built for doc 376 (Salmasius), which has embedded Greek quotations that the
original OCR never recognized as Greek at all -- it force-fit the polytonic
glyphs into Latin-alphabet lookalikes (see ingest/ocr_fix.py's module
docstring for the long-s story; this is a *different*, unrelated corruption).
Feeding that noise to a Latin NMT model doesn't fail loudly -- it produces
fluent, confident-sounding, entirely fabricated English, which is worse than
an honest "untranslatable" placeholder because it reads as real content.

A plain "fraction of words not in the reference vocabulary" signal is too
noisy to use alone (~27% baseline even for clean segments here, given rare
real words, proper nouns, and residual uncorrected OCR damage). An earlier
version of this also penalized a high fraction of very short (<=2 char)
"word" fragments, on the theory that letter-by-letter Greek misreadings
break words up more than normal Latin -- but Latin's own function words
(et, in, ad, ut, is...) are short and extremely common, so that signal
mostly just penalized ordinary short sentences and was dropped after
calibration showed it swamped everything else with false positives.

What's left combines signals specific to the transliteration-noise failure
mode rather than merely-imperfect Latin:

  * telltale punctuation that basically never appears in real Latin prose
    (guillemets, backslash, caret, percent -- OCR noise byte-shapes)
  * a letter/digit stuck together inside a token (isolated garbling, not the
    "1" for "I"/ligature artifacts we already special-cased in ocr_fix.py)
  * a run of 6+ consecutive consonants (implausible in real Latin phonotactics
    but common when Greek diacritics/breathing marks get OCR'd as consonant
    look-alikes)

Calibrated against this corpus by comparing scores across a large random
sample of doc 376 segments at increasing score bands and manually checking
precision; threshold=10 was chosen because segments scoring that high were,
on inspection, essentially all genuine noise (garbled Greek quotations or
index/page-number fragments) rather than ordinary imperfect Latin. Recall is
necessarily incomplete -- this catches the confident cases, not everything.
"""
from __future__ import annotations

import re

UNTRANSLATABLE_PLACEHOLDER = "[untranslatable: source text corrupted or non-Latin]"

_TELLTALE = re.compile(r"[«»%\\^]")  # NOT "*" -- that's a legitimate footnote/reference marker in this edition
_DIGIT_GLUED = re.compile(r"[A-Za-z]\d|\d[A-Za-z]")
_CONSONANT_RUN = re.compile(r"[bcdfghjklmnpqrstvwxz]{6,}", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z]+")


def garble_score(text: str) -> float:
    """Higher = more likely to be untranslatable transliteration noise.
    See module docstring for calibration; threshold=10 (the is_garbled/
    translate_pending.py default) was chosen for high precision, not recall."""
    if not text:
        return 0.0
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0

    telltale = len(_TELLTALE.findall(text))
    digit_glued = len(_DIGIT_GLUED.findall(text))
    consonant_runs = len(_CONSONANT_RUN.findall(text))

    raw = telltale * 3 + digit_glued * 2 + consonant_runs * 2
    return raw / max(len(words), 1) * 10


def is_garbled(text: str, threshold: float = 10.0) -> bool:
    return garble_score(text) >= threshold
