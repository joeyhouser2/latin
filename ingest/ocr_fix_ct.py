"""Correct this document's second OCR error class: c immediately before t
misread as f ("contraftus" for "contractus", "diftum" for "dictum", "Leftor"
for "Lector").

Doesn't need its own trained model. Genuine Latin essentially never has a
native f-before-t cluster (checked against the reference vocab: the only
"ft" hits are German loanwords that leaked in from other documents' titles,
or those other documents' own OCR noise -- not real Latin). The existing
long-s classifier (training/long_s_classifier.py) already scores every f for
P(was long-s); when it's confidently *low* for an f-before-t position, that's
strong evidence the true letter is something else entirely, and c is by far
the likeliest candidate for this document. So: reuse that model's confident
rejections as the trigger, then require the same whole-word dictionary
corroboration (exact or stem match) the other tiers use before applying.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional

from ingest.ocr_fix_model import score_positions
from ingest.ocr_fix import _fold, _prefix_family_hit

_WORD_RE = re.compile(r"[A-Za-z]+")


def ct_correct(word: str, vocab, vocab_prefixes, pipeline, not_s_threshold: float = 0.15):
    """Try flipping f-before-t position(s) to c. Returns corrected word or None."""
    candidates = [i for i in range(len(word) - 1) if word[i] in "fF" and word[i + 1] in "tT"]
    if not candidates or len(candidates) > 4:
        return None
    probs = score_positions(word, pipeline)
    eligible = [i for i in candidates if probs.get(i, 1.0) <= not_s_threshold]
    if not eligible:
        return None

    def flip(combo):
        chars = list(word)
        for i in combo:
            chars[i] = "c" if word[i] == "f" else "C"
        return "".join(chars)

    for k in range(len(eligible), 0, -1):
        for combo in combinations(eligible, k):
            cand = flip(combo)
            if vocab.get(_fold(cand), 0) > 0:
                return cand

    cand = flip(eligible)
    if vocab_prefixes is not None and _prefix_family_hit(cand, vocab_prefixes):
        return cand
    return None


@dataclass
class CtCorrection:
    original: str
    corrected: str


@dataclass
class CtPassResult:
    text: str
    changes: List[CtCorrection] = field(default_factory=list)


def apply_ct_tier(text: str, vocab, vocab_prefixes, pipeline) -> CtPassResult:
    """Single-token only (no merges -- the ct-pattern hasn't shown up split
    across letter-spaced fragments the way the long-s title-page runs did)."""
    tokens = re.findall(r"\s+|[A-Za-z]+|[^A-Za-z\s]+", text)
    out: List[str] = []
    changes: List[CtCorrection] = []
    for tok in tokens:
        if not _WORD_RE.fullmatch(tok) or "f" not in tok.lower():
            out.append(tok)
            continue
        corrected = ct_correct(tok, vocab, vocab_prefixes, pipeline)
        if corrected:
            out.append(corrected)
            changes.append(CtCorrection(tok, corrected))
        else:
            out.append(tok)
    return CtPassResult("".join(out), changes)
