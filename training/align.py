"""Embedding-based sentence alignment for parallel-corpus mining.

Given the full source (Latin) and target (English) texts of the same work, pair
each source sentence with its best-matching target sentence using the multilingual
embedder. Because the editions don't share a citation scheme, we don't rely on
chapter structure — only on the two texts being roughly parallel and monotonic.

Searches a window around the proportional position (sentence i of N_src maps near
i*N_tgt/N_src) and keeps only high-similarity, roughly-monotonic 1-1 matches. This
favors precision over recall: we want clean training pairs, not full coverage.

This is the cheap, dependency-light aligner. For comparable texts that aren't
pre-aligned, or to handle sentence splits/merges (1-2, 2-1), prefer the LaBSE +
monotonic-DP aligner in `training/aligner.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import re

import numpy as np


@dataclass
class AlignedPair:
    src: str
    tgt: str
    score: float


_EN_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_english(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _EN_SENT.split(text) if len(s.strip()) > 1]


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def align_sentences(src_sents: List[str], tgt_sents: List[str], embedder,
                    window: int = 20, threshold: float = 0.45,
                    min_chars: int = 25) -> List[AlignedPair]:
    if not src_sents or not tgt_sents:
        return []
    src_emb = _normalize(embedder.embed(src_sents))
    tgt_emb = _normalize(embedder.embed(tgt_sents))

    n_src, n_tgt = len(src_sents), len(tgt_sents)
    pairs: List[AlignedPair] = []
    last_j = -1
    for i in range(n_src):
        j0 = round(i * n_tgt / n_src)
        lo, hi = max(0, j0 - window), min(n_tgt, j0 + window + 1)
        sims = tgt_emb[lo:hi] @ src_emb[i]
        k = int(np.argmax(sims))
        j, best = lo + k, float(sims[k])
        # Keep only confident, roughly-monotonic, substantial matches.
        if best >= threshold and j >= last_j - 2 and len(src_sents[i]) >= min_chars:
            pairs.append(AlignedPair(src_sents[i], tgt_sents[j], round(best, 3)))
            last_j = max(last_j, j)
    return pairs
