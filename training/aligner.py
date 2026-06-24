"""Cross-lingual sentence aligner (LaBSE + monotonic DP, Bertalign-style).

This is the *proper* tool for mining parallel sentences from comparable texts that
are NOT pre-aligned — e.g. a Patrologia Latina work and its CCEL English
translation, or two Perseus editions that don't share a citation scheme. It also
*refines* the coarse chapter-level pairs from `training/parallel.py` into clean
sentence-level training pairs.

How it differs from `training/align.py` (greedy windowed argmax):
  * A strong cross-lingual encoder (LaBSE by default) instead of the lighter
    search embedder. LaBSE is trained for bitext mining, so cosine similarity is
    meaningful across Latin/Greek <-> English.
  * A global, monotonic dynamic program instead of per-sentence argmax. This finds
    the best *whole-text* alignment and supports many-to-one beads (1-2, 2-1, 2-2)
    plus insertions/deletions (1-0, 0-1) — essential when a translator splits or
    merges sentences, which is the norm for paraphrastic translations.

Algorithm (two passes, à la Bertalign / Vecalign):
  1. Anchor: a 1-1 Needleman-Wunsch over the cosine matrix gives an approximate
     backbone path. We only use it to center a band for pass 2 (skipped for very
     large inputs, where we fall back to the proportional diagonal).
  2. Refine: a banded DP over alignment "beads" (di, dj) in a small transition set,
     scoring each merge by the cosine of the *pooled* segment embeddings, with a
     gap penalty and a mild merge penalty. Backtracking yields the bead sequence.

Honest limitation: alignment quality is bounded by the encoder and by how
literal the translation is. LaBSE's Latin/Greek coverage is imperfect and
patristic translations are often paraphrastic, so this raises recall/precision but
does not guarantee clean pairs — keep `threshold` conservative and spot-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

# Reuse the pair type and English splitter from the lightweight aligner so both
# code paths emit the same training-pair shape.
from training.align import AlignedPair, split_english


LABSE_MODEL = "sentence-transformers/LaBSE"  # 109-lang bitext encoder; ~1.8GB download

# Allowed alignment beads (n_src, n_tgt). (1,0)/(0,1) are insertions/deletions.
# Larger merges than these are rarely clean training data, so we stop at sums of 4.
TRANSITIONS = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (1, 0), (0, 1)]

_NEG = -1.0e9  # DP sentinel for "unreachable"


@dataclass
class Bead:
    """One alignment unit: the source/target sentence indices it spans."""
    src: List[int]
    tgt: List[int]
    score: float  # cosine of the pooled segments (0.0 for pure insertions/deletions)


def make_labse_embedder():
    """Build a LaBSE-backed embedder (lazy: no model download until first .embed).

    Returns the project `Embedder` pointed at LaBSE, so it satisfies the same
    duck-typed `.embed(list[str]) -> np.ndarray` interface the aligner needs.
    """
    from core.embedder import Embedder
    return Embedder(LABSE_MODEL)


# -- core alignment ----------------------------------------------------------

def align(
    src_sents: Sequence[str],
    tgt_sents: Sequence[str],
    embedder,
    *,
    band: Optional[int] = None,
    gap_penalty: float = -0.2,
    merge_penalty: float = 0.1,
    anchor_cap: int = 250_000,
) -> List[Bead]:
    """Align two sentence lists into a monotonic sequence of beads.

    `embedder` is any object with `.embed(list[str]) -> [n, d] array` (e.g. the
    LaBSE embedder from `make_labse_embedder()`). `band` bounds how far the DP may
    stray from the anchor diagonal; None auto-sizes it and it auto-grows if the
    end state is unreachable.
    """
    n, m = len(src_sents), len(tgt_sents)
    if n == 0 or m == 0:
        return []

    S = _unit(np.asarray(embedder.embed(list(src_sents)), dtype=np.float32))
    T = _unit(np.asarray(embedder.embed(list(tgt_sents)), dtype=np.float32))
    # Prefix sums let us pool any contiguous span in O(1) for merged-bead scoring.
    ps_s = _prefix_sums(S)
    ps_t = _prefix_sums(T)

    centers = _anchor_centers(S, T, anchor_cap)
    if band is None:
        band = max(10, int(0.10 * max(n, m)))

    while True:
        dp, bp = _fill_dp(n, m, centers, band, ps_s, ps_t, gap_penalty, merge_penalty)
        if dp[n, m] > _NEG / 2:
            break
        if band >= max(n, m):  # already effectively full DP; give up gracefully
            return []
        band = min(max(n, m), band * 2)

    return _backtrack(dp, bp, ps_s, ps_t, n, m)


def _fill_dp(n, m, centers, band, ps_s, ps_t, gap_penalty, merge_penalty):
    dp = np.full((n + 1, m + 1), _NEG, dtype=np.float32)
    bp = np.full((n + 1, m + 1, 2), -1, dtype=np.int32)
    for i in range(n + 1):
        c = int(centers[i])
        j_lo, j_hi = max(0, c - band), min(m, c + band)
        for j in range(j_lo, j_hi + 1):
            if i == 0 and j == 0:
                dp[0, 0] = 0.0
                continue
            best, bi, bj = _NEG, -1, -1
            for di, dj in TRANSITIONS:
                pi, pj = i - di, j - dj
                if pi < 0 or pj < 0:
                    continue
                prev = dp[pi, pj]
                if prev <= _NEG / 2:
                    continue
                if di == 0 or dj == 0:
                    cand = prev + gap_penalty
                else:
                    sim = float(_seg(ps_s, pi, i) @ _seg(ps_t, pj, j))
                    cand = prev + sim - merge_penalty * (di + dj - 2)
                if cand > best:
                    best, bi, bj = cand, pi, pj
            dp[i, j] = best
            bp[i, j, 0], bp[i, j, 1] = bi, bj
    return dp, bp


def _backtrack(dp, bp, ps_s, ps_t, n, m) -> List[Bead]:
    beads: List[Bead] = []
    i, j = n, m
    while i > 0 or j > 0:
        pi, pj = int(bp[i, j, 0]), int(bp[i, j, 1])
        if pi < 0:  # broken path (shouldn't happen once dp[n,m] is reachable)
            return []
        src_idx = list(range(pi, i))
        tgt_idx = list(range(pj, j))
        score = (float(_seg(ps_s, pi, i) @ _seg(ps_t, pj, j))
                 if src_idx and tgt_idx else 0.0)
        beads.append(Bead(src_idx, tgt_idx, round(score, 3)))
        i, j = pi, pj
    beads.reverse()
    return beads


# -- high-level helpers ------------------------------------------------------

def beads_to_pairs(
    beads: Sequence[Bead],
    src_sents: Sequence[str],
    tgt_sents: Sequence[str],
    *,
    threshold: float = 0.45,
    min_chars: int = 25,
    max_merge: int = 2,
) -> List[AlignedPair]:
    """Keep only confident, substantial beads as training pairs.

    Drops insertions/deletions, low-similarity beads, over-large merges, and tiny
    source fragments. Multi-sentence sides are joined with spaces. Favors precision
    over recall — clean fine-tuning data, not full coverage.
    """
    out: List[AlignedPair] = []
    for b in beads:
        if not b.src or not b.tgt:
            continue
        if len(b.src) > max_merge or len(b.tgt) > max_merge:
            continue
        if b.score < threshold:
            continue
        s = " ".join(src_sents[k] for k in b.src).strip()
        t = " ".join(tgt_sents[k] for k in b.tgt).strip()
        if len(s) < min_chars or not t:
            continue
        out.append(AlignedPair(s, t, b.score))
    return out


def align_texts(
    src_text: str,
    tgt_text: str,
    embedder,
    *,
    src_lang: str = "la",
    threshold: float = 0.45,
    min_chars: int = 25,
    max_merge: int = 2,
    **align_kwargs,
) -> List[AlignedPair]:
    """Segment two raw texts, align them, and return clean sentence pairs.

    Source is segmented with the language-aware splitter; the English target uses
    the same regex splitter as `training/align.py`.
    """
    from core.segmenter import segment_text
    src = segment_text(src_text, lang=src_lang)
    tgt = split_english(tgt_text)
    beads = align(src, tgt, embedder, **align_kwargs)
    return beads_to_pairs(
        beads, src, tgt, threshold=threshold, min_chars=min_chars, max_merge=max_merge
    )


def align_texts_via_mt(
    src_text: str,
    tgt_text: str,
    translator,
    embedder,
    *,
    src_lang: str = "la",
    threshold: float = 0.45,
    min_chars: int = 25,
    max_merge: int = 2,
    batch_size: int = 8,
    **align_kwargs,
) -> List[AlignedPair]:
    """Translate-then-align: render the source into rough English with `translator`,
    align that against the target English, but emit the ORIGINAL source in the pairs.

    Monolingual (English<->English) alignment is more robust than cross-lingual and
    needs no Latin/Greek-capable encoder — so a light embedder (the project's MiniLM)
    suffices. `translator` is any Translator (e.g. the routed `nllb-latin`); it is used
    purely as a matching heuristic. Because the translated sentences are 1-1 with the
    originals, bead indices map straight back, and kept pairs are original-src <->
    gold-tgt — never the MT output, so this does not train a model on its own guesses.
    """
    from core.segmenter import segment_text
    src = segment_text(src_text, lang=src_lang)
    tgt = split_english(tgt_text)
    if not src or not tgt:
        return []
    src_en = translator.translate_batch(src, batch_size=batch_size)
    beads = align(src_en, tgt, embedder, **align_kwargs)
    # Score/align on the English proxy, but emit the original source text.
    return beads_to_pairs(
        beads, src, tgt, threshold=threshold, min_chars=min_chars, max_merge=max_merge
    )


# -- numerics ----------------------------------------------------------------

def _unit(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _prefix_sums(mat: np.ndarray) -> np.ndarray:
    """[n, d] -> [n+1, d] prefix sums, so span [a:b] = ps[b] - ps[a]."""
    return np.vstack([np.zeros((1, mat.shape[1]), np.float32), np.cumsum(mat, axis=0)])


def _seg(ps: np.ndarray, a: int, b: int) -> np.ndarray:
    """Pooled (summed + renormalized) embedding of the contiguous span [a:b)."""
    v = ps[b] - ps[a]
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


def _anchor_centers(S: np.ndarray, T: np.ndarray, cap: int) -> np.ndarray:
    """Per-source-prefix band centers from a 1-1 Needleman-Wunsch backbone.

    For large inputs (n*m > cap) the NW pass is skipped and we use the proportional
    diagonal i*m/n, which the auto-growing band compensates for.
    """
    n, m = len(S), len(T)
    diag = np.array([round(i * m / n) for i in range(n + 1)], dtype=np.int32)
    if n * m > cap:
        return diag

    gap = -0.1
    dp = np.full((n + 1, m + 1), _NEG, dtype=np.float32)
    dp[0, 0] = 0.0
    # back: 0 = match (diag), 1 = skip src (up), 2 = skip tgt (left)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    dp[1:, 0] = gap * np.arange(1, n + 1)
    back[1:, 0] = 1
    dp[0, 1:] = gap * np.arange(1, m + 1)
    back[0, 1:] = 2
    for i in range(1, n + 1):
        sim_row = S[i - 1] @ T.T  # cosine of src i-1 against every target
        for j in range(1, m + 1):
            diag_score = dp[i - 1, j - 1] + float(sim_row[j - 1])
            up = dp[i - 1, j] + gap
            left = dp[i, j - 1] + gap
            best, b = diag_score, 0
            if up > best:
                best, b = up, 1
            if left > best:
                best, b = left, 2
            dp[i, j] = best
            back[i, j] = b

    centers = diag.copy()
    i, j = n, m
    while i > 0 or j > 0:
        centers[i] = j
        b = back[i, j]
        if b == 0:
            i, j = i - 1, j - 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
    centers[0] = 0
    return centers
