"""Model-based tier for long-s correction.

Complements ingest.ocr_fix's dictionary tiers with a trained character-context
classifier (training/long_s_classifier.py) that scores each 'f' in a word for
P(this position was originally a long-s). Two things the dictionary tiers
can't do:

  * resolve rare-but-legitimate inflected forms that never appear verbatim
    elsewhere in the corpus (the whole reason the dictionary's stem-fallback
    tier existed, and was so error-prone -- the model handles this directly
    via local context instead of a coincidental whole-word/stem match).
  * decline correctly on this document's *other* OCR error (ct misread as
    ft), since the model was trained adversarially against exactly that
    pattern and doesn't confuse "f before t" with long-s.

Only applies a correction when *every* f-position in a word is unambiguous
(either confidently long-s or confidently genuine); any position that falls
in the uncertain middle band means the whole word is left untouched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

MODEL_PATH = "models/long_s_classifier.joblib"
WINDOW = 5

_WORD_RE = re.compile(r"[A-Za-z]+")
_model_cache: dict = {}


def load_model(path: str = MODEL_PATH):
    if path not in _model_cache:
        import joblib
        _model_cache[path] = joblib.load(path)["pipeline"]
    return _model_cache[path]


def _window(word: str, pos: int) -> str:
    padded = ("^" * WINDOW) + word + ("$" * WINDOW)
    p = pos + WINDOW
    left = padded[p - WINDOW: p]
    right = padded[p + 1: p + 1 + WINDOW]
    return f"{left}@{right}"


def score_positions(word: str, pipeline) -> dict:
    """position -> P(was long-s), for every f/F in word."""
    positions = [i for i, c in enumerate(word) if c in "fF"]
    if not positions:
        return {}
    ctxs = [_window(word.lower(), p) for p in positions]
    probs = pipeline.predict_proba(ctxs)[:, 1]
    return dict(zip(positions, probs))


def model_correct(word: str, pipeline, high: float = 0.9, low: float = 0.1,
                   vocab=None, vocab_prefixes=None):
    """Returns (corrected_word_or_None, min_confidence_or_None).

    None means "decline" -- either nothing needed fixing, some f-position
    landed in the ambiguous middle band, or a known-risky pattern applies (see
    guards below). We only ever flip a position, never leave it -- so the only
    way this tool causes real damage is by flipping a position that was
    already correct (a genuine f). Two such regressions showed up in
    real-document validation despite high/very-high model confidence:

      * word-initial f immediately followed by c ("fcenore" for "foenore",
        genuine word-initial f) scored 0.9998 -- essentially every clean
        corpus word starting literally "fc" comes from a long-s "sc-" word
        (scio, scelus, scientia...), so the model has learned that pattern is
        overwhelmingly long-s and can't tell this specific case apart, where
        the real defect is an unrelated o/c misread one letter later, not a
        long-s misread at all. No amount of raising the threshold fixes this
        since the wrong call is already made near-certainly.
      * multiple f's in one word: even a *specific* case that scored
        merely-high (~0.93) for a bad flip can score near-certain (~0.9995)
        for the *same* wrong flip in another word ("fuiftent" for "fuissent",
        genuine word-initial f) -- there's no fixed threshold that reliably
        separates these, since the model's confidence tracks the local
        n-gram context, not whether this particular instance is correct. For
        multi-f words we instead require independent corroboration: the
        fully-corrected candidate must pass ingest.ocr_fix's whole-word
        dictionary check (exact match or stem-family match). This trades
        recall for safety -- some genuinely-correct multi-f corrections
        (e.g. "femijfes" -> "semijses", real word "semisses" underneath, just
        not one this narrow dictionary check recognizes) get left unresolved
        rather than risk a regression. Pass vocab/vocab_prefixes to enable it;
        without them, multi-f words are never auto-corrected.
    """
    probs = score_positions(word, pipeline)
    if not probs:
        return None, None

    multi_f = len(probs) > 1

    chars = list(word)
    any_flip = False
    confidences = []
    for pos, p in probs.items():
        if pos == 0 and len(word) > 1 and word[1].lower() == "c":
            return None, None  # word-initial f-before-c: known regression pattern
        if p >= high:
            chars[pos] = "s" if word[pos] == "f" else "S"
            any_flip = True
            confidences.append(p)
        elif p <= low:
            confidences.append(1.0 - p)
        else:
            return None, None  # ambiguous -- decline the whole word
    if not any_flip:
        return None, None

    corrected = "".join(chars)
    if multi_f:
        if vocab is None or vocab_prefixes is None:
            return None, None
        from ingest.ocr_fix import _fold, _prefix_family_hit
        folded = _fold(corrected)
        if vocab.get(folded, 0) == 0 and not _prefix_family_hit(corrected, vocab_prefixes):
            return None, None  # no dictionary corroboration -- decline
    return corrected, min(confidences)


def merge_correct(joined: str, pipeline, moderate: float = 0.7, low: float = 0.1,
                   vocab=None, vocab_prefixes=None):
    """Like model_correct, but for a candidate reconstructed by joining
    letter-spaced fragments (e.g. "V"+"fur"+"arum" -> "Vfurarum").

    Reconstructing a word boundary is inherently higher-risk than fixing an
    f in a token that was already tokenized correctly, so this always
    requires dictionary corroboration (exact or stem match) regardless of
    how many f's are involved -- but in exchange accepts a lower model
    confidence bar (0.7 instead of 0.9), since the corroboration carries more
    of the weight. This is what resolves "usurarum" (the central term of the
    Salmasius treatise -- V+fur+arum): its lone f-position scores 0.876 here,
    just under the strict single-token bar, but "usur" is independently
    well-attested via usuram/usuras/usuris.
    """
    if vocab is None or vocab_prefixes is None:
        return None, None
    probs = score_positions(joined, pipeline)
    if not probs:
        return None, None
    chars = list(joined)
    any_flip = False
    confidences = []
    for pos, p in probs.items():
        if pos == 0 and len(joined) > 1 and joined[1].lower() == "c":
            return None, None
        if p >= moderate:
            chars[pos] = "s" if joined[pos] == "f" else "S"
            any_flip = True
            confidences.append(p)
        elif p <= low:
            confidences.append(1.0 - p)
        else:
            return None, None
    if not any_flip:
        return None, None
    corrected = "".join(chars)
    from ingest.ocr_fix import _fold, _prefix_family_hit
    if vocab.get(_fold(corrected), 0) == 0 and not _prefix_family_hit(corrected, vocab_prefixes):
        return None, None
    return corrected, min(confidences)


@dataclass
class ModelCorrection:
    original: str
    corrected: str
    confidence: float


@dataclass
class ModelPassResult:
    text: str
    changes: List[ModelCorrection] = field(default_factory=list)
    still_unresolved: List[str] = field(default_factory=list)


def apply_model_tier(text: str, pipeline, high: float = 0.9, low: float = 0.1,
                      vocab=None, vocab_prefixes=None) -> ModelPassResult:
    """Second pass over text that ingest.ocr_fix.correct_long_s has already
    run on -- only touches tokens that still contain an f/F (i.e. the
    dictionary tiers didn't resolve them).

    Also attempts merges across whitespace-only gaps (2-3 tokens), mirroring
    ingest.ocr_fix's dictionary-tier merge logic -- letter-spaced runs like
    "V fur arum" (the OCR having read inter-letter spacing in the original
    typesetting as word breaks) got a chance from the dictionary tier only if
    the corrected form was an *exact* vocab hit; "usurarum" only has
    stem-tier support, so it was correctly left alone there. This pass gives
    it a second chance via model confidence instead of a coincidental
    whole-word match, subject to the same multi-f dictionary-corroboration
    safety net as any other multi-f correction.

    vocab/vocab_prefixes (from ingest.ocr_fix.build_reference_vocab /
    build_prefix_index) are required for multi-f words to be corrected at all
    -- see model_correct's docstring."""
    tokens = re.findall(r"\s+|[A-Za-z]+|[^A-Za-z\s]+", text)
    out: List[str] = []
    changes: List[ModelCorrection] = []
    still_unresolved: List[str] = []

    from ingest.ocr_fix import _fold

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if not _WORD_RE.fullmatch(tok):
            out.append(tok)
            i += 1
            continue

        # Already a known word as-is -> leave untouched, and don't even
        # consider it as a possible merge-start (that's how "modo"/"quem"
        # ended up swallowing the next word: the old gate only required *some*
        # token in the window to contain an f, not the start of the window).
        if vocab is not None and vocab.get(_fold(tok), 0) > 0:
            out.append(tok)
            i += 1
            continue

        if "f" in tok or "F" in tok:
            corrected, conf = model_correct(tok, pipeline, high, low, vocab, vocab_prefixes)
            if corrected:
                out.append(corrected)
                changes.append(ModelCorrection(tok, corrected, conf))
                i += 1
                continue

        merged_ok = False
        for span in (2, 3):
            if i + 2 * (span - 1) >= n:
                continue
            window = tokens[i:i + 2 * span - 1]
            if any(not _WORD_RE.fullmatch(w) for w in window[0::2]):
                continue
            if any(not w.isspace() for w in window[1::2]):
                continue
            if not any("f" in w or "F" in w for w in window[0::2]):
                continue
            joined = "".join(window[0::2])
            if len(joined) < 5 or len(joined) > 20:
                continue
            # Always merge_correct here (never plain model_correct): a merge
            # candidate's word boundary is itself a hypothesis, so even a
            # single-f candidate needs dictionary corroboration -- plain
            # model_correct only skips that requirement for single-f *because*
            # it assumes the token was already correctly segmented, which
            # isn't true for a reconstructed merge. Using it here let bad
            # merges like "ejfe cum" -> "ejsecum" slip through on model
            # confidence alone.
            m_corrected, m_conf = merge_correct(joined, pipeline, 0.7, low, vocab, vocab_prefixes)
            if m_corrected:
                out.append(m_corrected)
                changes.append(ModelCorrection(" ".join(window[0::2]), m_corrected, m_conf))
                i += 2 * span - 1
                merged_ok = True
                break
        if merged_ok:
            continue

        if "f" in tok or "F" in tok:
            still_unresolved.append(tok)
        out.append(tok)
        i += 1

    return ModelPassResult("".join(out), changes, still_unresolved)
