"""Detect and repair the "long s" OCR error in scanned early-modern Latin.

Pre-19th-century Latin type used a long s (ſ) for non-final s, distinct from the
round terminal s. OCR engines routinely misread ſ as f, since the two glyphs
differ mainly by a crossbar. Real f's (from Latin f) are read correctly — only
the ſ-as-f substitution is wrong — so correction is a per-token disambiguation:
does this token, with some of its f's flipped back to s, become a real word?

A second, rarer artifact shows up in some scans: a run of letters gets split
across stray spaces (e.g. "u fur arum" for "usurarum"), typically where the
source used letter-spaced type for emphasis. We handle this by attempting the
f->s fix on 2- and 3-token merges when the single-token fix fails.

Both repairs are validated against a reference vocabulary mined from the rest
of the corpus (clean, non-OCR digital editions), rather than an external
dictionary, so matches are tuned to the actual Latin/orthography already in
this library.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, List, Optional, Sequence

# u/v and i/j were used near-interchangeably in early modern printing
# ("Vsurarum" for "Usurarum"). Fold both to one form for *lookup only* --
# the output text keeps whatever letters it already had.
_FOLD = str.maketrans({"v": "u", "V": "u", "U": "u", "j": "i", "J": "i", "I": "i"})

_WORD_RE = re.compile(r"[A-Za-z]+")
_TOKEN_RE = re.compile(r"\s+|[A-Za-z]+|[^A-Za-z\s]+")

# Common function words whose f-corrupted form is unmistakable (no real Latin
# word collides with it), used to *score* a text's corruption level cheaply --
# independent of the vocabulary-driven correction pass below.
_SIGNATURE_PAIRS = [
    ("est", "eft"), ("esse", "effe"), ("sed", "fed"), ("sunt", "funt"),
    ("si ", "fi "), ("ipse", "ipfe"), ("ipsa", "ipfa"), ("quasi", "quafi"),
    ("nisi", "nifi"), ("magis", "magif"), ("his", "hif"), ("suis", "fuif"),
    ("causa", "caufa"), ("posse", "poffe"), ("esset", "effet"),
]


def _fold(word: str) -> str:
    return word.translate(_FOLD).lower()


def build_reference_vocab(store, exclude_doc_ids: Iterable[int] = ()) -> Counter:
    """Token-frequency counter over every document's latin_text except the
    given ids. Used as the "is this a real word" oracle for correction."""
    exclude = set(exclude_doc_ids)
    vocab: Counter = Counter()
    cur = store.conn.execute(
        """SELECT sec.doc_id, s.latin_text
           FROM segments s JOIN sections sec ON s.section_id = sec.id"""
    )
    for doc_id, text in cur:
        if doc_id in exclude or not text:
            continue
        for w in _WORD_RE.findall(text):
            if len(w) >= 2:
                vocab[_fold(w)] += 1
    return vocab


def long_s_score(text: str) -> tuple:
    """(corrupted_hits, clean_hits) counts of signature-pair occurrences.

    A document with corrupted > 0 and corrupted comparable to (or exceeding)
    clean is a strong long-s candidate; a clean digital edition should show
    ~0 corrupted hits against many clean hits.
    """
    low = text.lower()
    corrupted = clean = 0
    for good, bad in _SIGNATURE_PAIRS:
        clean += low.count(good)
        corrupted += low.count(bad)
    return corrupted, clean


def _f_positions(word: str) -> List[int]:
    return [i for i, c in enumerate(word) if c in "fF"]


def _apply_subst(word: str, positions: Sequence[int]) -> str:
    chars = list(word)
    for i in positions:
        chars[i] = "s" if chars[i] == "f" else "S"
    return "".join(chars)


# Common Latin nominal/verbal inflectional endings, used to peel a plausible
# suffix off a candidate before checking its stem against the corpus. Longest
# first so a candidate is matched against the most specific ending it fits.
# Deliberately excludes single-letter endings ("a", "e", "i", "o", "u") --
# those are real endings but far too permissive as a stem-boundary guess, and
# would let an unrelated-but-similar-length word "confirm" a bad candidate
# (see e.g. "fuifle", which has a genuine but unrelated f AND a separate l/s
# misread -- no f-only substitution of it is a real word, and it must stay
# unresolved rather than being coerced into a shape that happens to share a
# short stem with something real).
_ENDINGS = sorted({
    "arum", "orum", "erum", "ibus", "abus",
    "ae", "am", "as", "is", "os", "um", "us",
    "es", "em", "ei",
    "bam", "bas", "bat", "bamus", "batis", "bant",
    "bo", "bis", "bit", "bimus", "bitis", "bunt",
    "vi", "visti", "vit", "vimus", "vistis", "verunt", "verint",
    "avi", "avit", "avimus", "avistis", "averunt", "avisti",
    "issem", "isses", "isset", "issemus", "issetis", "issent", "isse",
    "ero", "eris", "erit", "erimus", "eritis", "erint",
    "amus", "atis", "ant",
    "emus", "etis", "ent",
    "mur", "mini", "ntur", "tur", "ris",
    "ndum", "ndus", "nda", "ndae", "ndorum", "ndarum", "ndis",
    "tus", "ta", "tum", "ti", "tae", "torum", "tarum", "tis",
    "ntis", "nti", "ntem", "ntes", "ntium", "ntibus", "ns",
    "are", "ere", "ire", "unt", "it", "et", "at",
}, key=len, reverse=True)


def _prefix_family_hit(cand: str, vocab_prefixes: dict) -> bool:
    """True if cand ends in a genuine Latin inflectional ending AND the stem
    left over is independently attested in the corpus (in some -- not
    necessarily the same -- inflected form).

    Catches domain-specific inflected forms too rare to appear verbatim
    elsewhere in the corpus (e.g. "usurarum", genitive plural of the very word
    this treatise is about: not itself in the vocab, but "usuram"/"usuras"/
    "usuris" are, so the "usur" stem is attested) while still requiring both
    a real morphological boundary AND an attested root -- so a candidate with
    an unrelated second typo (e.g. "fuifle", which has no real Latin ending at
    all once the f's are considered) can't coast in on a coincidence.
    """
    if len(cand) < 6:
        return False
    low = _fold(cand)
    for end in _ENDINGS:
        if low.endswith(end) and len(low) - len(end) >= 4:
            if low[: -len(end)] in vocab_prefixes:
                return True
    return False


def _best_fix(word: str, vocab: Counter, vocab_prefixes: Optional[dict] = None,
              min_count: int = 1) -> tuple:
    """Try replacing subsets of this word's f's with s.

    Returns (exact, stem): `exact` is a substitution that's an *attested* word
    elsewhere in the corpus (safe to auto-apply); `stem` is a weaker
    stem-morphology match (see `_prefix_family_hit`) offered only as a
    low-confidence suggestion for manual review, never auto-applied -- this
    document also has an unrelated OCR error (ct misread as ft) that produces
    other real Latin words when s-substituted (e.g. "contraftus" -> the real
    but *wrong* word "contrastare"'s family, when the actual fix is
    "contractus"), and no corpus-frequency signal can tell those apart from a
    genuine long-s match without semantic/contextual judgement.

    The unmodified word is *not* offered here -- callers check that separately
    so a word that's already valid never enters the substitution search.
    """
    positions = _f_positions(word)
    if not positions or len(positions) > 6:  # cap: avoid combinatorial blowup
        return None, None

    exact = None
    # Systematic long-s corruption flips *every* ſ in the word (and never
    # touches a genuine f), so try the all-f-flipped reading first and only
    # fall back to partial flips for mixed words.
    for k in range(len(positions), 0, -1):
        best = None  # (-freq, candidate)
        for combo in combinations(positions, k):
            cand = _apply_subst(word, combo)
            freq = vocab.get(_fold(cand), 0)
            if freq >= min_count and (best is None or freq > -best[0]):
                best = (-freq, cand)
        if best is not None:
            exact = best[1]
            break

    stem = None
    if vocab_prefixes is not None:
        for k in range(1, len(positions) + 1):
            for combo in combinations(positions, k):
                cand = _apply_subst(word, combo)
                if _prefix_family_hit(cand, vocab_prefixes):
                    stem = cand
                    break
            if stem:
                break

    return exact, stem


def build_prefix_index(vocab: Counter) -> dict:
    """Set (as dict-of-True) of stems attested in the corpus, one entry per
    (vocab word, ending it matches) -- e.g. "usuram" contributes "usur"."""
    prefixes: dict = {}
    for w in vocab:
        if len(w) < 6:
            continue
        for end in _ENDINGS:
            if w.endswith(end) and len(w) - len(end) >= 4:
                prefixes[w[: -len(end)]] = True
    return prefixes


_build_prefix_index = build_prefix_index  # internal alias used above


@dataclass
class Correction:
    original: str
    corrected: str


@dataclass
class CorrectionResult:
    text: str
    changes: List[Correction] = field(default_factory=list)       # applied -- exact vocab match
    suggestions: List[Correction] = field(default_factory=list)   # NOT applied -- stem match only, needs human review
    unresolved: List[str] = field(default_factory=list)           # tokens with f's we couldn't validate at all


def correct_long_s(text: str, vocab: Counter, vocab_prefixes: Optional[dict] = None) -> CorrectionResult:
    """Repair long-s misreads (and adjacent spurious word-splits) in `text`.

    Only exact vocabulary matches are written into the returned text --
    stem-based matches are collected as `suggestions` but left untouched in
    place, since they aren't reliably distinguishable from this document's
    other, unrelated OCR error (ct misread as ft) without human judgement.

    `vocab_prefixes` should be `build_prefix_index(vocab)`, built once by the
    caller and reused across calls -- pass it explicitly for bulk use so it
    isn't rebuilt per segment.
    """
    if vocab_prefixes is None:
        vocab_prefixes = build_prefix_index(vocab)
    tokens = _TOKEN_RE.findall(text)
    out: List[str] = []
    changes: List[Correction] = []
    suggestions: List[Correction] = []
    unresolved: List[str] = []

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if not _WORD_RE.fullmatch(tok):
            out.append(tok)
            i += 1
            continue

        # Already a known word as-is -> leave untouched.
        if vocab.get(_fold(tok), 0) > 0:
            out.append(tok)
            i += 1
            continue

        # A token butted right up against a digit (no whitespace) is usually
        # a fragment of a word broken by some other OCR artifact (e.g. a
        # misread ligature -- "PR1FATIO" for "PRAEFATIO"), not a genuine
        # standalone word. Fixing it in isolation risks landing on a real but
        # unrelated word (that fragment's "FATIO" -> the real word "satio",
        # sowing/planting -- nothing to do with the actual "praefatio").
        adjacent_digit = (i > 0 and re.search(r"\d", tokens[i - 1])) or \
                          (i + 1 < n and re.search(r"\d", tokens[i + 1]))

        # Single-token f->s fix.
        if (not adjacent_digit) and ("f" in tok or "F" in tok):
            exact, stem = _best_fix(tok, vocab, vocab_prefixes)
        else:
            exact = stem = None
        if exact:
            out.append(exact)
            changes.append(Correction(tok, exact))
            i += 1
            continue
        if stem:
            suggestions.append(Correction(tok, stem))
            # falls through to the merge attempt / unresolved handling below,
            # but tok itself is still emitted unchanged

        # Merge with next 1-2 tokens (only across whitespace) and retry --
        # handles letter-spaced runs like "u fur arum" -> "usurarum". Gated on
        # actual corruption evidence (an f in the window) and a minimum joined
        # length -- without those, short common words merge into unrelated
        # coincidental real words purely by chance (e.g. "a" + "per" ->
        # "aper" [wild boar], "a" + "via" -> "avia" [grandmother], "V" +
        # "runt" -> "Vrunt"/"urunt" [they burn] -- none of these have
        # anything to do with long-s corruption).
        merged_ok = False
        for span in (2, 3):
            end = i + 2 * span - 2
            if end >= n:
                continue
            if adjacent_digit or (end + 1 < n and re.search(r"\d", tokens[end + 1])):
                continue  # window's own edges shouldn't be digit-adjacent fragments either
            window = tokens[i:i + 2 * span - 1]
            if any(not _WORD_RE.fullmatch(w) for w in window[0::2]):
                continue  # word slots must actually be word tokens
            if any(not w.isspace() for w in window[1::2]):
                continue  # separators must be plain whitespace, not punctuation
            if not any("f" in w or "F" in w for w in window[0::2]):
                continue  # no corruption evidence in this window -- don't merge
            joined = "".join(window[0::2])
            if len(joined) < 5 or len(joined) > 20:
                continue
            if vocab.get(_fold(joined), 0) > 0:
                m_exact, m_stem = joined, None
            else:
                m_exact, m_stem = _best_fix(joined, vocab, vocab_prefixes)
            if m_exact:
                out.append(m_exact)
                changes.append(Correction(" ".join(window[0::2]), m_exact))
                i += 2 * span - 1
                merged_ok = True
                break
            if m_stem:
                suggestions.append(Correction(" ".join(window[0::2]), m_stem))
        if merged_ok:
            continue

        if "f" in tok or "F" in tok:
            unresolved.append(tok)
        out.append(tok)
        i += 1

    return CorrectionResult("".join(out), changes, suggestions, unresolved)
