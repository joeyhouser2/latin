"""Expand medieval abbreviations and tidy raw HTR output before translation.

The CATMuS Medieval model transcribes *graphematically*: it writes the glyphs on
the page ("ꝑfecto", "ueni&", "prȩstolabor", "dñs", "Etdehierusalem") and expands
nothing. A translator fed that gets nonsense. This module turns it into
ordinary Latin spelling in four conservative steps:

1. **Unambiguous glyphs** map directly: ȩ -> ae, ⁊ -> et, ꝑ -> per, ꝓ -> pro,
   ꝰ -> us, ꝝ -> rum, ſ -> s, superscript letters are inserted after their base.
2. **Marked letters** (macron / tilde / bar over a letter, i.e. a dropped m, n,
   or syllable) and the word-final ``&`` are *ambiguous*, so candidates are
   generated and the one found in the reference vocabulary (most frequent wins)
   is used. With no vocabulary hit the mark is dropped rather than guessed.
3. **Nomina sacra and stock shorthand** (dns, ds, xps, sps, scs...) expand from
   a table, but only when the source token actually carried an abbreviation
   mark -- plain "ds" or "di" in a text is left alone.
4. **Run-together words** ("Etdehierusalem") are split when, and only when, the
   token is unknown and splits cleanly into known words.

Also drops the junk lines the recogniser invents over musical notation,
decoration and stains (short lines of stray symbols).

The vocabulary comes from the library's own corpus (clean editions), exactly as
``ingest.ocr_fix`` does, so spelling matches what the translator already knows.
Everything here is heuristic: it makes manuscript text *translatable*, not
*edited*, and the original transcription should always be kept alongside.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
_VOCAB_CACHE = _REPO / "data" / "cache" / "latin_vocab.json"

_FOLD = str.maketrans({"v": "u", "j": "i", "ȩ": "e", "ę": "e"})

# ---- step 1: unambiguous glyphs -------------------------------------------
_GLYPH = {
    "ȩ": "ae", "ę": "ae", "Ȩ": "Ae", "Ę": "Ae", "æ": "ae", "Æ": "Ae",
    "œ": "oe", "Œ": "Oe", "⁊": "et", "ꝑ": "per", "Ꝑ": "Per", "ꝓ": "pro",
    "Ꝓ": "Pro", "ꝰ": "us", "ꝝ": "rum", "ꝛ": "r", "ſ": "s", "ꝫ": "et",
    "ꝯ": "con", "Ꝯ": "Con", "ꝙ": "quod", "ꝗ": "quam", "đ": "d", "ħ": "h",
    "ƀ": "b", "ł": "l", "¶": "", "·": "", "⸗": "-", "ʒ": "z",
}
_PRE_NFD = str.maketrans({"ȩ": "ae", "ę": "ae", "Ȩ": "Ae", "Ę": "Ae"})   # NFD would drop the hook
# combining superscript letters U+0363..U+036F -> the letter they stand for
_SUPER = {0x363: "a", 0x364: "e", 0x365: "i", 0x366: "o", 0x367: "u", 0x368: "c",
          0x369: "d", 0x36A: "h", 0x36B: "m", 0x36C: "r", 0x36D: "t", 0x36E: "v",
          0x36F: "x"}
# combining marks that signal "something omitted here"
_ABBREV_MARKS = {0x303, 0x304, 0x305, 0x30A, 0x30C, 0x335, 0x336, 0x33E, 0x342,
                 0x35E, 0x1DC4, 0x1DC5}

# ---- step 2: candidates for a mark over a letter ---------------------------
_AFTER = {  # letter carrying the mark -> strings that may follow it
    "a": ["n", "m"], "e": ["n", "m"], "i": ["n", "m"], "o": ["n", "m"], "u": ["n", "m"],
    "q": ["ue", "uae", "uod", "ui"], "p": ["er", "ro", "re", "ar"],
    "b": ["us", "er"], "c": ["on", "um"], "d": ["e", "i"], "l": ["l"],
    "m": ["m", "em"], "n": ["n", "on"], "r": ["um", "er"], "s": ["is", "us"],
    "t": ["ur", "er"], "v": ["er"],
}

# ---- step 3: shorthand, applied only when the token carried a mark ----------
_SHORT = {
    "dns": "dominus", "dni": "domini", "dno": "domino", "dnm": "dominum", "dne": "domine",
    "ds": "deus", "dm": "deum", "dei": "dei", "ihs": "iesus", "ihc": "iesus",
    "ihm": "iesum", "ihu": "iesu", "xps": "christus", "xpc": "christus", "xpi": "christi",
    "xpm": "christum", "xpo": "christo", "sps": "spiritus", "spu": "spiritu",
    "spm": "spiritum", "scs": "sanctus", "sci": "sancti", "sco": "sancto",
    "scm": "sanctum", "sca": "sancta", "scae": "sanctae", "eps": "episcopus",
    "epi": "episcopi", "epm": "episcopum", "ptr": "pater", "mr": "mater",
    "frs": "fratres", "fr": "frater", "oms": "omnes", "omnps": "omnipotens",
    "n": "non", "e": "est", "qd": "quod", "qm": "quoniam", "qn": "quando",
    "q": "que", "h": "hoc", "p": "pro", "ꝑ": "per", "i": "in", "t": "tamen",
    "ee": "esse", "ei": "enim", "sm": "secundum", "scd": "secundum", "aie": "animae",
    "ai": "anima", "ep": "episcopus", "gla": "gloria", "glie": "gloriae",
    "mia": "misericordia", "miae": "misericordiae", "reg": "regem",
}

_LETTERS = "A-Za-zÀ-ɏꜰ-ꟿ"
_TOKEN = re.compile(rf"^([^{_LETTERS}&⁊\d]*)(.*?)([^{_LETTERS}&⁊\d]*)$", re.DOTALL)


# ---- vocabulary ------------------------------------------------------------
def _fold(w: str) -> str:
    return w.lower().translate(_FOLD)


def build_vocab(db_path: Optional[str] = None, min_count: int = 2,
                refresh: bool = False) -> Counter:
    """Folded-word frequency counter from the corpus (cached on disk).

    Reads ``corpus.db`` read-only; never writes to it. The cache means this
    scan (a minute) happens once, not per manuscript.
    """
    if _VOCAB_CACHE.exists() and not refresh:
        return Counter(json.loads(_VOCAB_CACHE.read_text(encoding="utf-8")))
    path = Path(db_path or _REPO / "data" / "corpus.db")
    if not path.exists():
        return Counter()
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=30)
    word = re.compile(r"[A-Za-z]+")
    vocab: Counter = Counter()
    try:
        for (text,) in conn.execute(
                "SELECT s.latin_text FROM segments s "
                "JOIN sections sec ON s.section_id = sec.id "
                "JOIN documents d ON sec.doc_id = d.id WHERE d.language = 'la'"):
            for w in word.findall(text or ""):
                if len(w) >= 2:
                    vocab[_fold(w)] += 1
    finally:
        conn.close()
    vocab = Counter({w: c for w, c in vocab.items() if c >= min_count})
    _VOCAB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _VOCAB_CACHE.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    return vocab


# ---- per-token machinery ----------------------------------------------------
def _decompose(token: str) -> Tuple[str, bool]:
    """Resolve combining marks. Returns (text with superscripts inserted and
    marks removed, whether an omission mark was present)."""
    marked = False
    out: List[str] = []
    for ch in unicodedata.normalize("NFD", token):
        cp = ord(ch)
        if cp in _SUPER:
            out.append(_SUPER[cp])
            marked = True
        elif unicodedata.combining(ch):
            if cp in _ABBREV_MARKS:
                marked = True
                out.append("\0")                      # placeholder after the base letter
            # other combining marks (acute, diaeresis...) are dropped
        else:
            out.append(ch)
    return "".join(out), marked


def _mark_candidates(base: str) -> List[str]:
    """Expansions of a string whose \\0 markers sit after the marked letters."""
    if "\0" not in base:
        return [base]
    i = base.index("\0")
    head, tail = base[:i], base[i + 1:]
    letter = head[-1:].lower()
    results: List[str] = []
    for ins in _AFTER.get(letter, []):
        for rest in _mark_candidates(tail):
            results.append(head + ins + rest)
    for rest in _mark_candidates(tail):               # also: mark was decorative
        results.append(head + rest)
    return results


def _best(cands: List[str], vocab: Counter) -> Optional[str]:
    scored = [(vocab.get(_fold(c), 0), -i, c) for i, c in enumerate(cands)]
    scored = [s for s in scored if s[0] > 0]
    return max(scored)[2] if scored else None


def _glyphs(token: str) -> str:
    return "".join(_GLYPH.get(ch, ch) for ch in token)


def expand_token(token: str, vocab: Counter) -> str:
    m = _TOKEN.match(token)
    pre, core, post = m.groups() if m else ("", token, "")
    if not core:
        return token
    core = unicodedata.normalize("NFC", core).translate(_PRE_NFD)
    base, marked = _decompose(core)
    had_special = any(ch in _GLYPH and ch not in "ȩęȨĘæÆœŒ¶·ſ" for ch in core)
    marked = marked or had_special

    if core.startswith("&") and len(core) > 1:             # '&liberabo' = 'et liberabo'
        return pre + "et " + expand_token(core[1:], vocab) + post
    # word-final or standalone '&': 'et', or the -et/-it/-t ending of a verb
    if core == "&":
        return pre + "et" + post
    if base.endswith("&") and len(base) > 1:
        stem = base[:-1]
        cands = [stem + "et", stem + "it", stem + "t", stem + "ue"]
        pick = _best([_glyphs(c).replace("\0", "") for c in cands], vocab)
        return pre + (pick or _glyphs(stem) + "et") + post

    stripped = _glyphs(base).replace("\0", "").lower()
    if marked and stripped in _SHORT:
        out = _SHORT[stripped]
        return pre + (out.capitalize() if core[:1].isupper() else out) + post

    cands = [_glyphs(c) for c in _mark_candidates(base)]
    cands = [c for c in cands if c]
    if not cands:
        return pre + post
    # ꝑ/ꝓ/ꝯ have alternate readings: try the other prefixes too
    extra = []
    for c in cands:
        if "per" in c and "ꝑ" in core:
            extra += [c.replace("per", "par", 1), c.replace("per", "por", 1)]
        if "con" in c and "ꝯ" in core.lower():
            extra += [c.replace("con", "com", 1)]
    cands = cands + extra
    if marked:
        pick = _best(cands, vocab)
        if pick is None and len(cands) > 0:
            # no vocabulary evidence: prefer the plain n reading for a vowel mark
            pick = next((c for c in cands if "n" in c), cands[0]) if "\0" in base else cands[0]
    else:
        pick = cands[0]
    if core[:1].isupper() and pick[:1].islower():
        pick = pick[:1].upper() + pick[1:]
    return pre + pick + post


_PREFIXES = ("et", "de", "in", "ad", "ex", "ab", "cum", "per", "pro", "ut", "non",
             "sed", "qui", "quae", "quod", "si", "ne", "sub", "ante", "post", "inter")


def split_run_together(word: str, vocab: Counter, max_prefixes: int = 2) -> Optional[List[str]]:
    """Split scribal run-togethers: function words glued onto the next word.

    Medieval hands write "inmulieribus", "dedomo", "Etdehierusalem" with no space
    after short prepositions/conjunctions. Only that pattern is handled -- one
    or two function-word prefixes followed by a remainder that is itself a known
    word -- because splitting arbitrary unknown words into vocabulary pieces
    wrecks genuine inflected forms the (corpus-sized) vocabulary happens to lack.
    """
    w = _fold(word)
    if len(w) < 6 or not w.isalpha() or vocab.get(w, 0) > 0:
        return None
    parts: List[str] = []
    pos = 0
    for _ in range(max_prefixes):
        hit = None
        for p in sorted(_PREFIXES, key=len, reverse=True):
            rest = w[pos + len(p):]
            if w.startswith(p, pos) and len(rest) >= 3 and (
                    vocab.get(rest, 0) >= 2
                    or any(rest.startswith(q) and len(rest) > len(q) + 2 for q in _PREFIXES)):
                hit = p
                break
        if not hit:
            break
        parts.append(word[pos:pos + len(hit)])
        pos += len(hit)
        if vocab.get(w[pos:], 0) >= 2:
            break
    if not parts or vocab.get(w[pos:], 0) < 2 and len(parts) < 2:
        return None
    return parts + [word[pos:]]


def _is_junk_line(line: str, vocab: Counter) -> bool:
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return True
    if len(letters) / max(len(line.replace(" ", "")), 1) < 0.55:
        return True
    return len(letters) < 4


def expand_text(text: str, vocab: Optional[Counter] = None,
                split_words: bool = True, drop_junk: bool = True) -> str:
    """Expand a raw HTR transcription into translatable Latin."""
    vocab = vocab if vocab is not None else build_vocab()
    out_lines: List[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if not line.strip():
            out_lines.append("")
            continue
        if drop_junk and _is_junk_line(line, vocab):
            continue
        toks = []
        for tok in line.split():
            e = expand_token(tok, vocab)
            if split_words and vocab:
                m = _TOKEN.match(e)
                if m and m.group(2):
                    parts = split_run_together(m.group(2), vocab)
                    if parts:
                        e = m.group(1) + " ".join(parts) + m.group(3)
            if e:
                toks.append(e)
        if toks:
            out_lines.append(" ".join(toks))
    return unicodedata.normalize("NFC", "\n".join(out_lines))
