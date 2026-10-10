"""Which words and lines of a transcription are worth a human look.

Recognisers give a confidence per character; combined with the known-word check this
points at the likely misreadings, so a person can correct the dozen suspect words on a
page instead of reading all of it. No extra model is involved.

Word flags (``f``): 0 fine, 1 weak (look if time), 2 suspect (look).
  * suspect: confidence under ``BAD_CONF``, or under ``WEAK_CONF`` and not a known word;
  * weak: confidence under ``WEAK_CONF``, or a long unknown word (probably two words run
    together -- confidence cannot see word-division errors).
Line flag = its worst word; ``score`` orders pages by how much needs checking.

Honest limits: CTC confidences are optimistic, so a misreading the model is sure about
passes; and a correct rare word (a name, a late form) is flagged. It finds probable
errors, not all of them -- calibrate against hand-corrected pages before trusting it.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from . import abbrev, greek

BAD_CONF = 0.50
WEAK_CONF = 0.80
RUN_TOGETHER_LEN = 9


def _known(word: str, vocab: Counter, language: str) -> bool:
    if language not in ("la", "grc"):
        return True                      # no vocabulary for this language: confidence only
    if language == "grc":
        return greek.hit_rate(word, vocab) > 0 or len(word) < 3
    stripped = "".join(ch for ch in word if ch.isalpha())
    if len(stripped) < 3:
        return True
    return abbrev.vocab_hit_rate(abbrev.expand_text(stripped, vocab), vocab) > 0


def analyze(lines: List[Dict[str, Any]], vocab: Counter, language: str = "la") -> Dict[str, Any]:
    """Add flags to recogniser detail (``htr_worker.line_detail`` / ``ocr_words`` format)."""
    out, bad, weak, n_words = [], 0, 0, 0
    for line in lines:
        words = []
        for w in line.get("w", []):
            known = _known(w["t"], vocab, language)
            c = w.get("c", 1.0)
            flag = 0
            if c < BAD_CONF or (c < WEAK_CONF and not known):
                flag = 2
            elif c < WEAK_CONF or (not known and len(w["t"]) >= RUN_TOGETHER_LEN):
                flag = 1
            words.append({**w, "f": flag, "k": known})
            n_words += 1
            bad += flag == 2
            weak += flag == 1
        worst = max((w["f"] for w in words), default=0)
        out.append({**line, "w": words, "f": worst})
    return {"lines": out, "words": n_words, "suspect": bad, "weak": weak,
            "score": round(bad / n_words, 3) if n_words else 0.0}
