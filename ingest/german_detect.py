"""Detect German editorial prose embedded in Latin documents.

Critical editions like the Analecta Hymnica (Dreves/Blume) interleave the Latin
hymn texts with German apparatus and commentary ("Die Melodie von 1, 2 u.",
"Abgesehen von einigen Differenzen steht die Gruppe der Hss."). OCR segments
those into ordinary segments, and sending them through the Latin translator
yields fabricated nonsense ("The day of Melody, of 1, 2 u.") -- NLLB reads
"die" as Latin "day". Stock NLLB handles German well when told it *is*
German (src_lang deu_Latn), so the job here is just routing.

High precision matters more than recall: a Latin segment misrouted as German
gets a worse translation than it had. So a segment counts as German only when
it has at least MIN_HITS German function words that are *not* also Latin words
(or common Latin OCR fragments), and those make up a meaningful share of its
tokens. Words that are also Latin ("die" = on the day, "das" = you give,
"fur" = thief, "an", "in", "es", "des", "da", "so") or that turn up in Latin
text as name particles ("von") never qualify a segment on their own; they
only add to the ratio once unambiguous German is present. The lists include the usual
Fraktur/antiqua OCR corruptions seen in these volumes (ß -> "fs", ü -> "ii").
"""
from __future__ import annotations

import re

# German function words that do not occur as Latin words.
_UNAMBIGUOUS = frozenset("""
und der ist nicht mit sich den dem ein eine einer eines einem einen
vom zum zur bei aus nach auf fiir für uber iiber über oder aber auch
wie als noch nur wird werden wurde wurden sind hat haben hatte war waren ich
wir sie ihr ihre ihrer ihren sein seine seiner seinen dieser diese dieses
diesem diesen jedoch dass dafs daß welche welcher welches zwei drei
im am beim wo wenn weil sondern denn dann schon hier
""".split()) | frozenset("""
melodie fehlen fehlt sequenz sequenzen strophen lesart lesarten varianten
handschrift handschriften zweimal titel nebst statt vgl
""".split())  # Dreves/Blume apparatus vocabulary ("strophe", "bis" are Latin too)

# German function words that are also Latin words / frequent Latin tokens.
_AMBIGUOUS = frozenset("die das des es an in da so um fur von".split())

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

MIN_HITS = 2        # unambiguous German function words required
MIN_RATIO = 0.15    # (unambiguous + ambiguous) German tokens / all word tokens


def german_score(text: str) -> tuple:
    """(unambiguous_hits, ratio) for a segment."""
    words = [w.lower() for w in _WORD.findall(text or "")]
    if not words:
        return 0, 0.0
    hard = sum(w in _UNAMBIGUOUS for w in words)
    soft = sum(w in _AMBIGUOUS for w in words)
    return hard, (hard + soft) / len(words)


def is_german(text: str) -> bool:
    hard, ratio = german_score(text)
    return hard >= MIN_HITS and ratio >= MIN_RATIO
