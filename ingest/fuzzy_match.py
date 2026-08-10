"""Shared title/author matching for translation-status lookups.

Every "does this work already have a translation?" source (Repertorium Fontium,
series catalogs, Wikidata) is queried by title and gets back a handful of
loosely-similar candidates. The author is normally just a soft disambiguator —
but for a bare generic title ("carmina", "versus", "epistolae", ...) title
similarity alone is worthless: dozens of unrelated works share the same
one-word title, so string similarity picks *some* candidate with total
confidence regardless of who actually wrote it. Concretely: an earlier version
of the Repertorium lookup collapsed 26 different "carmina" documents onto one
werk id this way.

The fix used everywhere here: for generic titles, require the author to
actually appear in the candidate's context text, or stay honestly ``unknown``
rather than guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Generic, List, Optional, Sequence, TypeVar
import re

T = TypeVar("T")

# Latin/English nouns so commonly reused as bare work-titles or cataloguing
# labels (many unrelated authors' "Carmina", "Versus", "Epistolae",
# "Carminum Appendix", "Sermons"...) that a title-only match against them is
# not trustworthy evidence of *which* work it is. Includes common inflected
# forms since Latin declines them (carmen/carmina/carminum/carminibus, ...) —
# stemming would be more complete but risks swallowing real distinguishing
# words (e.g. the proper name "Vitalis" via a bare "vit" stem), so this stays
# an explicit, deliberately generous word list instead.
GENERIC_WORDS = {
    "carmen", "carmina", "carminum", "carminibus", "carmine",
    "versus", "versuum", "versibus",
    "epistola", "epistolae", "epistolarum", "epistula", "epistulae",
    "epistularum", "epistle", "epistles", "letter", "letters",
    "sermo", "sermones", "sermonum", "sermon", "sermons",
    "vita", "vitae", "vitarum", "life",
    "passio", "passiones", "passionum", "miracula", "miraculorum", "miracles",
    "homilia", "homiliae", "homiliarum", "homily", "homilies",
    "hymnus", "hymni", "hymnorum", "hymn", "hymns",
    "opuscula", "opusculorum", "tractatus", "treatise",
    "gesta", "chronica", "chronicon", "chronicle",
    "annales", "annalium", "annals", "visio", "visiones", "vision",
    "martyrologium", "necrologium", "regula", "regulae",
    "capitula", "capitulare", "poems", "poem",
    "appendix", "appendicis", "appendices",
    "opera", "operum", "fragmenta", "fragmentum", "fragmentorum",
    "varia", "miscellanea", "excerpta",
}
_ROMAN_OR_NUM = re.compile(r"^(?:[ivxlcdm]+|\d+)$")


def norm(s: Optional[str]) -> str:
    """Lowercase, drop punctuation/parentheticals for fuzzy comparison."""
    s = (s or "").lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_search_title(title: str) -> str:
    """Trim a library title to a clean head suitable for a literal-ish external
    search: cut at the first structural separator, drop editorial recension
    tails, and drop single-letter abbreviations (Roman praenomina like "L."
    in "In L. Catilinam") that external search endpoints often treat as
    literal tokens rather than ignoring."""
    head = re.split(r"[/(:;]", title, 1)[0]
    head = re.sub(r"\b(recensio|rec\.|pars|liber|lib\.)\b.*$", "", head, flags=re.I)
    head = re.sub(r"\b[A-Z]\.\s*", "", head)
    return re.sub(r"\s+", " ", head).strip(" ,.-") or title


# Genre-marker words that lead a "<genre> of/about <name>" pattern reused
# across many unrelated works on the same popular subject — "Vita Antonii"/
# "Passio ..."/"Gesta ..." exist for dozens of different saints/rulers, so a
# title starting with one of these is not safe to trust on string similarity
# alone even once a proper name is attached (unlike GENERIC_WORDS, which
# requires *every* word to be generic, this fires on the lead word alone).
# Hagiography especially: many "Vita X" survive only in Latin while a
# DIFFERENT "Vita X" of the same saint has long been Englished.
_GENRE_MARKER_HEADS = {
    "vita", "vitae", "passio", "passiones", "miracula", "gesta",
    "sermo", "sermones", "homilia", "homiliae",
    "epistola", "epistolae", "epistula", "epistulae",
    "carmen", "carmina", "carminum", "versus", "hymnus", "hymni",
    "visio", "chronica", "chronicon", "tractatus", "opuscula",
    "regula", "capitula", "capitulare", "annales",
    "martyrologium", "necrologium",
}


def is_generic_title(norm_title: str) -> bool:
    """True if a normalised title isn't safe to trust on string similarity
    alone: either every content word is a generic cataloguing label
    ("carmina", "carminum appendix"), or it opens with a genre-marker word
    that leads a reused "<genre> of <name>" pattern ("Vita Antonii", "Gesta
    Berengarii") — plenty of different works share that shape."""
    content = [w for w in norm_title.split() if not _ROMAN_OR_NUM.match(w)]
    if not content:
        return True
    if all(w in GENERIC_WORDS for w in content):
        return True
    return content[0] in _GENRE_MARKER_HEADS


@dataclass
class MatchResult(Generic[T]):
    candidate: Optional[T]
    score: float
    confident: bool
    author_matched: bool


def best_match(
    want_title: str,
    want_author: Optional[str],
    candidates: Sequence[T],
    *,
    title_of: Callable[[T], str],
    context_of: Callable[[T], str],
    threshold: float = 0.62,
    author_bonus: float = 0.1,
) -> MatchResult[T]:
    """Best-match ``want_title``/``want_author`` against ``candidates``.

    ``title_of(c)`` returns the candidate's comparable title text.
    ``context_of(c)`` returns extra searchable text (sort key, author field,
    blurb, ...) the author surname might appear in, for disambiguation.

    Not confident (``candidate`` is ``None``) when: no candidates, best score
    under ``threshold``, or the title is generic and the author doesn't
    corroborate the match — matching the Repertorium fix's logic exactly.
    """
    if not candidates:
        return MatchResult(None, 0.0, False, False)

    want = norm(want_title)
    author_key = norm(want_author).split()[-1] if want_author else None

    def score_of(c: T) -> tuple[float, bool]:
        s = SequenceMatcher(None, want, norm(title_of(c))).ratio()
        author_matched = bool(author_key and author_key in norm(context_of(c)))
        if author_matched:
            s = min(s + author_bonus, 1.0)
        return s, author_matched

    scored = [(c, *score_of(c)) for c in candidates]
    best_c, best_s, best_am = max(scored, key=lambda t: t[1])

    if best_s < threshold:
        return MatchResult(None, best_s, False, best_am)
    if is_generic_title(want) and not best_am:
        return MatchResult(None, best_s, False, best_am)
    return MatchResult(best_c, best_s, True, best_am)
