"""Translation-status lookup against published series catalogs.

Complements the Repertorium Fontium (medieval Latin narrative sources only):
these series cover the classical/late-antique/patristic material Repertorium
doesn't touch, plus Greek.

Three sources, all matched with the same generic-title-guarded fuzzy matcher
as the Repertorium lookup (see ``ingest.fuzzy_match``):

- **Loeb Classical Library** — live site search (loebclassics.com/search).
  A commercial, subscription-gated edition -> a match means
  ``TRANSLATED_PAYWALLED``, not ``TRANSLATED``: the text has an English
  version, but it isn't freely readable, so our own MT rendering still adds
  real value.
- **Dumbarton Oaks Medieval Library (DOML)** — static list of the 93 published
  volumes (scraped once from Wikipedia's volume table into
  ``data/clean/doml_volumes.json``; DOML's own site is a JS-rendered SPA that
  doesn't return content to a plain HTTP GET). Also commercial ->
  ``TRANSLATED_PAYWALLED``.
- **ANF / NPNF** (Ante-Nicene / Nicene and Post-Nicene Fathers) — static list
  of ~420 works scraped once from the New Advent index
  (``data/clean/anf_npnf_works.json``). These are public-domain 19th-century
  translations hosted free at newadvent.org/ccel.org -> a match means
  ``TRANSLATED`` (freely available), not paywalled.

Usage:
    from ingest.series_catalogs import SeriesCatalogLookup
    lk = SeriesCatalogLookup()
    res = lk.lookup("In L. Catilinam", author="M. Tullius Cicero")
    print(res.status, res.source, res.matched_title)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import List, Optional, Tuple

import requests

from .translation_status import TRANSLATED, TRANSLATED_PAYWALLED, UNKNOWN
from .fuzzy_match import best_match, clean_search_title

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "clean")
_DOML_JSON = os.path.join(_DATA_DIR, "doml_volumes.json")
_ANF_NPNF_JSON = os.path.join(_DATA_DIR, "anf_npnf_works.json")


@dataclass
class SeriesResult:
    status: str                       # TRANSLATED | TRANSLATED_PAYWALLED | UNKNOWN
    reason: str
    source: Optional[str] = None      # "Loeb" | "DOML" | "ANF/NPNF"
    matched_title: Optional[str] = None
    match_score: float = 0.0

    @property
    def has_english(self) -> bool:
        return self.status in (TRANSLATED, TRANSLATED_PAYWALLED)


class _StaticListLookup:
    """Shared logic for the two scraped-once JSON catalogs (DOML, ANF/NPNF)."""

    source_name = "?"
    result_status = TRANSLATED_PAYWALLED
    threshold = 0.62

    def __init__(self):
        self._entries: Optional[List[Tuple]] = None

    def _load(self) -> List[Tuple]:
        raise NotImplementedError

    def _title_of(self, entry) -> str:
        raise NotImplementedError

    def _context_of(self, entry) -> str:
        raise NotImplementedError

    def lookup(self, title: str, author: Optional[str] = None) -> SeriesResult:
        if self._entries is None:
            self._entries = self._load()
        m = best_match(
            title, author, self._entries,
            title_of=self._title_of, context_of=self._context_of,
            threshold=self.threshold,
        )
        if m.candidate is None:
            return SeriesResult(UNKNOWN, f"no confident {self.source_name} match",
                                match_score=m.score)
        return SeriesResult(
            self.result_status, f"matched in {self.source_name} catalog",
            source=self.source_name, matched_title=self._title_of(m.candidate),
            match_score=m.score,
        )


class DOMLLookup(_StaticListLookup):
    """Dumbarton Oaks Medieval Library — 93 volumes, commercial (Harvard UP)."""

    source_name = "DOML"
    result_status = TRANSLATED_PAYWALLED

    def _load(self) -> List[Tuple]:
        if not os.path.isfile(_DOML_JSON):
            return []
        with open(_DOML_JSON, encoding="utf-8") as f:
            rows = json.load(f)  # [(volume, title_text, language), ...]
        return [tuple(r) for r in rows]

    def _title_of(self, entry) -> str:
        return entry[1]

    def _context_of(self, entry) -> str:
        return entry[1]  # author names are embedded in the title text itself


class ANFNPNFLookup(_StaticListLookup):
    """Ante-Nicene / Nicene & Post-Nicene Fathers — free at newadvent.org."""

    source_name = "ANF/NPNF"
    result_status = TRANSLATED

    def _load(self) -> List[Tuple]:
        if not os.path.isfile(_ANF_NPNF_JSON):
            return []
        with open(_ANF_NPNF_JSON, encoding="utf-8") as f:
            rows = json.load(f)  # [(author, title, work_id), ...]
        return [tuple(r) for r in rows]

    def _title_of(self, entry) -> str:
        return entry[1]

    def _context_of(self, entry) -> str:
        return entry[0]  # author name, separate from the title


class LoebLookup:
    """Loeb Classical Library — live site search, subscription-gated."""

    BASE = "https://www.loebclassics.com"
    SEARCH = BASE + "/search"
    source_name = "Loeb"

    def __init__(self, timeout: float = 30.0, match_threshold: float = 0.62):
        self.timeout = timeout
        self.match_threshold = match_threshold
        self.session = requests.Session()
        # A generic browser UA is required — the site 403s the default
        # python-requests UA (Loeb's WAF, not a robots-disallow).
        self.session.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0 Safari/537.36"),
        })

    def search(self, query: str, limit: int = 25) -> List[Tuple[str, str]]:
        """Return [(author, title), ...] candidates from Loeb's site search."""
        resp = self.session.get(self.SEARCH, params={"q": query}, timeout=self.timeout)
        resp.raise_for_status()
        html = resp.text
        out = []
        for m in re.finditer(
            r'bibliographicAuthorName">(.*?)</span>.*?workTitle">(.*?)</span>',
            html, re.S,
        ):
            author = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            out.append((author, title))
            if len(out) >= limit:
                break
        return out

    def lookup(self, title: str, author: Optional[str] = None) -> SeriesResult:
        try:
            candidates = self.search(clean_search_title(title))
        except Exception as exc:                       # network failure
            return SeriesResult(UNKNOWN, f"lookup failed: {exc}")
        if not candidates:
            return SeriesResult(UNKNOWN, "no Loeb match")

        m = best_match(
            title, author, candidates,
            title_of=lambda c: c[1], context_of=lambda c: c[0],
            threshold=self.match_threshold,
        )
        if m.candidate is None:
            return SeriesResult(UNKNOWN, "no confident Loeb match", match_score=m.score)
        return SeriesResult(
            TRANSLATED_PAYWALLED, "matched in Loeb catalog", source="Loeb",
            matched_title=m.candidate[1], match_score=m.score,
        )


class SeriesCatalogLookup:
    """Tries ANF/NPNF, then DOML, then Loeb; returns the first confident hit."""

    def __init__(self):
        self.anf_npnf = ANFNPNFLookup()
        self.doml = DOMLLookup()
        self.loeb = LoebLookup()

    def lookup(self, title: str, author: Optional[str] = None) -> SeriesResult:
        for lk in (self.anf_npnf, self.doml, self.loeb):
            res = lk.lookup(title, author)
            if res.status != UNKNOWN:
                return res
        return SeriesResult(UNKNOWN, "no match in any series catalog")
