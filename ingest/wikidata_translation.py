"""Translation-status lookup against Wikidata.

Broadest-coverage signal (any work notable enough to have a Wikidata item,
any era, any language) but the sparsest per-item data, so it's the last
source tried, after the Repertorium and series catalogs.

Two calls per work:
  1. ``wbsearchentities`` — free-text search for candidate items (fast, no
     SPARQL needed just to find candidates).
  2. A SPARQL query on the best-matched candidate's P747 ("has edition or
     translation") statements, checking each edition's P407 ("language of
     work or name"):
       - an edition in English (Q1860)      -> TRANSLATED
       - editions exist, none in English    -> UNTRANSLATED (into English)
       - no P747 statements at all          -> UNKNOWN (no signal)

As with the Repertorium and series-catalog lookups, a bare generic title only
counts as a confident match if the author corroborates it — here via the
candidate's Wikidata *description*, which usually names the author
("speech by Cicero", "poem by ...").

Usage:
    from ingest.wikidata_translation import WikidataLookup
    r = WikidataLookup().lookup("In L. Catilinam", author="M. Tullius Cicero")
    print(r.status, r.qid, r.reason)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests

from .translation_status import TRANSLATED, UNTRANSLATED, UNKNOWN
from .fuzzy_match import best_match, clean_search_title

_ENGLISH_QID = "Q1860"


@dataclass
class WikidataResult:
    status: str                       # TRANSLATED | UNTRANSLATED | UNKNOWN
    reason: str
    qid: Optional[str] = None
    matched_title: Optional[str] = None
    match_score: float = 0.0

    @property
    def has_english(self) -> bool:
        return self.status == TRANSLATED


class WikidataLookup:
    SEARCH = "https://www.wikidata.org/w/api.php"
    SPARQL = "https://query.wikidata.org/sparql"

    def __init__(self, timeout: float = 30.0, match_threshold: float = 0.62):
        self.timeout = timeout
        self.match_threshold = match_threshold
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly; jth156@case.edu)"}
        )

    # -- public API ------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> List[Tuple[str, str, str]]:
        """Return [(qid, label, description), ...] candidates."""
        resp = self.session.get(self.SEARCH, params={
            "action": "wbsearchentities", "search": query, "language": "en",
            "format": "json", "type": "item", "limit": limit,
        }, timeout=self.timeout)
        resp.raise_for_status()
        out = []
        for r in resp.json().get("search", []):
            label = (r.get("label") or "")
            desc = (r.get("description") or "")
            out.append((r["id"], label, desc))
        return out

    def editions(self, qid: str) -> List[str]:
        """English names of languages this work has an edition/translation in.

        Checked three ways, since a title search can land on either the
        abstract "work" item or a specific "edition" item: (1) qid IS the work
        -> follow P747 "has edition or translation" to each edition's P407
        "language of work"; (2) qid IS an edition itself, tagged directly with
        P407; (3) qid is an edition -> follow P629 "edition or translation of"
        up to the parent work, then its other P747 editions' languages."""
        query = (
            "SELECT DISTINCT ?langLabel WHERE { "
            f"{{ wd:{qid} wdt:P747 ?ed . ?ed wdt:P407 ?lang . }} "
            f"UNION {{ wd:{qid} wdt:P407 ?lang . }} "
            f"UNION {{ wd:{qid} wdt:P629 ?work . ?work wdt:P747 ?ed . "
            "?ed wdt:P407 ?lang . } "
            'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }'
        )
        resp = self.session.get(self.SPARQL, params={"query": query},
                                headers={"Accept": "application/sparql-results+json"},
                                timeout=self.timeout)
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
        return [b["langLabel"]["value"] for b in bindings if "langLabel" in b]

    def lookup(self, title: str, author: Optional[str] = None) -> WikidataResult:
        try:
            candidates = self.search(clean_search_title(title))
        except Exception as exc:                        # network/JSON failure
            return WikidataResult(UNKNOWN, f"lookup failed: {exc}")
        if not candidates:
            return WikidataResult(UNKNOWN, "no Wikidata match")

        m = best_match(
            title, author, candidates,
            title_of=lambda c: c[1], context_of=lambda c: c[2],
            threshold=self.match_threshold,
        )
        if m.candidate is None:
            return WikidataResult(UNKNOWN, "no confident Wikidata match",
                                  match_score=m.score)

        qid, label, _ = m.candidate
        try:
            langs = self.editions(qid)
        except Exception as exc:
            return WikidataResult(UNKNOWN, f"edition lookup failed: {exc}",
                                  qid=qid, matched_title=label, match_score=m.score)

        has_eng = any(l.lower() == "english" for l in langs)
        if has_eng:
            status, reason = TRANSLATED, "Wikidata lists an English edition"
        elif langs:
            status = UNTRANSLATED
            reason = f"editions listed but none English ({', '.join(langs)})"
        else:
            status = UNKNOWN
            reason = "no edition/translation statements on Wikidata"
        return WikidataResult(status, reason, qid=qid, matched_title=label,
                              match_score=m.score)
