"""Translation-status lookup against the Geschichtsquellen des deutschen
Mittelalters / Repertorium Fontium Historiae Medii Aevi (BADW).

The Repertorium records, per work, an explicit "Übersetzungen" (translations)
section broken down by language — so it is a direct, authoritative signal of
whether a Latin work has been Englished. Coverage: narrative historical sources
of the medieval German Reich, c. 750-1500 (Latin/German, NOT Greek), so this is
useful mainly for the MGH-style Latin material, not the Byzantine Greek corpus.

No public API; the site's search is backed by a JSON endpoint and work records
are plain HTML, both of which we read directly:

    search:   https://www.geschichtsquellen.de/filter/json?text=<query>
              -> rows with a /werk/<id> link and a <cite>title</cite>
    article:  https://www.geschichtsquellen.de/werk/<id>
              -> an <h2>Übersetzungen</h2> block followed by per-language <h3>s

The result maps onto our translation_status vocabulary, English-specifically
(has_existing_translation means *English* translation exists):
  - English <h3> present                         -> TRANSLATED
  - work found, translations listed but no English-> UNTRANSLATED (into English)
  - work found, no translations section at all    -> UNTRANSLATED (none recorded)
  - no confident title match                      -> UNKNOWN (don't guess)

Usage:
    from ingest.repertorium import RepertoriumLookup
    r = RepertoriumLookup()
    res = r.lookup("Vita Karoli Magni", author="Einhardus")
    print(res.status, res.has_english, res.languages, res.werk_id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from difflib import SequenceMatcher
import re

import requests

from .translation_status import TRANSLATED, UNTRANSLATED, UNKNOWN
from .fuzzy_match import best_match, norm as _norm, clean_search_title as _search_title


# German names (as they appear in <h3> headings) for the English language.
_ENGLISH_NAMES = ("englisch",)


@dataclass
class RepertoriumResult:
    status: str                       # TRANSLATED | UNTRANSLATED | UNKNOWN
    reason: str
    werk_id: Optional[int] = None
    matched_title: Optional[str] = None
    match_score: float = 0.0
    languages: List[str] = field(default_factory=list)  # German language names found

    @property
    def has_english(self) -> bool:
        return self.status == TRANSLATED


class RepertoriumLookup:
    BASE = "https://www.geschichtsquellen.de"
    SEARCH = BASE + "/filter/json"
    WERK = BASE + "/werk/{wid}"

    def __init__(self, timeout: float = 30.0, match_threshold: float = 0.62):
        self.timeout = timeout
        self.match_threshold = match_threshold
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly; jth156@case.edu)"}
        )

    # -- public API ----------------------------------------------------------

    def search(self, query: str, limit: int = 25) -> List[Tuple[int, str, str]]:
        """Return [(werk_id, title, sort_title), ...] for a free-text query."""
        resp = self.session.get(self.SEARCH, params={"text": query}, timeout=self.timeout)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
        out: List[Tuple[int, str, str]] = []
        for r in rows:
            cell = (r.get("1") or {})
            html = cell.get("_", "")
            m = re.search(r"/werk/(\d+)", html)
            cite = re.search(r"<cite>(.*?)</cite>", html, re.S)
            if not m:
                continue
            title = re.sub(r"<[^>]+>", "", cite.group(1)).strip() if cite else ""
            out.append((int(m.group(1)), title, (cell.get("s") or title).lower()))
            if len(out) >= limit:
                break
        return out

    def translations(self, werk_id: int) -> List[str]:
        """Language names (German, e.g. 'Englisch') under the work's
        Übersetzungen section. Empty list if there is no such section."""
        resp = self.session.get(self.WERK.format(wid=werk_id), timeout=self.timeout)
        resp.raise_for_status()
        html = resp.text
        # Isolate the Übersetzungen <h2> block (up to the next <h2> or end).
        # Match on "bersetzung" to dodge the ü/encoding question.
        start = re.search(r"<h2[^>]*>[^<]*bersetzung", html, re.I)
        if not start:
            return []
        rest = html[start.end():]
        nxt = re.search(r"<h2[^>]*>", rest)
        block = rest[: nxt.start()] if nxt else rest
        langs = [re.sub(r"<[^>]+>", "", h).strip()
                 for h in re.findall(r"<h3[^>]*>(.*?)</h3>", block, re.S)]
        return [l for l in langs if l]

    def lookup(self, title: str, author: Optional[str] = None) -> RepertoriumResult:
        """Best-match a work and infer its English-translation status."""
        # The JSON search ANDs its words, and author names usually aren't in the
        # searchable title text — so search by TITLE and use the author only to
        # disambiguate among candidates. Library titles often carry editorial
        # tails ("/Liber P...", "recensio prior", "(Getica)") that the AND-search
        # chokes on, so search on the cleaned head of the title.
        try:
            candidates = self.search(_search_title(title))
        except Exception as exc:                       # network/JSON failure
            return RepertoriumResult(UNKNOWN, f"lookup failed: {exc}")
        if not candidates:
            return RepertoriumResult(UNKNOWN, "no Repertorium match")

        # See ingest.fuzzy_match: bare generic titles ("carmina", "versus", ...)
        # require real author corroboration, not just title similarity, or we
        # collapse many unrelated works onto one confident-looking match.
        m = best_match(
            title, author, candidates,
            title_of=lambda c: c[1], context_of=lambda c: c[2],
            threshold=self.match_threshold,
        )
        if m.candidate is None:
            reason = ("no confident match" if m.score < self.match_threshold
                      else "title too generic to trust without author match")
            best = max(candidates, key=lambda c: SequenceMatcher(
                None, _norm(title), _norm(c[1])).ratio())
            return RepertoriumResult(
                UNKNOWN, f"{reason} (best {m.score:.2f}: {best[1]!r})",
                match_score=m.score)

        wid, matched_title, _ = m.candidate
        langs = self.translations(wid)
        has_eng = any(any(en in l.lower() for en in _ENGLISH_NAMES) for l in langs)
        if has_eng:
            status, reason = TRANSLATED, "Repertorium lists an English translation"
        elif langs:
            status = UNTRANSLATED
            reason = f"translations listed but none English ({', '.join(langs)})"
        else:
            status = UNTRANSLATED
            reason = "no translations recorded in Repertorium"
        return RepertoriumResult(status, reason, werk_id=wid,
                                 matched_title=matched_title, match_score=m.score,
                                 languages=langs)
