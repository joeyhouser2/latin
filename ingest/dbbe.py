"""Connector for DBBE — Database of Byzantine Book Epigrams (Ghent University).

~13,000 Greek verse epigrams from medieval manuscripts (7th-15th c.) -- metrical
paratexts: poems in and about the books that carry them. Exactly the Byzantine
poetry gap none of our other Greek connectors cover (First1KGreek/PTA/PG Corpus
are patristic prose).

NO EXPLICIT LICENSE is published for the transcriptions (checked the about/help
pages; only a citation convention is stated: reference an occurrence by its id
and the date consulted). Treat as personal-research use only, not for
republishing -- this connector stores the consultation date and DBBE occurrence
id in ``source`` precisely so that citation is always possible. Don't bulk-export
this data elsewhere.

Access: the JSON API (``/occurrences/search_api``) needs no auth and gives
paginated metadata (Elasticsearch-style ``search_after`` cursor: pass the prior
page's last ``sort`` value back as ``search_after``) but not verse text; the
per-occurrence detail JSON endpoint requires login, but the plain HTML page at
``/occurrences/<id>`` renders full verse text for anonymous visitors (a
``<table class="... verses">`` of ``<tr><td class="verse">line</td></tr>``).
So: discover() via the JSON API, fetch() by scraping the HTML page.

Verse, like Musa Medievalis: do NOT bulk-MT with NLLB (verse=mush per
[[translation-models-plan]]); ingest for reading/search, translate via the LLM
verse-stylizer path instead.

Usage:
    from ingest.dbbe import DBBEConnector
    meta, parts = DBBEConnector().fetch("17276")
    ids = DBBEConnector().discover("all", limit=100)
"""

from __future__ import annotations

import datetime
import re
import time
from typing import List, Optional

import requests

from .base import Connector, RawWork

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0 Safari/537.36"}
# search_api 400s without these (it wants to look like a same-site XHR call)
# -- but sending a Referer on the *occurrence page* fetch instead trips its
# auth check and redirects to login, so this is passed only to that one call.
_API_HEADERS = {**_HEADERS, "Accept": "application/json",
                "Referer": "https://www.dbbe.ugent.be/occurrences/search"}
_VERSE_ROW = re.compile(
    r'<td class="line-number"[^>]*>.*?</td>\s*'
    r'<td class="verse">(.*?)</td>', re.S)


class DBBEConnector(Connector):
    name = "dbbe"
    BASE = "https://www.dbbe.ugent.be"
    SEARCH_API = BASE + "/occurrences/search_api"

    def __init__(self, timeout: float = 30.0, retries: int = 4,
                page_sleep: float = 0.5):
        self.timeout = timeout
        self.retries = retries
        self.page_sleep = page_sleep
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)

    def _get(self, url: str, **kwargs) -> requests.Response:
        """GET with retry/backoff -- DBBE's server drops connections under any
        sustained request rate, even a polite one, so a single blip shouldn't
        kill a multi-hour bulk pull."""
        last_exc = None
        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, timeout=self.timeout, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise last_exc

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        occ_id = str(identifier).strip()
        resp = self._get(f"{self.BASE}/occurrences/{occ_id}")
        html = resp.text

        lines = []
        for cell in _VERSE_ROW.findall(html):
            line = re.sub(r"<[^>]+>", "", cell)          # drop <a>/<i> wrappers
            line = line.replace("&nbsp;", " ").strip()
            if line:
                lines.append(line)
        text = "\n".join(lines)

        # <title> is just the generic site title on this page, not per-occurrence
        # -- use the incipit (first verse line) instead, falling back to the id.
        page_title = self._title(html)
        if page_title and "database of byzantine" in page_title.lower():
            page_title = None
        title = page_title or (lines[0][:80] if lines else None) or f"DBBE occurrence {occ_id}"
        today = datetime.date.today().isoformat()

        meta = {
            "title": title,
            "language": "grc",
            "language_stage": "late_antique" if _is_early(html) else "medieval",
            "genre": "poetry",
            "source": f"DBBE ({occ_id}, consulted {today})",
            "license": "no explicit license published -- personal research use, cite by id+date",
            "has_existing_translation": False,
            "translation_status": "unknown",  # not checked against translation-status sources
            "_verse": True,
        }
        meta.update(meta_overrides)
        return meta, [("Text", text)]

    def discover(self, query: str, limit: int = 200) -> List[str]:
        """Page through the search API with plain ``page=N`` pagination.

        (The response records carry an ``_search_after``/``sort`` field that
        looks like an Elasticsearch cursor, but the server ignores that as a
        request param entirely -- repeated identical pages resulted. Plain
        integer ``page=`` is what the frontend actually uses.)

        query is currently ignored (no server-side text filter wired up) --
        'all' returns occurrences in the database's own order, paging until
        `limit` is reached or the results run dry.

        The backend appears to be Elasticsearch with the classic ~10,000-result
        deep-pagination ceiling (25/page caps out around page 400): beyond that
        it 500s rather than paginating further. Caught here and treated as the
        natural end of what's reachable this way, not a crash -- so a `limit`
        above ~10,000 silently returns fewer ids than asked for."""
        ids: List[str] = []
        page = 1
        while len(ids) < limit:
            if page > 1:
                time.sleep(self.page_sleep)   # be polite -- rapid-fire paging
            try:
                resp = self._get(self.SEARCH_API, params={"page": page}, headers=_API_HEADERS)
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 500:
                    break   # hit the deep-pagination ceiling
                raise
            data = resp.json().get("data", [])
            if not data:
                break
            for rec in data:
                ids.append(str(rec["id"]))
                if len(ids) >= limit:
                    return ids
            page += 1
        return ids

    # Base params the frontend always sends alongside a date filter (found by
    # watching the real network request a browser makes when submitting the
    # search form -- plain GET query params like `date_floor_year=` are
    # silently ignored; it has to be this exact bracketed `filters[...]`
    # shape). `date_search_type=overlap` is what makes it a real range filter
    # -- the default `exact` mode returns 0 for any non-trivial range.
    _DATE_BASE_PARAMS = {
        "filters[text_mode]": "greek", "filters[comment_mode]": "latin",
        "filters[date_search_type]": "overlap", "filters[text_fields]": "text",
        "filters[text_combination]": "all", "filters[metre_op]": "or",
        "filters[genre_op]": "or", "filters[subject_op]": "or",
        "filters[manuscript_content_op]": "or", "filters[acknowledgement_op]": "or",
        "filters[exactly_dated]": "false",
    }

    def _date_range_count(self, year_from: int, year_to: int) -> int:
        params = {**self._DATE_BASE_PARAMS, "page": 1, "limit": 1,
                  "filters[date][from]": year_from, "filters[date][to]": year_to}
        resp = self._get(self.SEARCH_API, params=params, headers=_API_HEADERS)
        return resp.json().get("count", 0)

    def _page_date_range(self, year_from: int, year_to: int) -> List[str]:
        ids: List[str] = []
        page = 1
        while True:
            if page > 1:
                time.sleep(self.page_sleep)
            params = {**self._DATE_BASE_PARAMS, "page": page, "limit": 100,
                      "filters[date][from]": year_from, "filters[date][to]": year_to}
            resp = self._get(self.SEARCH_API, params=params, headers=_API_HEADERS)
            data = resp.json().get("data", [])
            if not data:
                break
            ids.extend(str(rec["id"]) for rec in data)
            page += 1
        return ids

    def discover_all(self, year_from: int = 1, year_to: int = 1600,
                     ceiling: int = 9000, min_span: int = 1) -> List[str]:
        """Every occurrence id, working around the ~10,000-result deep-
        pagination ceiling by bisecting the date range: a query whose
        ``overlap``-mode count is under ``ceiling`` gets paged directly,
        otherwise the range is split in half and each half is queued the
        same way. Adjacent ranges can both match the same boundary/wide-dated
        occurrence, so results are deduped before returning.

        Bounded, iterative (a worklist, not recursion): ``overlap`` mode means
        a widely/uncertainly-dated occurrence (e.g. "12th century") can match
        *every* narrow sub-range that touches it, so the count doesn't
        necessarily shrink as a range narrows -- bisection isn't guaranteed to
        converge. Once a span hits ``min_span`` (a single year) it's paged
        as-is regardless of count, accepting best-effort completeness there
        rather than looping forever.

        This can issue a lot of requests for a wide, busy span -- call it once
        and cache/persist the result rather than re-running per session."""
        seen: dict = {}   # id -> True, for stable dedup while preserving order
        work = [(year_from, year_to)]
        while work:
            a, b = work.pop()
            count = self._date_range_count(a, b)
            if count == 0:
                continue
            if count <= ceiling or (b - a) <= min_span:
                for occ_id in self._page_date_range(a, b):
                    seen.setdefault(occ_id, True)
                continue
            mid = (a + b) // 2
            work.append((a, mid))
            work.append((mid, b))
        return list(seen)

    @staticmethod
    def _title(html: str) -> Optional[str]:
        m = re.search(r"<title>(.*?)</title>", html, re.S)
        if not m:
            return None
        t = re.sub(r"\s+", " ", m.group(1)).strip()
        return t.split(" | ")[0].strip() or None


def _is_early(html: str) -> bool:
    """Rough century check from the date range shown on the page, so 7th-9th c.
    epigrams land as late_antique rather than medieval. Best-effort only."""
    m = re.search(r'"date_ceiling_year"\s*:\s*(\d+)', html)
    return bool(m and int(m.group(1)) < 900)
