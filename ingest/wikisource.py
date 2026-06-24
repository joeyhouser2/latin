"""Connector for Latin Wikisource (la.wikisource.org) via the MediaWiki API.

fetch(page_title) renders a page to HTML and extracts the prose. discover(query)
finds page titles by full-text search, or lists a category's members when the
query starts with "Categoria:". License is CC-BY-SA (Wikisource).

Usage:
    from ingest.wikisource import WikisourceConnector
    c = WikisourceConnector()
    meta, parts = c.fetch("Confessiones (ed. Migne)/1")
    titles = c.discover("Augustinus Confessiones", limit=10)
"""

from __future__ import annotations

from typing import List
import re

import requests

from .base import Connector, RawWork
from ._html import extract_paragraphs


API = "https://la.wikisource.org/w/api.php"

# MediaWiki UI cruft that survives into rendered text.
_SKIP_CLASSES = {
    "mw-editsection", "reference", "noprint", "mw-cite-backlink",
    "navigation-not-searchable", "ws-noexport", "mw-references-wrap",
}
_EDIT_MARK = re.compile(r"\[\s*(recensere|recense|edit)\s*\]", re.IGNORECASE)
_BRACKET_NUM = re.compile(r"\[\s*\d+\s*\]")  # footnote markers like [1]


class WikisourceConnector(Connector):
    name = "wikisource"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly research)"}
        )

    def fetch(self, page_title: str, **meta_overrides) -> RawWork:
        data = self._api(
            action="parse", page=page_title,
            prop="text|displaytitle", formatversion="2", redirects="1",
        )
        if "parse" not in data:
            raise ValueError(
                f"Wikisource page not found: {page_title!r} "
                f"({data.get('error', {}).get('code', 'unknown error')})"
            )
        html = data["parse"]["text"]

        body = []
        for p in extract_paragraphs(html, skip_classes=_SKIP_CLASSES):
            p = _BRACKET_NUM.sub("", _EDIT_MARK.sub("", p)).strip()
            if len(p) >= 20 and sum(c.isalpha() for c in p) >= 15:
                body.append(p)

        title = self._clean_title(data["parse"].get("displaytitle", page_title))
        meta = {
            "title": title,
            "source": f"Latin Wikisource ({page_title})",
            "license": "CC BY-SA",
            "language_stage": "unknown",
            "has_existing_translation": False,
        }
        meta.update(meta_overrides)
        return meta, [("Text", "\n".join(body))]

    def discover(self, query: str, limit: int = 50) -> List[str]:
        if query.startswith(("Categoria:", "Category:")):
            return self._category_members(query, limit)
        return self._search(query, limit)

    # -- internals -----------------------------------------------------------

    def _search(self, query: str, limit: int) -> List[str]:
        data = self._api(
            action="query", list="search", srsearch=query,
            srlimit=str(min(limit, 500)), srnamespace="0",
        )
        return [h["title"] for h in data.get("query", {}).get("search", [])]

    def _category_members(self, category: str, limit: int) -> List[str]:
        data = self._api(
            action="query", list="categorymembers", cmtitle=category,
            cmlimit=str(min(limit, 500)), cmtype="page",
        )
        return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]

    def _api(self, **params) -> dict:
        params["format"] = "json"
        resp = self.session.get(API, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _clean_title(displaytitle: str) -> str:
        return re.sub(r"<[^>]+>", "", displaytitle).strip()
