"""Connector for Corpus Thomisticum (corpusthomisticum.org) — the complete works
of Thomas Aquinas, edited by Enrique Alarcón.

Each work (or work-part) is an HTML page whose text sits in a single <p>, peppered
with [NNNNN] Busa reference numbers that we strip. fetch(id-or-url) loads one page;
discover(index-url) lists the work pages linked from an index (e.g. iopera.html).

Note: long works (e.g. the Summa) are split across several pages on the site;
fetch() ingests one page. Point discover() at the relevant index to get them all.

Usage:
    from ingest.corpus_thomisticum import CorpusThomisticumConnector
    meta, parts = CorpusThomisticumConnector().fetch("sth0000")
"""

from __future__ import annotations

from typing import List
from urllib.parse import urljoin, urlparse
import re

import requests

from .base import Connector, RawWork
from ._html import extract_paragraphs


_BUSA_REF = re.compile(r"\[\d+\]")             # inline [28231] reference markers
# Work-text pages look like a 2-4 letter code + digits, e.g. sth0000, scg1001.
_WORK_PAGE = re.compile(r"^[a-z]{2,4}\d+[a-z]?\.html$", re.IGNORECASE)


class CorpusThomisticumConnector(Connector):
    name = "corpusthomisticum"
    BASE = "https://www.corpusthomisticum.org/"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly; jth156@case.edu)"}
        )

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        url = self._to_url(identifier)
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"

        body = []
        for p in extract_paragraphs(resp.text):
            p = _BUSA_REF.sub("", p).strip()
            if len(p) >= 30:
                body.append(p)

        meta = {
            "title": self._title(resp.text, url),
            "author": "Thomas de Aquino",
            "century": 13,
            "genre": "scholastic",
            "language_stage": "medieval",
            "source": f"Corpus Thomisticum ({url})",
            "license": "Corpus Thomisticum (Fundación Tomás de Aquino); research use",
            "has_existing_translation": False,
        }
        meta.update(meta_overrides)
        return meta, [("Text", "\n".join(body))]

    def discover(self, index_url: str, limit: int = 50) -> List[str]:
        """List work-text page URLs linked from an index page (e.g. iopera.html)."""
        url = self._to_url(index_url) if not index_url.startswith("http") else index_url
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"

        out: List[str] = []
        seen = set()
        for href in re.findall(r'href\s*=\s*["\']?([^"\' >]+)', resp.text, re.I):
            page = href.rsplit("/", 1)[-1]
            if not _WORK_PAGE.match(page):
                continue
            full = urljoin(self.BASE, href)
            if urlparse(full).netloc != urlparse(self.BASE).netloc or full in seen:
                continue
            seen.add(full)
            out.append(full)
            if len(out) >= limit:
                break
        return out

    # -- helpers -------------------------------------------------------------

    def _to_url(self, identifier: str) -> str:
        if identifier.startswith("http"):
            return identifier
        ident = identifier if identifier.endswith(".html") else identifier + ".html"
        return urljoin(self.BASE, ident)

    @staticmethod
    def _title(html: str, url: str) -> str:
        m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            t = re.sub(r"\s+", " ", m.group(1)).strip()
            if t:
                return t
        return url.rsplit("/", 1)[-1].replace(".html", "")
