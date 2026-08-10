"""Connector for plain-text items on the Internet Archive.

Generic -- works for any archive.org item with an OCR'd ``_djvu.txt`` derivative,
not just one collection. Built for Analecta Hymnica Medii Aevi (55 volumes,
Dreves/Blume/Bannister 1886-1922, public domain: essentially THE medieval
Latin liturgical-poetry corpus, almost entirely untranslated), but reusable
for any other pre-1929 scanned Latin/Greek text -- same pattern used manually
earlier this session for a Salmasius volume.

``fetch()`` hits ``/download/<id>/<id>_djvu.txt``, which 302-redirects straight
to the raw OCR text (unlike ``/stream/<id>_djvu.txt``, which serves an HTML
wrapper page requiring separate extraction). ``discover()`` uses the public
Advanced Search API (title full-text query, no auth needed).

Caveat: 19th/early-20th-c. OCR on Fraktur/mixed-script pages is noisy (see
[[ocr-long-s-correction]] equivalent tooling: scripts/ocr_fix*.py, garble
detection) -- expect some garbled segments, worse than modern TEI sources.

Usage:
    from ingest.archive_org import ArchiveOrgConnector
    meta, parts = ArchiveOrgConnector().fetch("analectahymnicam20drev")
    ids = ArchiveOrgConnector().discover('title:"analecta hymnica"', limit=100)
"""

from __future__ import annotations

from typing import List

import requests

from .base import Connector, RawWork

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0 Safari/537.36"}


class ArchiveOrgConnector(Connector):
    name = "archiveorg"
    SEARCH = "https://archive.org/advancedsearch.php"

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        item = identifier.strip()
        url = f"https://archive.org/download/{item}/{item}_djvu.txt"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        text = resp.text

        meta = {
            "title": item,
            "source": f"Internet Archive ({item})",
            "language_stage": "unknown",
            "license": "Public domain (pre-1929 scan, raw OCR text)",
            "has_existing_translation": False,
        }
        meta.update(meta_overrides)
        return meta, [("Text", text)]

    def discover(self, query: str, limit: int = 100) -> List[str]:
        """Search archive.org's Advanced Search API and return item identifiers.

        query is either a raw AS query string (e.g. 'title:"analecta hymnica"')
        or, if it has no ':', treated as a plain title-text search."""
        q = query.strip()
        if ":" not in q:
            q = f'title:"{q}"'
        resp = self.session.get(self.SEARCH, params={
            "q": q, "fl[]": "identifier", "rows": limit, "output": "json",
        }, timeout=self.timeout)
        resp.raise_for_status()
        docs = resp.json().get("response", {}).get("docs", [])
        seen, ids = set(), []
        for d in docs:
            ident = d.get("identifier")
            if ident and ident not in seen:
                seen.add(ident)
                ids.append(ident)
                if len(ids) >= limit:
                    break
        return ids
