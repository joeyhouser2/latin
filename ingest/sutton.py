"""Connector for Dana Sutton's *Analytic Bibliography of On-line Neo-Latin Texts*.

Hosted by the Philological Museum (Univ. of Birmingham). It is a finding-aid,
not a text host: ~40 alphabetical pages of entries, each

    AUTHOR Filelfo, Francesco (1398 - 1481)
    TITLE  Breviores elegantooresque epistolae
    URL    <link to a digital facsimile>
    SITE   MDZ | Gallica | Google Books | HAB | Internet Archive | ...
    SUBJECT Epistolography
    NOTES  Dpr of the 1501 Leipzig edition

so this connector **catalogues**. ``catalog()`` returns decidable records and
flags the ones whose text can actually be pulled: only Internet Archive links
have a scriptable OCR derivative. Everything else (BSB/MDZ, Gallica, HAB,
Google Books) is page images or browser-gated, and ``fetch()`` says so rather
than ingesting a viewer shell.

``fetch("ia:<id>")`` or a Sutton record's archive.org URL is delegated to the
``treatises`` connector, which already handles archive.org's uneven text-file
naming and chunks long OCR into numbered sections.

Usage:
    from ingest.sutton import SuttonConnector
    s = SuttonConnector()
    for r in s.catalog("Filelfo"):            # author, title or subject match
        print(r["author"], r["title"], r["site"], r["fetchable"])
    s.catalog("epistolography", fetchable_only=True)
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

from .base import Connector, RawWork
from .treatises import TreatisesConnector

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0 Safari/537.36"}

BASE = "https://philological.cal.bham.ac.uk/bibliography/"

_FIELD = re.compile(r"\b(AUTHOR|TITLE|URL|SITE|SUBJECT|NOTES)\b")
_IA_URL = re.compile(r"archive\.org/(?:details|stream|download)/([^/?#\s\"']+)")


class SuttonUnfetchable(RuntimeError):
    """Raised by fetch(): the entry links to page images, not scriptable text."""


class SuttonConnector(Connector):
    name = "sutton"

    def __init__(self, timeout: float = 90.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self._entries: Optional[List[Dict[str, Any]]] = None
        self._treatises = TreatisesConnector(timeout=timeout)

    # -- index ---------------------------------------------------------------

    def pages(self) -> List[str]:
        """The alphabetical page filenames linked from the index."""
        html = self.session.get(BASE, timeout=self.timeout).text
        hrefs = re.findall(r'href="([^"]+\.html?)"', html, re.I)
        out: List[str] = []
        for h in hrefs:
            if h.startswith("http") or h.lower().startswith("index"):
                continue
            if h not in out:
                out.append(h)
        return out

    def entries(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """Every entry in the bibliography (downloaded once, then cached)."""
        if self._entries is None or refresh:
            out: List[Dict[str, Any]] = []
            for page in self.pages():
                try:
                    resp = self.session.get(urljoin(BASE, page), timeout=self.timeout)
                    resp.raise_for_status()
                except Exception:                              # noqa: BLE001
                    continue            # one dead page shouldn't lose the rest
                out.extend(self.parse_page(resp.text, page))
            self._entries = out
        return self._entries

    @staticmethod
    def parse_page(html: str, page: str = "") -> List[Dict[str, Any]]:
        out = []
        for block in re.findall(r"<p\b[^>]*>(.*?)</p>", html, re.S | re.I):
            if "AUTHOR" not in block:
                continue
            rec = SuttonConnector._parse_block(block)
            if rec and rec.get("title"):
                rec["page"] = page
                out.append(rec)
        return out

    @staticmethod
    def _parse_block(block: str) -> Dict[str, Any]:
        urls = [unescape(u) for u in re.findall(r'href="(https?://[^"]+)"', block, re.I)]
        text = re.sub(r"<br\s*/?>", "\n", block, flags=re.I)
        text = unescape(re.sub(r"<[^>]+>", "", text))
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        # Split on field keywords; the bibliography wraps lines freely.
        pieces = _FIELD.split(text)
        fields: Dict[str, str] = {}
        for key, val in zip(pieces[1::2], pieces[2::2]):
            fields.setdefault(key, re.sub(r"\s+", " ", val).strip())
        author = fields.get("AUTHOR", "")
        m = re.search(r"\((?:fl\.\s*)?(?:c\.\s*)?(\d{3,4})\s*[-–]\s*(\d{3,4})?\)", author)
        born = int(m.group(1)) if m else None
        died = int(m.group(2)) if m and m.group(2) else None
        ia = next((_IA_URL.search(u).group(1) for u in urls if _IA_URL.search(u)), None)
        return {
            "catalogue": "sutton",
            "author": re.sub(r"\s*\(.*$", "", author).strip() or None,
            "author_dates": (born, died),
            "title": fields.get("TITLE"),
            "site": fields.get("SITE"),
            "subject": fields.get("SUBJECT"),
            "notes": fields.get("NOTES"),
            "note": fields.get("NOTES"),
            "urls": list(dict.fromkeys(urls)),
            "url": urls[0] if urls else None,
            "publisher": fields.get("SITE"),
            "year": None,
            "lang_code": "la",
            "genre": fields.get("SUBJECT"),
            "ia_id": ia,
            "identifier": f"ia:{ia}" if ia else (urls[0] if urls else ""),
            "century": ((born + 25) // 100 + 1) if born else None,
            "fetchable": bool(ia),
        }

    # -- connector API -------------------------------------------------------

    def catalog(self, query: str = "", limit: int = 50,
                fetchable_only: bool = False) -> List[Dict[str, Any]]:
        """Entries whose author, title, subject or notes contain every word of ``query``.

        The word ``fetchable`` (or ``text``) in the query restricts to entries
        with scriptable OCR text, i.e. archive.org links.
        """
        words = [w for w in re.split(r"\s+", query.lower().strip()) if w]
        if "fetchable" in words:
            words.remove("fetchable")
            fetchable_only = True
        out = []
        for e in self.entries():
            if fetchable_only and not e["fetchable"]:
                continue
            hay = " ".join(str(e.get(k) or "") for k in
                           ("author", "title", "subject", "notes", "site")).lower()
            if all(w in hay for w in words):
                out.append(e)
                if len(out) >= limit:
                    break
        return out

    def discover(self, query: str, limit: int = 50) -> List[str]:
        """Archive.org identifiers (``ia:...``) for entries that can be fetched."""
        return [e["identifier"] for e in
                self.catalog(query, limit=limit, fetchable_only=True)]

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        ident = identifier.strip()
        m = _IA_URL.search(ident)
        if m:
            ident = f"ia:{m.group(1)}"
        if ident.startswith("ia:"):
            return self._treatises.fetch(ident, **meta_overrides)
        raise SuttonUnfetchable(
            f"{identifier!r} links to a page-image viewer (BSB/Gallica/HAB/Google "
            f"Books), not scriptable text. Download or OCR it manually and use "
            f"the 'file' connector, or search archive.org for another copy via "
            f"the 'treatises' connector."
        )
