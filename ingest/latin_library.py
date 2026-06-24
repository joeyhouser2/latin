"""Connector for The Latin Library (thelatinlibrary.com).

The site serves plain HTML with the text in <p> elements. We fetch a page, strip
navigation/markup, and return the body paragraphs as a single section. Stdlib
only (html.parser) so there is no extra dependency.

Usage:
    from ingest.latin_library import LatinLibraryConnector
    meta, parts = LatinLibraryConnector().fetch("https://www.thelatinlibrary.com/vegetius1.html")
"""

from __future__ import annotations

from typing import List, Tuple
from urllib.parse import urljoin, urlparse
import re

import requests

from .base import Connector, RawWork
from ._html import extract_paragraphs


# Paragraphs that are site chrome, not text, are dropped.
_CHROME = re.compile(
    r"(the latin library|the classics page|christian latin|medieval latin|"
    r"^\s*$)", re.IGNORECASE
)

# A leading run of 3+ bare numbers is the page's chapter-index navigation
# (e.g. "1 2 3 ... 33") sitting in front of the first sentence.
_LEADING_NUMS = re.compile(r"^(?:\s*\d+){3,}\s*")


def _clean_paragraph(p: str) -> str:
    """Strip the chapter-index number run that prefixes the body text."""
    return _LEADING_NUMS.sub("", p).strip()


def _is_noise(p: str) -> bool:
    """True for navigation/number-only paragraphs that aren't real text."""
    if not p:
        return True
    digits = sum(c.isdigit() for c in p)
    letters = sum(c.isalpha() for c in p)
    # Mostly digits, or too few letters to be a sentence.
    return letters < 15 or digits > letters


class LatinLibraryConnector(Connector):
    name = "latinlibrary"
    BASE = "https://www.thelatinlibrary.com/"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly research)"}
        )

    def fetch(self, url: str, **meta_overrides) -> RawWork:
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "latin-1"

        body = []
        for p in extract_paragraphs(resp.text):
            p = _clean_paragraph(p)
            if not _CHROME.search(p) and not _is_noise(p):
                body.append(p)
        title = self._guess_title(resp.text, url)

        meta = {
            "title": title,
            "source": f"The Latin Library ({url})",
            "license": "public domain (per thelatinlibrary.com)",
            "language_stage": "unknown",
            "has_existing_translation": False,
        }
        meta.update(meta_overrides)
        # One section for now; structural splitting (books/chapters) is future work.
        parts: List[Tuple[str, str]] = [("Text", "\n".join(body))]
        return meta, parts

    def discover(self, index_url: str, limit: int = 50) -> List[str]:
        """Crawl an author/index page and return the work-page URLs it links to.

        e.g. discover("https://www.thelatinlibrary.com/aug.html") -> Augustine's works.
        """
        resp = self.session.get(index_url, timeout=self.timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "latin-1"

        hrefs = re.findall(r'href\s*=\s*["\']([^"\']+)["\']', resp.text, re.IGNORECASE)
        out: List[str] = []
        seen = set()
        for href in hrefs:
            if not href.lower().endswith(".html"):
                continue
            full = urljoin(index_url, href)
            # same site, not a navigation/index page, not the page itself
            if urlparse(full).netloc != urlparse(self.BASE).netloc:
                continue
            if full in seen or full.rstrip("/") == index_url.rstrip("/"):
                continue
            if re.search(r"(index|classics|medieval|christian|misc|imperial|"
                         r"neo|ecclesiastic)\.html$", full, re.IGNORECASE):
                continue
            seen.add(full)
            out.append(full)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _guess_title(html: str, url: str) -> str:
        m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            t = re.sub(r"\s+", " ", m.group(1)).strip()
            if t:
                return t
        return url.rsplit("/", 1)[-1].replace(".html", "")
