"""Connector for Musa Medievalis (musamedievalis.it) — medieval Latin poetry,
c. 650-1250; this connector targets the era-filtered (6th-10th c.) slice.

The site is a JavaScript app, so author/work discovery is done ONCE offline by
scripts/crawl_musamedievalis.py, which writes a work-code manifest to
data/musamedievalis_catalog.json. The text pages themselves ARE server-rendered,
so fetch() uses plain requests: GET /testo/testo/codice/<CODE> returns the poem
with each verse line in a <p class="c_v"> element.

Verse is segmented one line per segment (meta["_verse"]) so the side-by-side
reader and scansion align line-for-line. translation_status defaults to
"unknown"; note these are POEMS — machine translation (NLLB) renders verse
poorly, so prefer the LLM verse-stylizer path over the stock translator.

Usage:
    from ingest.musamedievalis import MusaMedievalisConnector
    c = MusaMedievalisConnector()
    meta, parts = c.fetch("ABBO_FLO|acro|001")
    codes = c.discover("c9")        # 9th-century works; also "all", author substr
"""

from __future__ import annotations

from html import unescape
from typing import List
import json
import os
import re
import urllib.parse

import requests

from .base import Connector, RawWork
from .translation_status import UNKNOWN

_CATALOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "musamedievalis_catalog.json")


class MusaMedievalisConnector(Connector):
    name = "musamedievalis"
    TEXT = "https://www.musamedievalis.it/testo/testo/codice/{code}"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly; jth156@case.edu)"}
        )
        self._catalog = None  # lazy {code: entry}

    # -- fetch ---------------------------------------------------------------

    def fetch(self, code: str, **meta_overrides) -> RawWork:
        code = code.strip()
        first = self._get(code)
        # A work is split into sections (praefatio, 1, 2, ...) listed in a
        # <select name="codice"> dropdown; fetch them all so the whole poem lands.
        sections = self._section_codes(first, code)
        cache = {code: first}
        parts = []
        for scode, label in sections:
            html = cache.get(scode) or self._get(scode)
            lines = self._verse_lines(html)
            if lines:
                parts.append((label, "\n".join(lines)))
        if not parts:
            parts = [("carmen", "\n".join(self._verse_lines(first)))]

        entry = self._lookup(code)
        century = entry.get("century")
        meta = {
            "title": entry.get("title") or code,
            "author": entry.get("author"),
            "century": century,
            "genre": "poetry",
            "language": "la",
            "language_stage": "late_antique" if (century and century <= 6) else "medieval",
            "source": f"Musa Medievalis ({code})",
            "license": "Musa Medievalis (MQDQ Galaxy, Ca' Foscari); check per-text terms",
            "has_existing_translation": False,
            "translation_status": UNKNOWN,
            "_verse": True,
        }
        meta.update(meta_overrides)
        return meta, parts

    # -- discover ------------------------------------------------------------

    def discover(self, query: str = "all", limit: int = 1000) -> List[str]:
        """Return work codes from the manifest. query: 'all', a century filter
        like 'c9'/'c7', or an author/title substring (case-insensitive)."""
        cat = self._load_catalog()
        q = query.strip().lower()
        if q in ("all", ""):
            out = list(cat)
        elif re.fullmatch(r"c\d{1,2}", q):
            cent = int(q[1:])
            out = [c for c, e in cat.items() if e.get("century") == cent]
        else:
            out = [c for c, e in cat.items()
                   if q in (e.get("author", "") + " " + e.get("title", "")).lower()]
        return out[:limit]

    def catalogue(self) -> List[dict]:
        """The full manifest (list of {code, author, dates, century, title})."""
        return list(self._load_catalog().values())

    # -- helpers -------------------------------------------------------------

    def _get(self, code: str) -> str:
        resp = self.session.get(self.TEXT.format(code=urllib.parse.quote(code, safe="")),
                                timeout=self.timeout)
        resp.raise_for_status()
        resp.encoding = resp.encoding or "utf-8"
        return resp.text

    def _section_codes(self, html: str, code: str):
        """All (section_code, label) of a work, from the <select name="codice">
        dropdown; falls back to just the given code if there's no dropdown."""
        sel = re.search(r'<select[^>]*name="codice".*?</select>', html, re.S)
        if not sel:
            return [(code, "carmen")]
        out = []
        for val, txt in re.findall(r'<option[^>]*value="([^"]+)"[^>]*>(.*?)</option>',
                                   sel.group(0), re.S):
            label = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", txt))).strip()
            label = label if not label.isdigit() else f"pars {label}"
            out.append((unescape(val).strip(), label or "carmen"))
        return out or [(code, "carmen")]

    def _verse_lines(self, html: str) -> List[str]:
        lines: List[str] = []
        for raw in re.findall(r'<p class="c_v">(.*?)</p>', html, re.S):
            t = unescape(re.sub(r"<[^>]+>", " ", raw))
            t = re.sub(r"\s+", " ", t).strip()
            if not t or re.fullmatch(r"\d+", t):   # drop blanks + line-number markers
                continue
            lines.append(t)
        return lines

    def _load_catalog(self) -> dict:
        if self._catalog is None:
            try:
                with open(_CATALOG, encoding="utf-8") as f:
                    self._catalog = {e["code"]: e for e in json.load(f)}
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Musa Medievalis manifest missing ({_CATALOG}); "
                    "run scripts/crawl_musamedievalis.py first.")
        return self._catalog

    def _lookup(self, code: str) -> dict:
        try:
            return self._load_catalog().get(code, {})
        except FileNotFoundError:
            return {}
