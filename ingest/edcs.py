"""Connector for EDCS — Epigraphik-Datenbank Clauss/Slaby (edcs.hist.uzh.ch).

EDCS holds ~542,000 Latin inscriptions — overwhelmingly untranslated. Its search
UI is a JS app, but it is backed by a DataTables JSON API at /api/query, which we
call directly with requests (the browser was only needed to discover it).

A query returns many short inscriptions, so each inscription becomes one segment
(``_pre_segmented``) rather than being sentence-split. A fetch() ingests one
query's worth of inscriptions as a single Document.

Usage:
    from ingest.edcs import EDCSConnector
    meta, parts = EDCSConnector().fetch("Augustus", limit=50)
    meta, parts = EDCSConnector().fetch("province=Roma", limit=100)
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import requests

from .base import Connector, RawWork


class EDCSConnector(Connector):
    name = "edcs"
    API = "https://edcs.hist.uzh.ch/api/query"
    PAGE = 100  # rows per API call

    def __init__(self, timeout: float = 40.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "LatinReader-Research/1.0 (scholarly; jth156@case.edu)",
            "Referer": "https://edcs.hist.uzh.ch/de/search",
            "X-Requested-With": "XMLHttpRequest",
        })

    def fetch(self, query: str, limit: int = 100, **meta_overrides) -> RawWork:
        rows = self._query(query, limit)

        parts: List[Tuple[str, str]] = []
        for obj in rows:
            text = self._inscription_text(obj)
            if not text:
                continue
            parts.append((self._label(obj), text))

        meta = {
            "title": f"EDCS inscriptions: {query}",
            "author": None,
            "genre": "inscription",
            "language_stage": "classical",
            "source": f"EDCS ({query})",
            "license": "EDCS — edcs.hist.uzh.ch (Clauss/Slaby)",
            "has_existing_translation": False,
            "_pre_segmented": True,  # each inscription is one segment
        }
        meta.update(meta_overrides)
        return meta, parts

    # -- API -----------------------------------------------------------------

    def _query(self, query: str, limit: int) -> List[dict]:
        # Map the query to a filter field: "field=value" overrides; else free text.
        filt = {"searchtext1": query}
        if "=" in query:
            field, _, value = query.partition("=")
            filt = {field.strip(): value.strip()}

        rows: List[dict] = []
        start = 0
        while len(rows) < limit:
            page = min(self.PAGE, limit - len(rows))
            data = self._api_call(start, page, filt)
            batch = data.get("data", [])
            if not batch:
                break
            for entry in batch:
                obj = entry.get("obj") if isinstance(entry, dict) else None
                if obj:
                    rows.append(obj)
            total = data.get("recordsFiltered", 0)
            start += page
            if start >= total:
                break
        return rows[:limit]

    def _api_call(self, start: int, length: int, filt: dict) -> dict:
        # Minimal DataTables server-side params + the column data keys EDCS expects.
        cols = ["obj.edcs-id", "", "obj.inschriften", "obj.material",
                "obj.datierung", "obj.anzahl_bilder"]
        params = {"draw": "1", "start": str(start), "length": str(length),
                  "order[0][column]": "0", "order[0][dir]": "asc",
                  "search[value]": "", "search[regex]": "false"}
        for i, c in enumerate(cols):
            params[f"columns[{i}][data]"] = c
            params[f"columns[{i}][searchable]"] = "true"
            params[f"columns[{i}][orderable]"] = "true"
        params.update(filt)
        resp = self.session.get(self.API, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # -- field extraction ----------------------------------------------------

    @staticmethod
    def _inscription_text(obj: dict) -> Optional[str]:
        ins = obj.get("inschriften")
        raw = None
        if isinstance(ins, list) and ins:
            first = ins[0]
            raw = first[0] if isinstance(first, list) and first else first
        elif isinstance(ins, str):
            raw = ins
        if not isinstance(raw, str) or not raw.strip():
            return None
        # EDCS uses " / " for line breaks; normalize to single spaces for reading.
        return " ".join(raw.replace("/", " / ").split())

    @staticmethod
    def _label(obj: dict) -> str:
        edcs_id = obj.get("edcs-id", "EDCS-?")
        belege = obj.get("belege")
        cite = ""
        if isinstance(belege, list) and belege and isinstance(belege[0], list):
            cite = " ".join(str(x) for x in belege[0] if x)
        place = obj.get("provinz") or obj.get("ort") or ""
        bits = [edcs_id]
        if cite:
            bits.append(cite)
        if place:
            bits.append(place)
        return " · ".join(bits)
