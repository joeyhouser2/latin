"""Connector for DigilibLT — Digital Library of late-antique Latin Texts.

DigilibLT (digiliblt.uniupo.it) serves each work as TEI-XML at
    /teidocs/idno/<DLT id>/format/xml
so this connector fetches that XML and reuses the generic TEI parser. discover()
crawls an author page or the catalogue ("canone") for DLT ids.

Texts are © their editors, released by DigilibLT under CC BY-NC-ND — fine for
personal research; mind the terms before redistributing.

Usage:
    from ingest.digiliblt import DigilibLTConnector
    meta, parts = DigilibLTConnector().fetch("DLT000001")
    ids = DigilibLTConnector().discover("AUT000003")        # one author's works
    ids = DigilibLTConnector().discover("canone", limit=50) # whole catalogue
"""

from __future__ import annotations

from typing import List
import re

import requests

from .base import Connector, RawWork
from .tei import TEIConnector


class DigilibLTConnector(Connector):
    name = "digiliblt"
    BASE = "https://digiliblt.uniupo.it/teidocs"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly; jth156@case.edu)"}
        )
        self._tei = TEIConnector(timeout=timeout)

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        idno = self._normalize(identifier)
        url = f"{self.BASE}/idno/{idno}/format/xml"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()

        meta = {
            "source": f"DigilibLT ({idno})",
            "language_stage": "late_antique",
            "license": "CC BY-NC-ND (DigilibLT)",
            "has_existing_translation": False,
        }
        meta.update(meta_overrides)
        return self._tei.parse_xml(resp.content, label=idno, **meta)

    def discover(self, query: str, limit: int = 50) -> List[str]:
        """List DLT work-ids from an author page (AUT...) or the catalogue."""
        if re.fullmatch(r"AUT\d+", query.strip(), re.IGNORECASE):
            urls = [f"{self.BASE}/author/{query.strip().upper()}"]
        elif query.strip().lower() in ("canone", "all", "catalogue", "catalogo"):
            urls = self._canone_pages(limit)
        elif query.startswith("http"):
            urls = [query]
        else:
            raise ValueError(
                "discover() expects an author id (AUT000003), 'canone', or a URL"
            )

        ids: List[str] = []
        seen = set()
        for url in urls:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            for dlt in re.findall(r"/idno/(DLT\d+)", resp.text):
                if dlt not in seen:
                    seen.add(dlt)
                    ids.append(dlt)
                    if len(ids) >= limit:
                        return ids
        return ids

    # -- helpers -------------------------------------------------------------

    def _canone_pages(self, limit: int) -> List[str]:
        # The catalogue is paginated by ?l=<page size>&o=<offset>.
        page = 16
        return [f"{self.BASE}/canone?l={page}&o={o}"
                for o in range(0, max(limit, page) + page, page)]

    @staticmethod
    def _normalize(identifier: str) -> str:
        m = re.search(r"(DLT\d+)", identifier, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        raise ValueError(f"Not a DigilibLT id/URL: {identifier!r}")
