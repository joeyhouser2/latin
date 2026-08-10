"""Connector for CroALa — Croatiae auctores Latini.

Latin texts by/about people of Croatian origin, medieval through the 20th c.
The project (croala.ffzg.unizg.hr) publishes its TEI-XML source republished
on GitHub at nevenjovanovic/croatiae-auctores-latini-textus, ~575 flat files
in ``txts/<slug>.xml``, mostly CC0/public-domain-dedicated per work. TEI P5,
namespaced -- parses directly with the generic TEIConnector.

Coverage skews toward exactly the kind of material with no English
translation: minor/regional authors, occasional verse, epistolography.

Usage:
    from ingest.croala import CroALaConnector
    meta, parts = CroALaConnector().fetch("adam-parisius-vaticanum-officium-1059")
    ids = CroALaConnector().discover("all", limit=50)
"""

from __future__ import annotations

from typing import List

import requests

from .base import Connector, RawWork
from .tei import TEIConnector


class CroALaConnector(Connector):
    name = "croala"
    REPO = "nevenjovanovic/croatiae-auctores-latini-textus"
    API = f"https://api.github.com/repos/{REPO}/contents/txts"
    RAW = f"https://raw.githubusercontent.com/{REPO}/master/txts"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly; jth156@case.edu)"}
        )
        self._tei = TEIConnector(timeout=timeout)

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        slug = identifier.strip().removesuffix(".xml")
        resp = self.session.get(f"{self.RAW}/{slug}.xml", timeout=self.timeout)
        resp.raise_for_status()

        meta = {
            "source": f"CroALa ({slug})",
            "language_stage": "unknown",  # spans medieval to C20; per-work century unset
            "license": "CC0/CC-BY (CroALa, per-work -- check teiHeader)",
            "has_existing_translation": False,
        }
        meta.update(meta_overrides)
        return self._tei.parse_xml(resp.content, label=slug, **meta)

    def discover(self, query: str, limit: int = 200) -> List[str]:
        """List CroALa slugs. query='all' lists the whole txts/ dir; anything
        else is treated as a substring filter on the filename (e.g. an author
        surname slug like 'marulic')."""
        resp = self.session.get(self.API, timeout=self.timeout)
        resp.raise_for_status()
        names = [e["name"].removesuffix(".xml") for e in resp.json()
                 if e["type"] == "file" and e["name"].endswith(".xml")]
        q = query.strip().lower()
        if q not in ("all", ""):
            names = [n for n in names if q in n.lower()]
        return names[:limit]
