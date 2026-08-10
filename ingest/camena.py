"""Connector for CAMENA — Corpus Automatum Multiplex Electorum Neolatinitatis
Auctorum.

Neo-Latin texts of early-modern Germany (mostly 16th-17th c.): scholarly,
poetic, historical/political prose -- the CAMENA project (Heidelberg/Mannheim,
1999-2013) is now defunct as a live site, but its CC BY-SA XML was republished
on GitHub at nevenjovanovic/camena-neolatinlit, split into four collections:
``poemata`` (poetry), ``historicapolitica`` (history/politics), ``thesaurus``
(scholarly prose/dictionaries), ``cera`` (letters/orations).

Files are TEI.2 (older TEI Lite DTD, no namespace) in ISO-8859-1 -- parses
fine with the generic TEIConnector (its tag matching is namespace-agnostic),
as long as the raw bytes (not re-decoded text) are handed over so the
declared encoding is honored.

Neo-Latin is one of the largest almost-entirely-untranslated strata of Latin
literature -- most of this collection has no English rendering anywhere.

Usage:
    from ingest.camena import CAMENAConnector
    meta, parts = CAMENAConnector().fetch("poemata/Abel_carmina")
    ids = CAMENAConnector().discover("poemata", limit=50)
    ids = CAMENAConnector().discover("all", limit=50)   # across all 4 collections
"""

from __future__ import annotations

from typing import List

import requests

from .base import Connector, RawWork
from .tei import TEIConnector

COLLECTIONS = ("cera", "historicapolitica", "poemata", "thesaurus")


class CAMENAConnector(Connector):
    name = "camena"
    REPO = "nevenjovanovic/camena-neolatinlit"
    API = f"https://api.github.com/repos/{REPO}/contents"
    RAW = f"https://raw.githubusercontent.com/{REPO}/master"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly; jth156@case.edu)"}
        )
        self._tei = TEIConnector(timeout=timeout)

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        collection, stem = self._split(identifier)
        resp = self.session.get(f"{self.RAW}/{collection}/{stem}.xml",
                                timeout=self.timeout)
        resp.raise_for_status()

        meta = {
            "source": f"CAMENA ({collection}/{stem})",
            "language_stage": "early_modern",
            "license": "CC BY-SA (CAMENA, GitHub republish)",
            "has_existing_translation": False,
        }
        meta.update(meta_overrides)
        # resp.content (raw bytes): the file declares ISO-8859-1 in its XML
        # prolog, so ET needs the original bytes, not requests' guessed .text.
        return self._tei.parse_xml(resp.content, label=stem, **meta)

    def discover(self, query: str, limit: int = 200) -> List[str]:
        """List "<collection>/<stem>" identifiers. query is a collection name
        (cera/historicapolitica/poemata/thesaurus), 'all' for every collection,
        or "<collection>:<substring>" to filter by filename within one."""
        q = query.strip().lower()
        substr = None
        if ":" in q:
            q, substr = q.split(":", 1)

        cols = list(COLLECTIONS) if q in ("all", "") else [q]
        unknown = [c for c in cols if c not in COLLECTIONS]
        if unknown:
            raise ValueError(f"Unknown CAMENA collection(s) {unknown}; "
                             f"choose from {COLLECTIONS}")

        ids: List[str] = []
        for col in cols:
            resp = self.session.get(f"{self.API}/{col}", timeout=self.timeout)
            resp.raise_for_status()
            for e in resp.json():
                if e["type"] != "file" or not e["name"].endswith(".xml"):
                    continue
                stem = e["name"].removesuffix(".xml")
                if substr and substr not in stem.lower():
                    continue
                ids.append(f"{col}/{stem}")
                if len(ids) >= limit:
                    return ids
        return ids

    @staticmethod
    def _split(identifier: str) -> tuple:
        identifier = identifier.strip().removesuffix(".xml")
        if "/" not in identifier:
            raise ValueError(
                f"CAMENA identifier must be '<collection>/<stem>', got {identifier!r}"
            )
        collection, stem = identifier.split("/", 1)
        if collection not in COLLECTIONS:
            raise ValueError(f"Unknown CAMENA collection {collection!r}; "
                             f"choose from {COLLECTIONS}")
        return collection, stem
