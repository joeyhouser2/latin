"""Connector for openMGH — the open-access TEI edition texts of the Monumenta
Germaniae Historica (the core edited corpus for European history c. 400-1500:
Merovingian, Carolingian, Lombard, and late-antique sources).

openMGH ships each MGH volume as a single TEI-XML file inside a per-volume zip:
    https://data.mgh.de/openmgh/<bsbNNNNNNNN>.zip   ->  <bsbNNNNNNNN>.xml
identified by a Bavarian State Library (BSB) number. ~153 volumes are available.
fetch() downloads the zip, extracts the XML, and reuses the generic TEI parser.
discover() scrapes the editions index for BSB ids, optionally filtered by a
series/title substring (e.g. "Merov", "Urkunden", "Auct. ant.").

Released by the MGH + BSB under CC BY 4.0 (attribute www.mgh.de/mgh-digital/openmgh).

NOTE ON TRANSLATIONS: MGH is a critical-edition series, NOT an inherently
untranslated corpus the way DigilibLT is. Many MGH texts (Gregory of Tours,
Einhard, Bede-adjacent works) have well-known English translations, while the
Diplomata/Formulae are almost entirely untranslated. So translation status is
inferred per work (see ingest.translation_status), defaulting to ``unknown``
rather than blanket-claiming untranslated.

Usage:
    from ingest.mgh import MGHConnector
    meta, parts = MGHConnector().fetch("bsb00000785")        # Salvian
    ids = MGHConnector().discover("all", limit=200)          # every volume
    ids = MGHConnector().discover("Urkunden")                # charters (Diplomata)
"""

from __future__ import annotations

from typing import List, Tuple, Dict
import io
import re
import zipfile

import requests

from .base import Connector, RawWork
from .tei import TEIConnector
from .translation_status import infer_translation_status


class MGHConnector(Connector):
    name = "mgh"
    DATA = "https://data.mgh.de/openmgh"
    EDITIONS_URL = "https://www.mgh.de/en/digital-mgh/openmgh/mgh-editions-in-openmgh"

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly; jth156@case.edu)"}
        )
        self._tei = TEIConnector(timeout=timeout)

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        bsb = self._normalize(identifier)
        url = f"{self.DATA}/{bsb}.zip"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()

        xml = self._extract_xml(resp.content, bsb)
        meta, parts = self._tei.parse_xml(xml, label=bsb)

        # Infer translation status from the work title (carries the MGH series,
        # e.g. "(MGH Diplomata ...)"), unless the caller overrides it.
        status = infer_translation_status(title=meta.get("title"))
        meta.update({
            "source": f"MGH ({bsb})",
            "language": "la",
            "language_stage": "medieval",   # MGH spans late-antique->medieval; override per-work
            "license": "CC BY 4.0 (openMGH, MGH/BSB)",
            "has_existing_translation": status.has_existing_translation,
            "translation_status": status.status,
        })
        meta.update(meta_overrides)
        return meta, parts

    def discover(self, query: str = "all", limit: int = 200) -> List[str]:
        """List openMGH BSB ids, optionally filtered by a series/title substring.

        query "all"/"openmgh"/"editions" -> every available volume.
        Any other string -> volumes whose index title contains it (case-insensitive),
        e.g. "Merov", "Urkunden", "Auct. ant.", "Fortunatus".
        """
        catalogue = self._catalogue()
        q = query.strip().lower()
        if q in ("all", "openmgh", "editions", "catalogue", ""):
            ids = list(catalogue)
        else:
            ids = [bsb for bsb, title in catalogue.items() if q in title.lower()]
        return ids[:limit]

    def catalogue(self) -> List[Tuple[str, str]]:
        """(bsb_id, title) for every available volume — handy for browsing."""
        return list(self._catalogue().items())

    # -- helpers -------------------------------------------------------------

    def _catalogue(self) -> Dict[str, str]:
        """Scrape the editions index into an ordered {bsb_id: title} map."""
        resp = self.session.get(self.EDITIONS_URL, timeout=self.timeout)
        resp.raise_for_status()
        out: Dict[str, str] = {}
        for m in re.finditer(
            r'<a[^>]+href="https://data\.mgh\.de/openmgh/(bsb\d+)\.zip"[^>]*>(.*?)</a>',
            resp.text, re.S,
        ):
            bsb = m.group(1)
            title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.group(2))).strip()
            out.setdefault(bsb, title)
        return out

    @staticmethod
    def _extract_xml(zip_bytes: bytes, bsb: str) -> bytes:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
            if not names:
                raise ValueError(f"No XML in openMGH zip for {bsb}: {zf.namelist()}")
            # Prefer the member named after the volume; else the first/largest XML.
            preferred = next((n for n in names if bsb in n), None)
            if preferred is None:
                preferred = max(names, key=lambda n: zf.getinfo(n).file_size)
            return zf.read(preferred)

    @staticmethod
    def _normalize(identifier: str) -> str:
        m = re.search(r"(bsb\d+)", identifier, re.IGNORECASE)
        if m:
            return m.group(1).lower()
        digits = identifier.strip()
        if digits.isdigit():
            return f"bsb{int(digits):08d}"
        raise ValueError(f"Not an openMGH id/URL: {identifier!r}")
