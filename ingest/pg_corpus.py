"""Connector for the Patrologia Graeca Corpus (Vidal-Gorene & Kindt) — an
open OCR + lemmatized rendering of Migne's Patrologia Graeca, i.e. patristic and
**Byzantine Greek** (the under-covered 600-1000 window). Published on Zenodo
(record 15780625) and GitHub `calfa-co/Patrologia-Graeca`, one plaintext file per
Migne PG volume: ``<PGvol>/<PGvol>_text.txt`` (e.g. ``PG003/PG003_text.txt``).

The OCR has already isolated the Greek columns (Migne prints Greek + a Latin
translation side by side); the text files are essentially pure Greek. Each file
is interleaved with page markers on their own line:

    $0=3 $8=71 $9=1        ->  PG volume 3, page 71, column 1

so we cut the text at every marker and make each page-column a Section (label
doubles as a citable source_loc), then segment the Greek within. A whole PG
volume is ingested as one Document (it bundles several works, but the corpus has
no per-work boundaries — page/column is the finest reliable structure).

These are critical-edition reprints with no English translation alongside, so
translation_status defaults to "unknown" (Migne carries a *Latin* rendering, not
English). Released open (see the Zenodo record for the exact CC terms).

Usage:
    from ingest.pg_corpus import PGCorpusConnector
    meta, parts = PGCorpusConnector().fetch("PG003")     # or "3"
    vols = PGCorpusConnector().discover("all")           # every available volume
"""

from __future__ import annotations

from typing import List, Tuple
import re

import requests

from .base import Connector, RawWork
from .translation_status import UNKNOWN


# A marker line: one or more "$<key>=<value>" tokens, nothing else.
_MARKER = re.compile(r"^\s*(\$\d+=\S+)(?:\s+\$\d+=\S+)*\s*$")
_KV = re.compile(r"\$(\d+)=(\S+)")


class PGCorpusConnector(Connector):
    name = "pg_corpus"
    REPO = "calfa-co/Patrologia-Graeca"
    API = "https://api.github.com/repos/calfa-co/Patrologia-Graeca/contents"

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly; jth156@case.edu)"}
        )

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        vol = self._normalize(identifier)
        download_url = self._text_download_url(vol)
        resp = self.session.get(download_url, timeout=self.timeout)
        resp.raise_for_status()
        resp.encoding = "utf-8"

        parts = self._split_by_marker(resp.text, vol)
        meta = {
            "title": f"Patrologia Graeca {self._vol_num(vol)} (Migne, OCR)",
            "author": None,                 # a PG volume bundles several authors
            "language": "grc",
            "language_stage": "late_antique",
            "source": f"PG Corpus ({vol})",
            "license": "Patrologia Graeca Corpus (calfa-co); see Zenodo 15780625",
            "has_existing_translation": False,
            "translation_status": UNKNOWN,
        }
        meta.update(meta_overrides)
        return meta, parts

    def discover(self, query: str = "all", limit: int = 200) -> List[str]:
        """List the PG volume ids present in the corpus (e.g. ['PG003', ...])."""
        vols = [r["name"] for r in self._list("")
                if r.get("type") == "dir" and re.fullmatch(r"PG\d+(?:_\d+)?", r["name"])]
        vols.sort()
        q = query.strip().lower()
        if q not in ("all", "", "pg"):
            vols = [v for v in vols if q in v.lower()]
        return vols[:limit]

    # -- helpers -------------------------------------------------------------

    def _split_by_marker(self, text: str, vol: str) -> List[Tuple[str, str]]:
        """Cut the volume into (page-column label, Greek text) sections."""
        parts: List[Tuple[str, str]] = []
        label = f"PG {self._vol_num(vol)}"   # fallback before the first marker
        buf: List[str] = []

        def flush():
            body = " ".join(buf).strip()
            if body:
                parts.append((label, body))
            buf.clear()

        for line in text.splitlines():
            if _MARKER.match(line):
                flush()
                label = self._marker_label(line, vol)
            else:
                buf.append(line)
        flush()
        return parts

    @staticmethod
    def _marker_label(line: str, vol: str) -> str:
        kv = {k: v for k, v in _KV.findall(line)}
        # $0 = PG volume, $8 = page, $9 = column (from the calfa OCR format).
        v = kv.get("0", PGCorpusConnector._vol_num(vol))
        page, col = kv.get("8"), kv.get("9")
        out = f"PG {v}"
        if page:
            out += f", p.{page}"
        if col:
            out += f", col.{col}"
        return out

    def _text_download_url(self, vol: str) -> str:
        for r in self._list(vol):
            if r.get("type") == "file" and r["name"].lower().endswith(".txt"):
                return r["download_url"]
        raise ValueError(f"No text file found in {self.REPO}/{vol}")

    def _list(self, path: str) -> list:
        resp = self.session.get(f"{self.API}/{path}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _vol_num(vol: str) -> str:
        m = re.search(r"PG0*(\d+(?:_\d+)?)", vol)
        return m.group(1) if m else vol

    @staticmethod
    def _normalize(identifier: str) -> str:
        s = identifier.strip()
        m = re.search(r"PG(\d+(?:_\d+)?)", s, re.IGNORECASE)
        if m:
            num = m.group(1)
            # zero-pad the leading volume number to 3 digits (PG3 -> PG003).
            head, _, tail = num.partition("_")
            return f"PG{int(head):03d}" + (f"_{tail}" if tail else "")
        if s.isdigit():
            return f"PG{int(s):03d}"
        raise ValueError(f"Not a PG volume id: {identifier!r}")
