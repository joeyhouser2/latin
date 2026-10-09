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
        if identifier.strip().startswith("ia:"):
            from .treatises import TreatisesConnector
            ia_id = identifier.strip()[3:]
            m = re.search(r"vol_0*(\d+)|cursus0*(\d+)migngoog", ia_id)
            num = int(next(g for g in m.groups() if g)) if m else None
            who = self.VOLUME_AUTHORS.get(num) if num else None
            meta, parts = TreatisesConnector(timeout=self.timeout).fetch(ia_id)
            meta.update({"language": "grc", "language_stage": "late_antique",
                         "title": f"Patrologia Graeca {num} (Migne, archive.org scan)"
                                  if num else meta["title"],
                         "author": who[0] if who else meta.get("author"),
                         "translation_status": UNKNOWN})
            meta.update({k: v for k, v in meta_overrides.items() if v is not None})
            return meta, parts
        vol = self._normalize(identifier)
        download_url = self._text_download_url(vol)
        resp = self.session.get(download_url, timeout=self.timeout)
        resp.raise_for_status()
        resp.encoding = "utf-8"

        parts = self._split_by_marker(resp.text, vol)
        meta = {
            "title": f"Patrologia Graeca {self._vol_num(vol)} (Migne, OCR)",
            "author": (self.VOLUME_AUTHORS.get(int(self._vol_num(vol).split("_")[0])) or (None, None))[0],
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

    # Volumes whose author is known; a PG volume otherwise bundles several.
    # Chrysostom runs PG 47-64 (the Old Testament homilies are 53-56: Genesis
    # 53-54, Psalms 55, Isaiah/Job etc. 56; New Testament homilies 57-63).
    VOLUME_AUTHORS = {n: ("Ioannes Chrysostomus", "John Chrysostom")
                      for n in range(47, 65)}
    VOLUME_NOTES = {
        53: "Homilies on Genesis (1-32)", 54: "Homilies on Genesis (33-67), Psalms",
        55: "Expositions on the Psalms", 56: "Isaiah, Job, Jeremiah; spuria",
        57: "Matthew (1-45)", 58: "Matthew (46-90)", 59: "John (1-88)",
    }

    def catalog(self, query: str = "all", limit: int = 200):
        """Volume records for the UI. ``query`` may be 'all', 'chrysostom', or a
        substring of a volume id / note. Every record is fetchable."""
        q = query.strip().lower()
        out = []
        have = set()
        for vol in self.discover("all", limit=1000):
            have.add(int(self._vol_num(vol).split("_")[0]))
            num = int(self._vol_num(vol).split("_")[0])
            who = self.VOLUME_AUTHORS.get(num)
            note = self.VOLUME_NOTES.get(num, "")
            hay = f"{vol} {who[0] + ' ' + who[1] if who else ''} {note}".lower()
            if q not in ("all", "", "pg") and q not in hay:
                continue
            out.append({
                "catalogue": "pg_corpus", "identifier": vol,
                "title": f"Patrologia Graeca {num}" + (f" — {note}" if note else ""),
                "author": who[1] if who else None,
                "publisher": "Migne PG (Calfa OCR)",
                "year": None, "lang_code": "grc", "genre": "patristic",
                "fetchable": True,
                "translation_status": "unknown",
                "url": f"https://github.com/{self.REPO}/tree/main/{vol}",
            })
            if len(out) >= limit:
                break
        # Calfa has only ~33 volumes; Chrysostom (PG 47-64) is not among them.
        # Fall back to Migne's scans on archive.org (Greek + Latin columns
        # together in one OCR stream, so noisier than Calfa's Greek-only text).
        missing = [n for n in sorted(self.VOLUME_AUTHORS) if n not in have]
        if missing and len(out) < limit:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=6) as pool:
                found = list(pool.map(self._ia_volume, missing))
            for num, rec in zip(missing, found):
                if not rec:
                    continue
                note = self.VOLUME_NOTES.get(num, "")
                who = self.VOLUME_AUTHORS[num]
                hay = f"pg{num:03d} {who[0]} {who[1]} {note}".lower()
                if q not in ("all", "", "pg") and q not in hay:
                    continue
                out.append(rec | {"note": "archive.org scan of Migne; Greek and Latin columns interleaved"})
        return out[:limit]

    def _ia_volume(self, num: int):
        """The archive.org item for PG volume ``num``, or None."""
        try:
            r = self.session.get("https://archive.org/advancedsearch.php", params={
                "q": f"identifier:(patrologiae_cursus_completus_gr_vol_{num:03d}* OR "
                     f"patrologicursus{num}migngoog) AND mediatype:texts",
                "fl[]": ["identifier", "title"], "rows": 1, "output": "json",
            }, timeout=self.timeout).json()
            docs = r["response"]["docs"]
        except Exception:                                      # noqa: BLE001
            return None
        if not docs:
            return None
        ident = docs[0]["identifier"]
        who = self.VOLUME_AUTHORS[num]
        note = self.VOLUME_NOTES.get(num, "")
        return {
            "catalogue": "archive", "identifier": f"ia:{ident}",
            "title": f"Patrologia Graeca {num}" + (f" — {note}" if note else ""),
            "author": who[1], "publisher": "Migne PG (archive.org scan)",
            "year": None, "lang_code": "grc", "genre": "patristic",
            "fetchable": True, "translation_status": "unknown",
            "url": f"https://archive.org/details/{ident}",
        }

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
