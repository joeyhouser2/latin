"""Connector for VD17 and VD18 (German-region imprints, via the K10plus SRU).

The Verzeichnis der im deutschen Sprachbereich erschienenen Drucke is the
national bibliography of German-region printing: VD17 (1601-1700, ~300k
records) and VD18 (1701-1800). Both are served, openly, by the GBV/K10plus
SRU endpoint as MARC21 -- no key, no bot check -- and a large share of records
carry a free link to a digitized copy. Most of that Latin (dissertations,
legal and theological treatises, occasional verse) has never been translated.

**This is a catalogue first.** A VD record describes a book; the scan lives at
whichever library digitized it (MDZ, Gottingen, Halle, Wolfenbuttel, ...).
``fetch()`` follows the record's links through ``ingest.copies`` and ingests
the first copy a full-text connector can read (today MDZ and archive.org); when
none can, it raises ``NoReadableCopy`` listing the links rather than guessing.
``catalog()`` returns the records either way, so the connector is useful for
discovery even where fetching is not yet possible.

VD16 is not exposed on this endpoint (no database path answers), so it is not
covered.

Identifiers:  ``vd17:<PPN>`` / ``vd18:<PPN>`` (PPN = MARC field 001).
discover():   free words search titles; a string containing ``pica.`` is passed
              through as raw CQL. Both databases are searched unless the query
              is prefixed ``vd17:`` / ``vd18:``. Latin only (``pica.spr=lat``).

Usage:
    python scripts/ingest.py vd "vd17:005436001#pages=1-30"
    python scripts/ingest.py vd "usuris" --discover --limit 20
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import requests

from .base import Connector, RawWork
from .copies import fetch_first_readable
from .treatises import _stage_for

_SRU = "https://sru.k10plus.de/{db}"
_NS = {"zs": "http://www.loc.gov/zing/srw/", "m": "http://www.loc.gov/MARC21/slim"}
_DBS = ("vd17", "vd18")
_YEAR = re.compile(r"\b(1[5-8][0-9]{2})\b")
_LANG3 = {"lat": "la", "grc": "grc", "gre": "grc", "ger": "de", "fre": "fr",
          "ita": "it", "dut": "nl"}


def _subs(field: ET.Element, codes: str) -> List[str]:
    return [(s.text or "").strip() for s in field.findall("m:subfield", _NS)
            if s.get("code") in codes and s.text]


def parse_record(rec: ET.Element, db: str) -> Dict[str, Any]:
    """MARC21 record element -> catalogue dict."""
    def ctrl(tag):
        c = rec.find(f"m:controlfield[@tag='{tag}']", _NS)
        return (c.text or "").strip() if c is not None and c.text else ""

    def fields(tag):
        return rec.findall(f"m:datafield[@tag='{tag}']", _NS)

    def first(tag, codes="a"):
        for f in fields(tag):
            v = _subs(f, codes)
            if v:
                return " ".join(v)
        return None

    title_main = first("245", "a") or ""
    title_sub = first("245", "b")
    title = (title_main + (" : " + title_sub if title_sub else "")).strip(" /:")
    author = first("100") or first("700")
    imprint = first("264", "c") or first("260", "c") or ""
    year_m = _YEAR.search(imprint) or _YEAR.search(ctrl("008")[7:11])
    year = int(year_m.group(1)) if year_m else None
    langs = [_LANG3.get(l, l) for f in fields("041") for l in _subs(f, "a")]
    if not langs:
        l008 = ctrl("008")[35:38]
        langs = [_LANG3.get(l008, l008)] if l008.strip() else []
    vd_no = None
    for f in fields("024"):
        if "vd1" in " ".join(_subs(f, "2")).lower():
            vd_no = " ".join(_subs(f, "a"))
    links = []
    for f in fields("856"):
        u = _subs(f, "u")
        if u:
            note = " ".join(_subs(f, "z") + _subs(f, "3") + _subs(f, "x"))
            links.append({"url": u[0], "note": note})
    free = [l["url"] for l in links
            if any(w in l["note"].lower() for w in ("kostenfrei", "volltext", "digitalisierung"))]
    return {
        "ppn": ctrl("001"), "db": db, "vd_number": vd_no, "title": title,
        "author": author, "year": year, "languages": langs,
        "place": first("264", "a") or first("260", "a"),
        "shelfmark": first("924", "g"),
        "links": links,
        "free_links": free or [l["url"] for l in links],
    }


class VDConnector(Connector):
    name = "vd"

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "LatinRAG-Research/1.0 (scholarly research)"

    # ---- SRU --------------------------------------------------------------
    def _search(self, db: str, cql: str, limit: int, start: int = 1) -> List[Dict[str, Any]]:
        resp = self.session.get(_SRU.format(db=db), params={
            "version": "1.1", "operation": "searchRetrieve", "query": cql,
            "maximumRecords": max(1, min(limit, 100)), "startRecord": start,
            "recordSchema": "marcxml"}, timeout=self.timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        diag = root.find(".//{*}diagnostic/{*}message")
        if diag is not None:
            raise ValueError(f"SRU error from {db}: {diag.text}")
        return [parse_record(r, db) for r in root.findall(".//m:record", _NS)]

    def catalog(self, query: str, limit: int = 50,
                dbs: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Catalogue records (with digital-copy links) for a query."""
        q = query.strip()
        use = dbs or _DBS
        m = re.match(r"(vd17|vd18):\s*(.*)", q, re.IGNORECASE)
        if m:
            use, q = (m.group(1).lower(),), m.group(2)
        cql = q if "pica." in q else f"pica.tit={self._term(q)} and pica.spr=lat"
        out: List[Dict[str, Any]] = []
        for db in use:
            out += self._search(db, cql, limit - len(out))
            if len(out) >= limit:
                break
        return out[:limit]

    @staticmethod
    def _term(q: str) -> str:
        words = re.findall(r"\w+", q)
        return " and pica.tit=".join(words) if words else q

    def discover(self, query: str, limit: int = 50) -> List[str]:
        """Identifiers of records that have a link to a digital copy."""
        ids = []
        for rec in self.catalog(query, limit=limit * 3):
            if rec["links"]:
                ids.append(f"{rec['db']}:{rec['ppn']}")
            if len(ids) >= limit:
                break
        return ids

    # ---- fetch ------------------------------------------------------------
    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        base, _, opts = identifier.partition("#")
        m = re.match(r"(vd17|vd18):(\w+)$", base.strip(), re.IGNORECASE)
        if not m:
            raise ValueError(f"expected vd17:<PPN> or vd18:<PPN>, got {identifier!r}")
        db, ppn = m.group(1).lower(), m.group(2)
        recs = self._search(db, f"pica.ppn={ppn}", 1)
        if not recs:
            raise ValueError(f"no {db.upper()} record with PPN {ppn}")
        rec = recs[0]
        if not rec["links"]:
            raise ValueError(f"{db.upper()} {ppn} ({rec['title'][:60]!r}) has no link to a "
                             "digital copy; nothing to ingest")
        cat_meta = {
            "title": rec["title"], "author": rec["author"],
            "century": ((rec["year"] - 1) // 100 + 1) if rec["year"] else None,
            "language": "la" if "la" in rec["languages"] else
                        (rec["languages"] or ["la"])[0],
            "language_stage": _stage_for(rec["year"]),
            "shelfmark": rec["shelfmark"],
        }
        cat_meta = {k: v for k, v in cat_meta.items() if v}
        if len(set(rec["languages"])) > 1:
            # mixed Latin/German: let the OCR stage judge which dominates
            cat_meta.pop("language", None)
        cat_meta.update(meta_overrides)
        meta, parts = fetch_first_readable(rec["free_links"], options=opts, **cat_meta)
        meta["source"] = (f"{db.upper()} {rec['vd_number'] or ppn} "
                          f"(K10plus PPN {ppn}); {meta.get('source', '')}").strip("; ")
        return meta, parts
