"""Connector for Gallica (Bibliothèque nationale de France).

Gallica is the best single catalogue for early-modern Latin *technical* prose --
the juridical, fiscal and commercial treatises that nobody has ever translated
because nobody wants to read them. A single search for "de usuris" restricted to
Latin returns 461 works; the BnF's own Droit/Économie department holds most of
them.

**This connector catalogues, it does not fetch.** Gallica's search API (SRU) is
open and unauthenticated, but every full-text endpoint -- ``.texteBrut``,
``RequestDigitalElement``, the page-image viewer -- now sits behind an ALTCHA
bot check that returns a JavaScript shell to any non-browser client. Two things
follow, and both matter:

* ``fetch()`` raises instead of returning text. The failure mode it prevents is
  the nasty one: the bot-check shell is a 50KB HTML page of French navigation
  chrome, and a connector that blindly stripped tags would ingest "Recherche
  avancée Accéder au menu" as a Latin treatise and queue it for translation.
* what the connector is *for* is discovery. ``catalog()`` returns records rich
  enough to decide from -- title, author, date, language, BnF shelfmark, ark
  URL -- and the companion ``treatises`` connector pairs each one against
  archive.org, which does serve its OCR.

Usage:
    from ingest.gallica import GallicaConnector
    g = GallicaConnector()
    for rec in g.catalog("de usuris", limit=10):
        print(rec["year"], rec["title"][:60], rec["url"])
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

import requests

from .base import Connector, RawWork

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0 Safari/537.36"}

_NS = {"srw": "http://www.loc.gov/zing/srw/",
       "dc": "http://purl.org/dc/elements/1.1/",
       "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/"}

# ark prefixes: bpt6k = printed monograph (has OCR behind the bot check),
# btv1b = manuscript or image-only scan (never has OCR at all).
_PRINTED_PREFIX = "bpt6k"

_BOT_CHECK_MARKERS = ("Vérification de sécurité", "ALTCHA", "je ne suis pas un robot")


class GallicaFullTextUnavailable(RuntimeError):
    """Raised by fetch(): BnF serves full text only to interactive browsers."""


class GallicaConnector(Connector):
    name = "gallica"
    SRU = "https://gallica.bnf.fr/SRU"

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)

    # -- catalogue -----------------------------------------------------------

    def catalog(self, query: str, limit: int = 25, language: str = "lat",
                doc_type: str = "monographie") -> List[Dict[str, Any]]:
        """Search Gallica and return one dict per work.

        ``query`` is either a plain phrase (matched across the record with the
        ``gallica all`` index) or a raw CQL clause if it already contains an
        index name and ``all``/``any``, so power users can hand-write e.g.
        ``dc.creator all "Salmasius"``.
        """
        records = self._search(self._cql(query, language, doc_type), limit)
        return [self._to_record(r) for r in records]

    def discover(self, query: str, limit: int = 25) -> List[str]:
        """Return ark identifiers. They are *not* fetchable -- see the module docstring."""
        return [rec["identifier"] for rec in self.catalog(query, limit=limit)]

    def _cql(self, query: str, language: str, doc_type: str) -> str:
        q = query.strip()
        if not re.search(r"\b(all|any|adj)\b", q):
            q = f'gallica all "{q}"'
        clauses = [f"({q})"]
        if language:
            clauses.append(f'(dc.language all "{language}")')
        if doc_type:
            clauses.append(f'(dc.type all "{doc_type}")')
        return " and ".join(clauses)

    def _search(self, cql: str, limit: int) -> List[ET.Element]:
        out: List[ET.Element] = []
        start = 1
        while len(out) < limit:
            page = min(50, limit - len(out))
            resp = self.session.get(self.SRU, params={
                "operation": "searchRetrieve", "version": "1.2", "query": cql,
                "maximumRecords": page, "startRecord": start,
            }, timeout=self.timeout)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            got = root.findall(".//srw:recordData/oai_dc:dc", _NS)
            if not got:
                break
            out.extend(got)
            start += page
            total = root.findtext("srw:numberOfRecords", default="0", namespaces=_NS)
            if start > int(total or 0):
                break
        return out[:limit]

    @staticmethod
    def _to_record(dc: ET.Element) -> Dict[str, Any]:
        def all_of(tag: str) -> List[str]:
            return [(e.text or "").strip() for e in dc.findall(f"dc:{tag}", _NS)
                    if (e.text or "").strip()]

        def first(tag: str) -> Optional[str]:
            vals = all_of(tag)
            return vals[0] if vals else None

        idents = all_of("identifier")
        url = next((i for i in idents if "ark:" in i), "")
        ark = url.rsplit("/", 1)[-1] if url else ""
        date = first("date") or ""
        year = None
        m = re.search(r"(\d{4})", date)
        if m:
            year = int(m.group(1))
        creator = first("creator") or ""
        # BnF appends a role to names: "Leotardo, Onorato. Auteur du texte".
        creator = re.sub(r"\.\s*(Auteur du texte|Éditeur scientifique).*$", "", creator).strip()
        return {
            "catalogue": "gallica",
            "identifier": ark,
            "title": (first("title") or "").strip(),
            "author": creator or None,
            "year": year,
            "date": date,
            "language": all_of("language"),
            "publisher": first("publisher"),
            "shelfmark": first("source"),
            "rights": first("rights"),
            "url": url,
            # Manuscripts have no OCR at all; printed monographs have it but it
            # is gated. Either way nothing here can be fetched automatically.
            "has_text": ark.startswith(_PRINTED_PREFIX),
            "fetchable": False,
            "note": "catalogue only — BnF serves full text to browsers only",
        }

    # -- fetch (deliberately unavailable) ------------------------------------

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        ark = identifier.strip().rsplit("/", 1)[-1]
        raise GallicaFullTextUnavailable(
            f"Gallica does not serve full text to scripts (ark {ark}). Open "
            f"https://gallica.bnf.fr/ark:/12148/{ark} in a browser to download "
            f"the text, then ingest it with the 'file' connector — or look for "
            f"the same work on archive.org via the 'treatises' connector."
        )

    @staticmethod
    def looks_like_bot_check(html: str) -> bool:
        """True if a response is the ALTCHA interstitial rather than content.

        Kept public so anything that ever does get text out of Gallica (a manual
        download, say) can check it before trusting the bytes.
        """
        head = html[:4000].lower()
        return any(m.lower() in head for m in _BOT_CHECK_MARKERS)
