"""Connector for the dull books: financial, fiscal and commercial treatises.

The library's centre of gravity has been poetry, liturgy and patristics --
material people translate because they want to read it. This connector goes
after the opposite: early-modern Latin (and some Greek) technical prose on
money, interest, exchange, taxation, weights and accounting. *De usuris*,
*De monetis*, *De cambiis*, *De vectigalibus populi Romani*. These are
genuinely untranslated, not "untranslated because nobody got round to it" --
nobody has ever wanted to read them in English, which is exactly why an MT
pipeline is the only way they ever will be.

Two catalogues, with different jobs:

* **archive.org** -- searched by title against a built-in vocabulary of the
  field, and the only one of the two that serves text. Its ``bub_gb_*`` items
  are Google Books scans of precisely this literature. Crucially, the fetch path
  goes through ``/metadata/<id>`` first to find the item's actual text
  derivative: roughly one item in five here is image-only (no OCR was ever run),
  and the naive ``<id>_djvu.txt`` URL that ``ingest.archive_org`` uses 404s on
  those. A catalogue entry says up front whether it can be fetched.
* **Gallica** -- catalogue only (see ``ingest.gallica``); the BnF's Droit/
  Économie holdings are the deepest collection of this material anywhere, so it
  is worth searching even though the text has to be got elsewhere.

Long OCR blobs are split into numbered sections of roughly ``section_words``
words. A 600-page folio arriving as one undivided section is not merely untidy:
every scoped pass in this project (``--section-range`` on the translate and
stylize scripts) works in sections, so a single-section document is all-or-
nothing -- 40,000 segments or none. Chunking is what makes "translate the first
part and see if the OCR is good enough" possible.

Usage:
    from ingest.treatises import TreatisesConnector
    t = TreatisesConnector()
    t.themes()                                  # what it knows how to look for
    recs = t.catalog("usury", limit=20)         # a theme name...
    recs = t.catalog("de ponderibus", limit=20) # ...or free text
    meta, parts = t.fetch("ia:bub_gb_D2hqS7meY7YC")

    python scripts/ingest.py treatises usury --discover --limit 10 --stage early_modern
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

from .base import Connector, RawWork
from .gallica import GallicaConnector
from .translation_status import infer_translation_status

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0 Safari/537.36"}


@dataclass
class Theme:
    """One subject area, with the words each catalogue actually responds to.

    The split is not redundancy. archive.org's title index tokenises and drops
    ``de``, so a quoted ``"de usuris"`` matches nothing while the bare stem
    ``usuris`` matches eighteen works; Gallica's SRU, by contrast, wants the
    phrase. Each catalogue gets the form it answers.
    """

    label: str
    ia_terms: List[str]
    gallica_phrases: List[str]
    genre: str
    grc_terms: List[str] = field(default_factory=list)


THEMES: Dict[str, Theme] = {
    "usury": Theme(
        label="Usury & interest",
        genre="finance: usury",
        ia_terms=["usuris", "usura", "usurarum", "usurario", "foenore", "fenore",
                  "faenore", "mutuo", "anatocismo"],
        gallica_phrases=["de usuris", "de usura", "de foenore",
                         "contractibus usurariis", "de mutuo"],
        grc_terms=["tokon", "tokos"],
    ),
    "money": Theme(
        label="Money & coinage",
        genre="finance: money",
        ia_terms=["monetis", "moneta", "monetarum", "nummaria", "nummis",
                  "numaria", "asse", "pecunia"],
        gallica_phrases=["de monetis", "de moneta", "de re nummaria",
                         "de nummis", "de asse"],
        grc_terms=["nomismatos", "nummis"],
    ),
    "exchange": Theme(
        label="Exchange & banking",
        genre="finance: exchange",
        ia_terms=["cambiis", "cambio", "cambiorum", "nundinis", "campsoribus"],
        gallica_phrases=["de cambiis", "de cambio", "de nundinis"],
    ),
    "commerce": Theme(
        label="Trade & commercial law",
        genre="commerce",
        ia_terms=["mercatura", "mercatorum", "negotiatione", "emptione",
                  "venditione", "societate", "contractibus", "assecurationibus"],
        gallica_phrases=["de mercatura", "de negotiatione",
                         "de emptione et venditione", "de contractibus",
                         "de assecurationibus"],
    ),
    "tax": Theme(
        label="Taxes, tribute & public revenue",
        genre="finance: fiscal",
        ia_terms=["vectigalibus", "tributis", "gabellis", "aerario", "censu",
                  "decimis", "portoriis"],
        gallica_phrases=["de vectigalibus", "de tributis", "de gabellis",
                         "de aerario", "de decimis"],
    ),
    "weights": Theme(
        label="Weights & measures",
        genre="metrology",
        ia_terms=["ponderibus", "mensuris", "metrologia"],
        gallica_phrases=["de ponderibus et mensuris", "de ponderibus",
                         "de mensuris"],
        grc_terms=["metron", "stathmon"],
    ),
    "accounting": Theme(
        label="Arithmetic & accounting",
        genre="accounting",
        ia_terms=["arithmetica", "computis", "computus", "logistica",
                  "rationibus"],
        gallica_phrases=["arithmetica mercatorum", "de computis",
                         "arithmetica practica"],
        grc_terms=["logistike", "arithmetike"],
    ),
    "economy": Theme(
        label="Household & political economy",
        genre="economy",
        ia_terms=["oeconomia", "oeconomica", "oeconomicus", "annona"],
        gallica_phrases=["oeconomia", "de re familiari", "de annona"],
        grc_terms=["oikonomikos", "oikonomia"],
    ),
}

# Language names as archive.org records them, mapped to our two-letter codes.
_LANG_MAP = {"latin": "la", "lat": "la", "la": "la",
             "greek": "grc", "ancient greek": "grc", "grc": "grc", "gre": "grc"}

# Default for printed items when we have a publication year: the *edition* is
# early modern even when the text is older. Calling it early_modern picks the
# right translator for the orthography actually on the page (u/v, long s, heavy
# abbreviation), which is what matters for MT; pass --stage to override when the
# work itself is ancient or medieval.
_EARLY_MODERN_FROM = 1450
_EARLY_MODERN_TO = 1850


class TreatisesConnector(Connector):
    name = "treatises"
    IA_SEARCH = "https://archive.org/advancedsearch.php"
    IA_META = "https://archive.org/metadata"
    IA_DOWNLOAD = "https://archive.org/download"

    def __init__(self, timeout: float = 60.0, section_words: int = 1200):
        self.timeout = timeout
        self.section_words = section_words
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self.gallica = GallicaConnector(timeout=timeout)

    # -- vocabulary ----------------------------------------------------------

    @staticmethod
    def themes() -> List[Dict[str, Any]]:
        """The built-in subject vocabulary, for a UI to offer as buttons."""
        return [{"key": k, "label": t.label, "genre": t.genre,
                 "terms": t.ia_terms[:6] + t.grc_terms[:2]}
                for k, t in THEMES.items()]

    # -- catalogue -----------------------------------------------------------

    def catalog(self, query: str, limit: int = 25, language: str = "",
                catalogues: Tuple[str, ...] = ("archive", "gallica"),
                check_text: bool = True) -> List[Dict[str, Any]]:
        """Search the catalogues and return merged, decidable records.

        ``query`` is a theme key ("usury"), ``"all"``, or free text. ``language``
        narrows to ``la`` or ``grc``; empty searches both. Records are sorted
        so the ones you can actually ingest come first.
        """
        theme = THEMES.get(query.strip().lower())
        records: List[Dict[str, Any]] = []
        if "archive" in catalogues:
            records += self._archive_search(query, theme, limit, language,
                                            check_text=check_text)
        if "gallica" in catalogues:
            try:
                records += self._gallica_search(query, theme, limit, language)
            except Exception as exc:                          # noqa: BLE001
                # A catalogue being down should degrade the result, not empty it.
                records.append({"catalogue": "gallica", "identifier": "",
                                "title": f"(Gallica search failed: {exc})",
                                "fetchable": False, "error": True})
        records.sort(key=lambda r: (not r.get("fetchable"), r.get("year") or 9999))
        return records[:limit * len(catalogues)]

    def discover(self, query: str, limit: int = 25) -> List[str]:
        """Identifiers for the *fetchable* hits only, ready for fetch()."""
        recs = self.catalog(query, limit=limit, catalogues=("archive",))
        return [r["identifier"] for r in recs if r.get("fetchable")]

    def _archive_search(self, query: str, theme: Optional[Theme], limit: int,
                        language: str, check_text: bool) -> List[Dict[str, Any]]:
        clauses = []
        terms = self._ia_terms(query, theme, language)
        clauses.append("title:(" + " OR ".join(terms) + ")")
        clauses.append("mediatype:(texts)")
        if language == "la":
            clauses.append("language:(latin)")
        elif language == "grc":
            clauses.append("language:(greek)")
        else:
            clauses.append("language:(latin OR greek)")
        resp = self.session.get(self.IA_SEARCH, params={
            "q": " AND ".join(clauses),
            "fl[]": ["identifier", "title", "year", "creator", "language",
                     "publisher", "subject"],
            "rows": limit, "output": "json",
        }, timeout=self.timeout)
        resp.raise_for_status()
        docs = resp.json().get("response", {}).get("docs", [])

        records = [self._ia_record(d, theme) for d in docs]
        if check_text and records:
            # Whether an item has OCR is the single most useful fact about it
            # here and costs one request each; do them in parallel so a 25-row
            # catalogue page still comes back in a couple of seconds.
            with ThreadPoolExecutor(max_workers=6) as pool:
                for rec, info in zip(records, pool.map(
                        lambda r: self.text_file(r["ia_id"]), records)):
                    rec["text_file"] = info
                    rec["fetchable"] = bool(info)
                    if not info:
                        rec["note"] = "no OCR text on archive.org (scan only)"
        return records

    def _gallica_search(self, query: str, theme: Optional[Theme], limit: int,
                        language: str) -> List[Dict[str, Any]]:
        phrases = (theme.gallica_phrases if theme else [query.strip()])
        if query.strip().lower() == "all":
            phrases = [p for t in THEMES.values() for p in t.gallica_phrases[:2]]
        lang = {"la": "lat", "grc": "grc"}.get(language, "lat")
        out: List[Dict[str, Any]] = []
        seen = set()
        # One SRU call per phrase, spreading the budget across them so a single
        # prolific phrase ("de monetis") cannot fill the whole page.
        per = max(2, limit // max(1, len(phrases[:5])))
        for phrase in phrases[:5]:
            for rec in self.gallica.catalog(phrase, limit=per, language=lang):
                if rec["identifier"] in seen:
                    continue
                seen.add(rec["identifier"])
                rec["genre"] = theme.genre if theme else None
                rec["theme"] = query if theme else None
                out.append(rec)
        return out

    @staticmethod
    def _ia_terms(query: str, theme: Optional[Theme], language: str) -> List[str]:
        if theme:
            terms = list(theme.ia_terms) + (list(theme.grc_terms)
                                            if language != "la" else [])
        elif query.strip().lower() == "all":
            terms = [t for th in THEMES.values() for t in th.ia_terms[:4]]
        else:
            # Free text: quote multi-word input, pass single words bare (the
            # title index drops short function words from phrases).
            q = query.strip()
            terms = [f'"{q}"' if " " in q else q]
        return terms

    def _ia_record(self, doc: Dict[str, Any], theme: Optional[Theme]) -> Dict[str, Any]:
        ident = doc.get("identifier", "")
        langs = doc.get("language") or []
        if isinstance(langs, str):
            langs = [langs]
        year = doc.get("year")
        try:
            year = int(year) if year else None
        except (TypeError, ValueError):
            year = None
        creator = doc.get("creator")
        if isinstance(creator, list):
            creator = creator[0] if creator else None
        return {
            "catalogue": "archive",
            "identifier": f"ia:{ident}",
            "ia_id": ident,
            "title": _clean(doc.get("title")),
            "author": _clean(creator),
            "year": year,
            "language": langs,
            "lang_code": _lang_code(langs),
            "publisher": _clean(doc.get("publisher")),
            "genre": theme.genre if theme else None,
            "theme": None if theme is None else theme.label,
            "url": f"https://archive.org/details/{ident}",
            "fetchable": True,          # refined by the metadata check
        }

    # -- fetch ---------------------------------------------------------------

    def text_file(self, ia_id: str) -> Optional[str]:
        """The name of the item's OCR text file, or None if it has none.

        archive.org's naming is not uniform -- ``<id>_djvu.txt`` is the common
        case but ``<id>_text.txt`` and bare ``<id>.txt`` both occur, and
        image-only items have nothing at all.
        """
        try:
            meta = self.session.get(f"{self.IA_META}/{ia_id}",
                                    timeout=self.timeout).json()
        except Exception:                                      # noqa: BLE001
            return None
        names = [f.get("name", "") for f in meta.get("files", [])]
        for candidate in (f"{ia_id}_djvu.txt", f"{ia_id}_text.txt", f"{ia_id}.txt"):
            if candidate in names:
                return candidate
        txts = [n for n in names if n.lower().endswith(".txt")
                and "_meta" not in n.lower()]
        return txts[0] if txts else None

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        """Fetch one work's OCR text, split into numbered sections."""
        ident = identifier.strip()
        if ident.startswith("gallica:"):
            return self.gallica.fetch(ident.split(":", 1)[1], **meta_overrides)
        ia_id = ident.split(":", 1)[1] if ident.startswith("ia:") else ident

        name = self.text_file(ia_id)
        if not name:
            raise ValueError(
                f"archive.org item {ia_id!r} has no OCR text derivative "
                f"(image-only scan). Nothing to ingest."
            )
        resp = self.session.get(f"{self.IA_DOWNLOAD}/{ia_id}/{name}",
                                timeout=max(self.timeout, 120))
        resp.raise_for_status()
        text = resp.text

        meta = self._meta_for(ia_id, meta_overrides)
        parts = self.split_sections(text, self.section_words)
        return meta, parts

    def _meta_for(self, ia_id: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        try:
            md = self.session.get(f"{self.IA_META}/{ia_id}",
                                  timeout=self.timeout).json().get("metadata", {})
        except Exception:                                      # noqa: BLE001
            md = {}
        langs = md.get("language") or []
        if isinstance(langs, str):
            langs = [langs]
        year = None
        m = re.search(r"(1[0-9]{3}|20[0-9]{2})", str(md.get("date") or md.get("year") or ""))
        if m:
            year = int(m.group(1))
        title = _clean(md.get("title")) or ia_id
        status = infer_translation_status(title=title, genre=_clean(md.get("subject")))
        meta = {
            "title": title,
            "author": _clean(md.get("creator")),
            "language": _lang_code(langs),
            "language_stage": _stage_for(year),
            "century": ((year - 1) // 100 + 1) if year else None,
            "source": f"Internet Archive ({ia_id})",
            "license": "Public domain (pre-1929 scan, raw OCR text)",
            "has_existing_translation": status.has_existing_translation,
            "translation_status": status.status,
        }
        meta.update({k: v for k, v in overrides.items() if v is not None})
        return meta

    @staticmethod
    def split_sections(text: str, section_words: int = 1200) -> List[Tuple[str, str]]:
        """Split a long OCR blob into numbered sections of ~``section_words``.

        Splits on blank-line paragraph boundaries and never mid-paragraph, so a
        sentence is not cut in half across a section boundary (which would leave
        the segmenter with two fragments and the translator with two
        hallucinations).
        """
        text = text.replace("\r\n", "\n")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return []
        sections: List[Tuple[str, str]] = []
        buf: List[str] = []
        count = 0
        for para in paragraphs:
            buf.append(para)
            count += len(para.split())
            if count >= section_words:
                sections.append((f"Part {len(sections) + 1}", "\n\n".join(buf)))
                buf, count = [], 0
        if buf:
            sections.append((f"Part {len(sections) + 1}", "\n\n".join(buf)))
        return sections


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if not value:
        return None
    return re.sub(r"\s+", " ", str(value)).strip()


def _lang_code(langs: List[str]) -> str:
    """Our two-letter code for an archive.org language list.

    A mixed Latin/Greek item is filed as Latin: that is what the bulk of the
    text is, and the Latin path already handles embedded Greek quotations
    through ingest.mixed_lang_translate.
    """
    codes = [_LANG_MAP.get(str(l).strip().lower()) for l in langs]
    codes = [c for c in codes if c]
    if "la" in codes:
        return "la"
    return codes[0] if codes else "la"


def _stage_for(year: Optional[int]) -> str:
    if year is None:
        return "unknown"
    if _EARLY_MODERN_FROM <= year <= _EARLY_MODERN_TO:
        return "early_modern"
    if year > _EARLY_MODERN_TO:
        return "unknown"       # a 19th-c. reprint says nothing about the text
    return "medieval"
