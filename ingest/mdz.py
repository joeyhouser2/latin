"""Connector for the Munich Digitization Center (MDZ / Bayerische Staatsbibliothek).

The MDZ is one of the largest sources of scanned early-modern Latin, and unlike
Gallica it is open to scripts: the IIIF Presentation API, the IIIF image API and
a per-page hOCR service all answer plain HTTP requests.

    manifest  https://api.digitale-sammlungen.de/iiif/presentation/v2/<id>/manifest
    page OCR  https://api.digitale-sammlungen.de/ocr/<id>/<page>        (hOCR)
    page img  https://api.digitale-sammlungen.de/iiif/image/v2/<id>_<nnnnn>/full/<size>/0/default.jpg

Text comes from one of two places, chosen by the ``ocr`` option:

``mdz``        the library's own hOCR. Free and instant, but made once, in bulk,
               with a model nobody tuned for Latin: expect long-s and
               abbreviation damage on 16th-18th-c. prints.
``tesseract``  download the page images and OCR them locally with the ``lat``
               model (see ``ingest.ocr_images``). Slow, and not always better --
               try both on a few pages (``pages=10-14``) before committing.
``auto``       (default) the library's hOCR, with Tesseract only for pages the
               library has no usable text for.

Identifiers (all equivalent): ``bsb10135975``, a ``/view/bsb...`` or manifest
URL, or a page id ``bsb10135975_00005``. Options go after ``#`` as
``key=value&key=value``, e.g. ``bsb10135975#ocr=tesseract&pages=1-30``:
``ocr`` (above), ``pages`` (``a-b``, 1-based inclusive), ``size`` (IIIF image
width for Tesseract, default 2500), ``lang`` (Tesseract model, default ``lat``),
``workers``.

``discover()`` is not supported: the MDZ search page is rendered client-side
from an undocumented API. Find items on https://www.digitale-sammlungen.de
(filter: language = Latin) and pass their ``bsb`` ids, or use the Gallica /
``treatises`` catalogues to find candidates.

Usage:
    python scripts/ingest.py mdz "bsb11234567#pages=1-60" --stage early_modern
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from .base import Connector, RawWork
from .ocr_images import (OCRUnavailable, assemble, check_latin, ocr_image_bytes,
                         ocr_pages, parse_options, parse_page_range)
from .pagetext import hocr_to_text
from .treatises import _stage_for

_API = "https://api.digitale-sammlungen.de"
_HEADERS = {"User-Agent": "LatinRAG-Research/1.0 (scholarly research)"}
_ID = re.compile(r"(bsb\d{8})", re.IGNORECASE)
_YEAR = re.compile(r"\b(1[0-9]{3})\b")
_ROMAN_YEAR = re.compile(r"\bM(?:DCCC|CM|CD|D)?C{0,3}(?:XC|XL|L)?X{0,3}(?:IX|IV|V)?I{0,3}\b")
_TAGS = re.compile(r"<[^>]+>")
_MIN_PAGE_CHARS = 25      # below this a page's library OCR is treated as absent

_LANG_NAMES = {
    "latin": "la", "greek": "grc", "ancient greek": "grc", "german": "de",
    "french": "fr", "italian": "it", "dutch": "nl", "polish": "pl",
    "hungarian": "hu", "russian": "ru", "english": "en",
}


def parse_identifier(identifier: str) -> str:
    """Pull the ``bsb########`` item id out of an id / page id / URL."""
    m = _ID.search(identifier)
    if not m:
        raise ValueError(
            f"no MDZ item id (bsb########) found in {identifier!r}")
    return m.group(1).lower()


def _text(value) -> str:
    """Flatten a IIIF metadata value (str, list, or [{'@value': ...}])."""
    if isinstance(value, list):
        en = [v for v in value if isinstance(v, dict) and v.get("@language") == "en"]
        pool = en or value
        return "; ".join(filter(None, (_text(v) for v in pool)))
    if isinstance(value, dict):
        return _text(value.get("@value", ""))
    return re.sub(r"\s+", " ", _TAGS.sub("", str(value or ""))).strip()


def _roman(token: str) -> int:
    vals = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}
    total = 0
    for a, b in zip(token, token[1:] + " "):
        v = vals[a]
        total += -v if vals.get(b, 0) > v else v
    return total


def _year_of(date: str) -> Optional[int]:
    """Arabic year, else a Roman-numeral one ('Anno MDCCXLIV.' -> 1744)."""
    m = _YEAR.search(date)
    if m:
        return int(m.group(1))
    for tok in _ROMAN_YEAR.findall(date.upper()):
        if len(tok) >= 3:
            return _roman(tok)
    return None


def manifest_fields(manifest: dict) -> Dict[str, List[str]]:
    """English-label -> [values] from a v2 manifest's metadata block."""
    fields: Dict[str, List[str]] = {}
    for entry in manifest.get("metadata", []):
        label = entry.get("label")
        if isinstance(label, list):
            en = [l for l in label if isinstance(l, dict) and l.get("@language") == "en"]
            label = en[0]["@value"] if en else (label[0].get("@value") if label else "")
        label = _text(label)
        if label:
            fields.setdefault(label, []).append(_text(entry.get("value")))
    return fields


class MDZConnector(Connector):
    name = "mdz"

    def __init__(self, timeout: float = 60.0, delay: float = 0.1,
                 section_words: int = 1200, strict: bool = True):
        self.timeout = timeout
        self.delay = delay
        self.section_words = section_words
        self.strict = strict
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)

    # ---- HTTP -------------------------------------------------------------
    def _get(self, url: str, **kw) -> requests.Response:
        for attempt in range(4):
            resp = self.session.get(url, timeout=self.timeout, **kw)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            return resp
        return resp

    def manifest(self, item: str) -> dict:
        resp = self._get(f"{_API}/iiif/presentation/v2/{item}/manifest")
        if resp.status_code == 404:
            raise ValueError(f"MDZ has no item {item!r}")
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def canvases(manifest: dict) -> List[dict]:
        seqs = manifest.get("sequences") or [{}]
        return seqs[0].get("canvases", [])

    def page_ocr_url(self, item: str, canvas: dict, n: int) -> str:
        see = canvas.get("seeAlso")
        if isinstance(see, dict) and see.get("@id"):
            return see["@id"]
        return f"{_API}/ocr/{item}/{n}"

    @staticmethod
    def page_image_url(canvas: dict, size: str) -> str:
        res = canvas["images"][0]["resource"]
        service = (res.get("service") or {}).get("@id")
        if not service:
            return res["@id"]
        return f"{service}/full/{size}/0/default.jpg"

    # ---- metadata ---------------------------------------------------------
    def build_meta(self, item: str, manifest: dict) -> dict:
        f = manifest_fields(manifest)
        first = lambda k: (f.get(k) or [""])[0] or None
        year = _year_of(first("Date") or "") or _year_of(str(manifest.get("navDate") or ""))
        langs = [v for v in f.get("Language", []) for v in re.split(r";\s*", v)]
        codes = [_LANG_NAMES.get(l.strip().lower()) for l in langs]
        language = ("la" if "la" in codes else "grc" if "grc" in codes
                    else next((c for c in codes if c), "la"))
        author = first("By")
        if author:
            author = re.sub(r"^(Begr\.|Hrsg\.|Ed\.|Auth\.):\s*", "", author)
        return {
            "title": _text(manifest.get("label")) or item,
            "author": author,
            "century": ((year - 1) // 100 + 1) if year else None,
            "language": language,
            "language_stage": _stage_for(year),
            "source": f"MDZ / Bayerische Staatsbibliothek ({item}); "
                      f"https://www.digitale-sammlungen.de/en/view/{item}",
            "shelfmark": first("Call number"),
            "license": _text(manifest.get("license")) or
                       "See MDZ rights statement (usually public domain for pre-1900 works)",
            "has_existing_translation": False,
        }

    # ---- fetch ------------------------------------------------------------
    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        base, opts = parse_options(identifier)
        item = parse_identifier(base)
        mode = opts.get("ocr", "auto")
        if mode not in ("auto", "mdz", "tesseract"):
            raise ValueError(f"ocr must be auto|mdz|tesseract, not {mode!r}")
        lang = opts.get("lang", "lat")
        size = opts.get("size", "2500,")
        if size.isdigit():
            size += ","
        workers = int(opts.get("workers", 4))

        manifest = self.manifest(item)
        canvases = self.canvases(manifest)
        if not canvases:
            raise ValueError(f"MDZ item {item} has no page images")
        sel = parse_page_range(opts.get("pages"), len(canvases))
        meta = self.build_meta(item, manifest)
        meta.update(meta_overrides)

        print(f"  {item}: {meta['title']!r} -- {len(canvases)} pages "
              f"(using {sel.start + 1}-{sel.stop}), ocr={mode}", file=sys.stderr)

        pages: Dict[int, str] = {}
        if mode in ("auto", "mdz"):
            pages = self._library_ocr(item, canvases, sel)
            missing = [i for i in sel if len(pages.get(i, "").strip()) < _MIN_PAGE_CHARS]
            print(f"  library OCR: {len(sel) - len(missing)}/{len(sel)} pages have text",
                  file=sys.stderr)
        else:
            missing = list(sel)
        if mode == "mdz":
            missing = []                       # pages without text stay empty
        if missing:
            pages.update(self._tesseract_ocr(item, canvases, missing, lang, size, workers))

        ordered = [pages.get(i, "") for i in sel]
        if not any(p.strip() for p in ordered):
            raise ValueError(f"no OCR text obtained for {item}")
        rate, warn = check_latin("\n".join(ordered), f"{item}", self.strict,
                                 meta.get("language", "la"))
        if warn:
            print(f"  WARNING {warn}", file=sys.stderr)
            if mode == "mdz" or mode == "auto":
                print("  (try #ocr=tesseract on a few pages and compare)", file=sys.stderr)
        meta["_ocr_latin_rate"] = rate
        meta["_ocr_engine"] = mode
        if "source" in meta and "[OCR:" not in meta["source"]:
            meta["source"] += f" [OCR: {'library' if mode == 'mdz' else mode}]"
        return meta, assemble(ordered, self.section_words)

    def _library_ocr(self, item: str, canvases: List[dict], sel: range) -> Dict[int, str]:
        from concurrent.futures import ThreadPoolExecutor

        def one(i: int) -> Tuple[int, str]:
            resp = self._get(self.page_ocr_url(item, canvases[i], i + 1))
            time.sleep(self.delay)
            if resp.status_code != 200:
                return i, ""
            return i, hocr_to_text(resp.text)

        with ThreadPoolExecutor(max_workers=4) as pool:
            return dict(pool.map(one, sel))

    def _tesseract_ocr(self, item: str, canvases: List[dict], idxs: List[int],
                       lang: str, size: str, workers: int) -> Dict[int, str]:
        try:
            from .ocr_images import find_tesseract
            find_tesseract()
        except OCRUnavailable as e:
            raise RuntimeError(f"cannot OCR pages locally: {e}") from e

        def make(i: int):
            def run() -> str:
                url = self.page_image_url(canvases[i], size)
                resp = self._get(url)
                resp.raise_for_status()
                return ocr_image_bytes(resp.content, lang=lang)
            return run

        cache = Path(__file__).resolve().parent.parent / "data" / "raw" / f"mdz_{item}" / f"ocr_{lang}.json"
        done = ocr_pages([(str(i), make(i)) for i in idxs],
                         cache_path=str(cache), workers=workers)
        return {int(k): v for k, v in done.items()}

    def discover(self, query: str, limit: int = 50) -> List[str]:
        raise NotImplementedError(
            "mdz does not support discover(): find items at "
            "https://www.digitale-sammlungen.de and pass their bsb ids")
