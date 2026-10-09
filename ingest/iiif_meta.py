"""Read bibliographic metadata and page images out of IIIF manifests.

Handles both Presentation 2.x (``@value`` / ``@language`` lists) and 3.x
(language maps). Shared by the generic ``iiif`` connector and the page-image
OCR connector so a manifest is interpreted the same way everywhere.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_TAGS = re.compile(r"<[^>]+>")
_YEAR = re.compile(r"\b(1[0-9]{3})\b")
_CENTURY_ORD = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\s+century", re.IGNORECASE)
_ROMAN_CENTURY = re.compile(r"\b(?:s\.|saec\.?|saeculum)\s*([IVX]{1,5})\b", re.IGNORECASE)
_ROMAN = {"I": 1, "V": 5, "X": 10}

_LANG_NAMES = {
    "latin": "la", "lat": "la", "la": "la", "greek": "grc", "ancient greek": "grc",
    "grc": "grc", "gre": "grc", "german": "de", "ger": "de", "deu": "de",
    "french": "fr", "fre": "fr", "fra": "fr", "italian": "it", "ita": "it",
    "dutch": "nl", "english": "en", "eng": "en",
}


def lang_text(value: Any, prefer: str = "en") -> str:
    """Flatten any IIIF language-bearing value into one plain string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", _TAGS.sub("", value)).strip()
    if isinstance(value, list):
        # 2.x: [{"@language":..,"@value":..}] -- keep preferred language when present
        dicts = [v for v in value if isinstance(v, dict) and "@value" in v]
        if dicts:
            pref = [v for v in dicts if v.get("@language") == prefer]
            value = pref or dicts
        return "; ".join(filter(None, (lang_text(v, prefer) for v in value)))
    if isinstance(value, dict):
        if "@value" in value:
            return lang_text(value["@value"], prefer)
        if "@id" in value and len(value) <= 2:      # a linked resource, not text
            return ""
        for key in (prefer, "none", "@none", *value.keys()):    # 3.x language map
            if key in value:
                return lang_text(value[key], prefer)
    return ""


def metadata_fields(manifest: dict) -> Dict[str, List[str]]:
    """Label -> [values] from the manifest's ``metadata`` block."""
    fields: Dict[str, List[str]] = {}
    for entry in manifest.get("metadata") or []:
        label = lang_text(entry.get("label"))
        value = lang_text(entry.get("value"))
        if label and value:
            fields.setdefault(label, []).append(value)
    return fields


def find_field(fields: Dict[str, List[str]], *names: str) -> Optional[str]:
    """First value whose label matches any name (case-insensitive substring)."""
    for name in names:
        for label, values in fields.items():
            if name.lower() in label.lower():
                return values[0]
    return None


def _roman(token: str) -> int:
    total = 0
    for a, b in zip(token.upper(), token.upper()[1:] + " "):
        v = _ROMAN.get(a, 0)
        total += -v if _ROMAN.get(b, 0) > v else v
    return total


def century_of(*texts: Optional[str]) -> Optional[int]:
    """Best-effort century from catalogue date text ('s. XII', '1170', '12th century')."""
    for text in texts:
        if not text:
            continue
        m = _CENTURY_ORD.search(text)
        if m:
            return int(m.group(1))
        m = _ROMAN_CENTURY.search(text)
        if m:
            return _roman(m.group(1)) or None
        years = [int(y) for y in _YEAR.findall(text)]
        if years:
            return (years[0] - 1) // 100 + 1
    return None


def language_code(text: Optional[str], default: str = "la") -> str:
    if not text:
        return default
    codes = [_LANG_NAMES.get(t.strip().lower()) for t in re.split(r"[;,/]", text)]
    codes = [c for c in codes if c]
    if "la" in codes:
        return "la"
    if "grc" in codes:
        return "grc"
    return codes[0] if codes else default


def manifest_label(manifest: dict) -> Optional[str]:
    return lang_text(manifest.get("label")) or None


def manifest_license(manifest: dict) -> Optional[str]:
    lic = manifest.get("license") or manifest.get("rights")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    attribution = lang_text(manifest.get("attribution")) or \
        lang_text((manifest.get("requiredStatement") or {}).get("value"))
    bits = [b for b in (lic if isinstance(lic, str) else None, attribution) if b]
    return " -- ".join(bits) or None


def bibliographic_meta(manifest: dict) -> Dict[str, Any]:
    """title/author/century/language/shelfmark/license from a manifest."""
    f = metadata_fields(manifest)
    date = find_field(f, "date", "datation", "origin", "created", "issued")
    return {
        "title": find_field(f, "title", "titre") or manifest_label(manifest),
        "author": find_field(f, "author", "creator", "auteur", "by"),
        "century": century_of(date, manifest_label(manifest)),
        "language": language_code(find_field(f, "language", "langue")),
        "shelfmark": find_field(f, "shelfmark", "call number", "cote", "signature",
                                "shelf mark", "identifier"),
        "license": manifest_license(manifest),
    }


def download_manifest_pages(url: str, work_dir: Path, size: str,
                            pages_range, log=lambda m: print(m, file=sys.stderr)
                            ) -> Tuple[List[str], dict]:
    """Download the selected page images of a manifest; return (paths, manifest).

    ``pages_range`` is a function ``total -> range`` (0-based), so callers decide
    the selection once the page count is known. Already-downloaded pages are
    reused, so reruns are cheap.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from iiif_downloader import IIIFDownloader
    dl = IIIFDownloader()
    manifest = dl.get_manifest(url)
    infos = dl.extract_image_urls(manifest)
    if not infos:
        raise ValueError(f"no page images found in manifest {url}")
    work_dir.mkdir(parents=True, exist_ok=True)
    selection = list(pages_range(len(infos)))
    if not selection:
        raise ValueError(f"page selection is empty: the manifest has {len(infos)} "
                         f"page(s) -- {url}")
    paths: List[str] = []
    for i in selection:
        info = infos[i]
        dest = work_dir / f"page_{info['index']:04d}.jpg"
        if not dest.exists():
            if not dl.download_image(dl.build_image_url(info, size), dest):
                log(f"  could not download page {info['index']}")
                continue
            time.sleep(dl.delay)
        paths.append(str(dest))
    return paths, manifest
