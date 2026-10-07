"""Route a link to a digitized copy onto the connector that can read it.

Catalogue connectors (VD17/VD18, Europeana, Google Books) mostly know *about* a
work and only point at a scan held elsewhere. This module turns such a pointer
into ``(connector_name, identifier)`` when one of our full-text connectors can
actually read that host, and says so plainly when none can -- guessing would
mean ingesting a repository's HTML chrome as if it were Latin.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from .base import RawWork

_BSB = re.compile(r"(bsb\d{8})", re.IGNORECASE)
_IA = re.compile(r"archive\.org/(?:details|download|stream)/([^/?#]+)")
_GBOOKS = re.compile(r"books\.google\.[a-z.]+/.*[?&]id=([\w-]{12})")


class NoReadableCopy(RuntimeError):
    """None of the known digital copies is on a host we can read."""


def route(url: str) -> Optional[Tuple[str, str]]:
    """Map one URL to ``(connector_name, identifier)``, or None."""
    u = url.strip()
    low = u.lower()
    if "digitale-sammlungen.de" in low or "mdz-nbn-resolving.de" in low \
            or "bsb-muenchen" in low or "urn:nbn:de:bvb:12-bsb" in low:
        m = _BSB.search(u)
        if m:
            return "mdz", m.group(1).lower()
    m = _IA.search(u)
    if m:
        return "treatises", f"ia:{m.group(1)}"
    m = _GBOOKS.search(u)
    if m:
        return "treatises", f"ia:bub_gb_{m.group(1)}"
    if "manifest" in low and low.startswith("http"):
        return "iiif", u
    manifest = iiif_manifest_for(u)
    return ("iiif", manifest) if manifest else None


# Viewer-page patterns whose IIIF manifest URL is fixed by the host's own
# convention (each checked against the live host before being listed here).
_HEIDELBERG = re.compile(r"digi\.ub\.uni-heidelberg\.de/diglit/(?:iiif/)?(\w+)")
_VATLIB = re.compile(r"digi\.vatlib\.it/(?:view|iiif)/(MSS_[\w.\-]+)")
_GDZ = re.compile(r"resolver\.sub\.uni-goettingen\.de/purl\?(PPN\w+)", re.IGNORECASE)
_MPI = re.compile(r"dlc\.mpg\.de/(?:piresolver\?id=|api/v1/records/)([\w.\-]+)")
_GOOBI = re.compile(r"^(https?://[^/]+/viewer)/(?:content|image|api/v1/records)/(PPN\w+)")
_ECODICES = re.compile(r"e-codices\.(?:unifr\.ch|ch)/\w+/(?:list/one|description|thumbs)/(\w+)/(\w+)")


def iiif_manifest_for(url: str) -> Optional[str]:
    """Manifest URL for a viewer/landing-page URL on a host we know, else None."""
    for pat, build in (
        (_HEIDELBERG, lambda m: f"https://digi.ub.uni-heidelberg.de/diglit/iiif/{m.group(1)}/manifest.json"),
        (_VATLIB, lambda m: f"https://digi.vatlib.it/iiif/{m.group(1)}/manifest.json"),
        (_GDZ, lambda m: f"https://manifests.sub.uni-goettingen.de/iiif/presentation/{m.group(1)}/manifest"),
        (_MPI, lambda m: f"https://dlc.mpg.de/api/v1/records/{m.group(1)}/manifest"),
        (_GOOBI, lambda m: f"{m.group(1)}/api/v1/records/{m.group(2)}/manifest/"),
        (_ECODICES, lambda m: f"https://www.e-codices.unifr.ch/metadata/iiif/{m.group(1)}-{m.group(2)}/manifest.json"),
    ):
        m = pat.search(url)
        if m:
            return build(m)
    return None


def fetch_first_readable(urls: Iterable[str], options: str = "",
                         **meta_overrides) -> RawWork:
    """Fetch from the first URL that routes to a connector that can read it.

    ``options`` is appended as a ``#key=val`` fragment for connectors that take
    one (mdz, iiif). Tries the next candidate if one fails.
    """
    from .registry import get_connector     # late: registry imports us

    unroutable: List[str] = []
    errors: List[str] = []
    seen = set()
    for url in urls:
        target = route(url)
        if not target:
            unroutable.append(url)
            continue
        if target in seen:              # several links can resolve to one copy
            continue
        seen.add(target)
        name, ident = target
        if options and name in ("mdz", "iiif"):
            ident = f"{ident}#{options.lstrip('#')}"
        try:
            return get_connector(name).fetch(ident, **meta_overrides)
        except Exception as e:                           # noqa: BLE001
            errors.append(f"{name}:{ident} -> {e}")
    detail = "; ".join(errors) if errors else "no link points at a host we can read"
    raise NoReadableCopy(
        f"{detail}. Unreadable links: {', '.join(unroutable) or '(none)'}. "
        "Download the scan by hand and ingest it with `ocrimages`.")
