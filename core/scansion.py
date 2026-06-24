"""Metrical scansion of verse lines (optional, CLTK-backed).

Scans a Latin verse line into its long/short pattern and reports the detected
metre. This serves two ends: a reading/discovery feature in its own right (show
the reader the metrical scheme), and context for the verse Stylizer presets (the
scansion is fed to the model so its English rendering can echo the original
movement).

CLTK's Latin prosody is solid for hexameter and the pentameter half of elegiac
couplets. Greek prosody support is far weaker, so Greek currently returns an
empty (best-effort) result rather than a wrong one. Everything degrades
gracefully: if CLTK isn't installed the functions return ``ScanResult``s with
``scansion=None`` and a note, never raising.

CLTK and its models are heavy and optional, so the import is lazy and cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ScanResult:
    """Outcome of scanning one verse line."""

    line: str
    scansion: Optional[str] = None     # e.g. "¯ ˘ ˘ ¯ ¯ ¯ ˘ ˘ ¯ ˘ ˘ ¯ ˘"
    meter: Optional[str] = None        # "hexameter" | "pentameter" | ...
    valid: bool = False                # did the scanner find a metrically valid scan?
    note: Optional[str] = None         # diagnostic when scansion is None/invalid


# Lazily-built CLTK scanners, keyed by meter name. None once an attempt failed.
_SCANNERS: dict = {}
_CLTK_OK: Optional[bool] = None


def _scanner(meter: str):
    """Return a cached CLTK scanner for ``meter`` (or None if unavailable)."""
    global _CLTK_OK
    if _CLTK_OK is False:
        return None
    if meter in _SCANNERS:
        return _SCANNERS[meter]
    try:
        if meter == "hexameter":
            from cltk.prosody.lat.hexameter_scanner import HexameterScanner
            scanner = HexameterScanner()
        elif meter == "pentameter":
            from cltk.prosody.lat.pentameter_scanner import PentameterScanner
            scanner = PentameterScanner()
        else:
            scanner = None
        _CLTK_OK = True
        _SCANNERS[meter] = scanner
        return scanner
    except Exception as exc:  # pragma: no cover - depends on optional install
        _CLTK_OK = False
        _SCANNERS[meter] = None
        print(f"CLTK scansion unavailable ({exc}); skipping scansion.")
        return None


def scan_line(line: str, lang: str = "la", meter: str = "hexameter") -> ScanResult:
    """Scan one verse line. Never raises; returns a ScanResult with a note on
    failure. ``lang`` is ISO-ish: 'la' Latin, 'grc' Greek (best-effort/None)."""
    text = (line or "").strip()
    if not text:
        return ScanResult(line=line, note="empty line")
    if lang == "grc":
        return ScanResult(line=line, meter=meter,
                          note="Greek scansion not supported yet")

    scanner = _scanner(meter)
    if scanner is None:
        return ScanResult(line=line, meter=meter, note="cltk unavailable")
    try:
        verse = scanner.scan(text)
    except Exception as exc:  # pragma: no cover - input-dependent
        return ScanResult(line=line, meter=meter, note=f"scan error: {exc}")
    scansion = (getattr(verse, "scansion", "") or "").strip() or None
    return ScanResult(
        line=line,
        scansion=scansion,
        meter=meter,
        valid=bool(getattr(verse, "valid", False)),
        note=None if scansion else "no valid scan found",
    )


def scan_lines(
    lines: List[str], lang: str = "la", meter: str = "hexameter"
) -> List[ScanResult]:
    """Scan a sequence of verse lines.

    For ``meter='elegiac'`` the couplet alternates: odd lines (1st, 3rd, ...) are
    hexameters and even lines pentameters. Otherwise every line is scanned with
    the single given metre.
    """
    results: List[ScanResult] = []
    for i, line in enumerate(lines):
        if meter == "elegiac":
            line_meter = "hexameter" if i % 2 == 0 else "pentameter"
        else:
            line_meter = meter
        results.append(scan_line(line, lang=lang, meter=line_meter))
    return results
