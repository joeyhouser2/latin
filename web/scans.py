"""Page images and transcriptions behind OCR'd documents, for the scan viewer.

A document made from scans records where its page images live in ``source``:
``[scan: <dir>|<cache file>]`` (``dir`` relative to ``data/raw`` or absolute). Older
documents have no stamp; for IIIF ones the folder is recoverable from the manifest
URL in ``source``. Nothing here touches corpus.db: corrections go to
``<dir>/corrections.json`` and are applied the next time the source is ingested
(``ingest.iiif.apply_corrections``).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data" / "raw"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

_SCAN_TAG = re.compile(r"\[scan: ([^|\]]*)\|?([^\]]*)\]")
_MANIFEST = re.compile(r"IIIF manifest (\S+)")
_ENGINE = re.compile(r"\[OCR: ([^\]]+)\]")


def locate(source: Optional[str]) -> Optional[Dict[str, Any]]:
    """Find a document's scan folder and transcription cache from its ``source``."""
    if not source:
        return None
    folder: Optional[Path] = None
    cache = ""
    m = _SCAN_TAG.search(source)
    if m:
        d = Path(m.group(1))
        folder = d if d.is_absolute() else RAW / d
        cache = m.group(2)
    else:
        m = _MANIFEST.search(source)
        if m:
            folder = RAW / ("iiif_" + hashlib.sha1(m.group(1).encode()).hexdigest()[:10])
    if folder is None or not folder.is_dir():
        return None
    eng = _ENGINE.search(source)
    return {"dir": folder, "cache": cache, "engine": eng.group(1) if eng else ""}


def _images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)


def _read_json(path: Path) -> Dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _read_json_any(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _guess_cache(folder: Path, engine: str) -> Optional[Path]:
    """For documents with no recorded cache: the fullest cache of the right kind."""
    pattern = "ocr_*.json" if engine in ("tesseract", "print", "ocr") else "htr_*.json"
    found = sorted((p for p in folder.glob(pattern) if not p.name.endswith(".lines.json")), key=lambda p: (len(_read_json(p)), p.stat().st_mtime),
                   reverse=True)
    return found[0] if found else None


class Scans:
    def __init__(self) -> None:
        self._vocab = None

    def _expand(self, raw: str) -> str:
        """Handwriting models give abbreviations as written; show what was ingested."""
        from ingest import abbrev
        if self._vocab is None:
            self._vocab = abbrev.build_vocab()
        return abbrev.expand_text(raw, self._vocab)

    def info(self, source: Optional[str]) -> Optional[Dict[str, Any]]:
        loc = locate(source)
        if not loc:
            return None
        folder: Path = loc["dir"]
        cache = folder / loc["cache"] if loc["cache"] else _guess_cache(folder, loc["engine"])
        raw = _read_json(cache) if cache and cache.exists() else {}
        fixes = _read_json(folder / "corrections.json")
        meta = _read_json(folder / "scan.json")
        if not meta.get("identifier"):
            m = _MANIFEST.search(source or "")
            if m:
                meta = {"identifier": m.group(1), "connector": "iiif"}
        pages = [{"name": p.name, "has_text": p.name in raw or p.name in fixes,
                  "corrected": p.name in fixes} for p in _images(folder)]
        return {"engine": loc["engine"], "cache": cache.name if cache else None,
                "identifier": meta.get("identifier"),
                "connector": meta.get("connector", "iiif"), "pages": pages,
                "corrected": sum(1 for p in pages if p["corrected"]),
                "transcribed": sum(1 for p in pages if p["has_text"])}

    def page(self, source: Optional[str], name: str) -> Optional[Dict[str, Any]]:
        loc = locate(source)
        if not loc or Path(name).name != name:
            return None
        folder: Path = loc["dir"]
        if not (folder / name).is_file():
            return None
        cache = folder / loc["cache"] if loc["cache"] else _guess_cache(folder, loc["engine"])
        raw = _read_json(cache).get(name, "") if cache and cache.exists() else ""
        raw = raw.replace("\r\n", "\n")          # Tesseract on Windows writes CRLF; the editor and saves use LF
        fixes = _read_json(folder / "corrections.json")
        expanded = self._expand(raw) if raw and loc["engine"] == "htr" else raw
        return {"name": name, "raw": raw, "ingested": expanded,
                "text": fixes.get(name, expanded), "corrected": name in fixes}

    def image_path(self, source: Optional[str], name: str) -> Optional[Path]:
        loc = locate(source)
        if not loc or Path(name).name != name or Path(name).suffix.lower() not in IMAGE_EXT:
            return None
        p = loc["dir"] / name
        return p if p.is_file() else None

    def save(self, source: Optional[str], name: str, text: str) -> Optional[Dict[str, Any]]:
        """Store a corrected page. Saving the engine's own text again removes the correction."""
        page = self.page(source, name)
        if page is None:
            return None
        folder: Path = locate(source)["dir"]                     # type: ignore[index]
        path = folder / "corrections.json"
        fixes = _read_json(path)
        text = text.replace("\r\n", "\n").strip("\n")
        if text.strip() == (page["ingested"] or "").strip():
            fixes.pop(name, None)
        else:
            fixes[name] = text
        fd, tmp = tempfile.mkstemp(dir=str(folder), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(fixes, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        return {"name": name, "corrected": name in fixes, "n_corrected": len(fixes)}

    # ---- line / word confidence (which parts of the page to double-check) ----
    def _cache_of(self, source: Optional[str]) -> Optional[tuple]:
        loc = locate(source)
        if not loc:
            return None
        folder: Path = loc["dir"]
        cache = folder / loc["cache"] if loc["cache"] else _guess_cache(folder, loc["engine"])
        return (loc, cache) if cache else None

    def lines(self, source: Optional[str], name: str, language: str = "la") -> Optional[Dict[str, Any]]:
        """Flagged lines and words for a page, or None if it has not been analysed yet."""
        from ingest import htr, linecheck
        found = self._cache_of(source)
        if not found or Path(name).name != name:
            return None
        loc, cache = found
        detail = _read_json_any(Path(htr.lines_path(str(cache)))).get(name)
        if detail is None:
            return None
        lang = "grc" if loc["engine"] == "htr-greek" else language
        vocab = self._greek_vocab() if lang == "grc" else self._latin_vocab()
        return linecheck.analyze(detail, vocab, lang)

    def analyze(self, source: Optional[str], name: str, language: str = "la") -> Optional[Dict[str, Any]]:
        """Compute confidence detail for one page (re-runs the recogniser; needs a free GPU
        for handwriting models, a second or two for Tesseract)."""
        from ingest import htr
        from ingest.ocr_images import ocr_words
        found = self._cache_of(source)
        if not found or Path(name).name != name:
            return None
        loc, cache = found
        folder: Path = loc["dir"]
        img = folder / name
        if not img.is_file():
            return None
        if cache.name.startswith("ocr_"):
            lang = cache.stem[len("ocr_"):] or "lat"
            detail = ocr_words(str(img), lang=lang)
            path = Path(htr.lines_path(str(cache)))
            store = _read_json_any(path)
            store[name] = detail
            path.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
        else:
            model = htr.MODEL_DIR / (cache.stem[len("htr_"):] + ".mlmodel")
            if not model.exists():
                raise RuntimeError(f"model {model.name} is not installed, cannot re-read this page")
            if htr.pick_gpu() is None and not os.environ.get("LATIN_GPU_PINNED"):
                raise RuntimeError("no GPU has enough free memory right now; try again when a job finishes")
            htr.transcribe_images([str(img)], cache_path=str(cache), model=str(model),
                                  log=lambda m: None, detail_only=True)
        return self.lines(source, name, language)

    # ---- line-level ground truth (the editor behind the training export) ----
    KINDS = ("text", "margin", "gloss", "noise")

    def _line_detail(self, source: Optional[str], name: str):
        """(loc, cache, raw detail list) for a page, or None before it has been analysed."""
        found = self._cache_of(source)
        if not found or Path(name).name != name:
            return None
        loc, cache = found
        from ingest import htr
        detail = _read_json_any(Path(htr.lines_path(str(cache)))).get(name)
        if not detail:
            return None
        return loc, cache, detail

    def line_texts(self, source: Optional[str], name: str) -> Optional[Dict[str, Any]]:
        """Every line of a page with the user's correction merged in."""
        got = self._line_detail(source, name)
        if got is None:
            return None
        loc, cache, detail = got
        store = _read_json_any(loc["dir"] / "line_corrections.json").get(name, {})
        lines = []
        for i, ln in enumerate(detail):
            rec = store.get(str(i))
            if rec and rec.get("b") != ln.get("b"):
                rec = None                                   # lines were recomputed since: not this line
            lines.append({"i": i, "t": rec["t"] if rec else ln["t"], "orig": ln["t"],
                          "c": ln.get("c", 0), "b": ln["b"],
                          "kind": rec.get("kind", "text") if rec else "text",
                          "ok": bool(rec and rec.get("ok")),
                          "edited": bool(rec and rec["t"] != ln["t"])})
        return {"lines": lines, "verified": sum(1 for x in lines if x["ok"]),
                "edited": sum(1 for x in lines if x["edited"])}

    def line_save(self, source: Optional[str], name: str, i: int, text: Optional[str] = None,
                  kind: Optional[str] = None, ok: Optional[bool] = None,
                  language: str = "la") -> Optional[Dict[str, Any]]:
        got = self._line_detail(source, name)
        if got is None or not 0 <= i < len(got[2]):
            return None
        loc, cache, detail = got
        if kind is not None and kind not in self.KINDS:
            raise ValueError(f"kind must be one of {', '.join(self.KINDS)}")
        path = loc["dir"] / "line_corrections.json"
        store = _read_json_any(path)
        page = store.setdefault(name, {})
        ln = detail[i]
        rec = page.get(str(i))
        if rec and rec.get("b") != ln.get("b"):
            rec = None
        rec = rec or {"t": ln["t"], "orig": ln["t"], "b": ln["b"], "kind": "text", "ok": False}
        if text is not None:
            rec["t"] = " ".join(text.splitlines()).strip()
        if kind is not None:
            rec["kind"] = kind
        if ok is not None:
            rec["ok"] = ok
        if rec["t"] == rec["orig"] and not rec["ok"] and rec["kind"] == "text":
            page.pop(str(i), None)                           # back to untouched
        else:
            page[str(i)] = rec
        if not page:
            store.pop(name, None)
        store["_meta"] = {"language": "grc" if loc["engine"] == "htr-greek" else language,
                          "cache": cache.name, "engine": loc["engine"]}
        fd, tmp = tempfile.mkstemp(dir=str(loc["dir"]), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(store, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        return self.line_texts(source, name)

    def line_image(self, source: Optional[str], name: str, i: int) -> Optional[bytes]:
        """PNG crop of one line, with the model's outline applied when it recorded one."""
        got = self._line_detail(source, name)
        if got is None or not 0 <= i < len(got[2]):
            return None
        loc, _cache, detail = got
        img = loc["dir"] / name
        if not img.is_file():
            return None
        import io
        from PIL import Image
        from training.htr_export import crop_line
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(img) as im:
            crop = crop_line(im, detail[i]["b"], detail[i].get("p"))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()

    def lines_to_page(self, source: Optional[str], name: str) -> Optional[Dict[str, Any]]:
        """Make the page's text from its lines: corrected text, noise lines left out."""
        got = self.line_texts(source, name)
        if got is None:
            return None
        loc = locate(source)
        raw = "\n".join(x["t"] for x in got["lines"] if x["kind"] != "noise" and x["t"].strip())
        text = self._expand(raw) if loc and loc["engine"] == "htr" else raw
        return self.save(source, name, text)

    def _latin_vocab(self):
        from ingest import abbrev
        if self._vocab is None:
            self._vocab = abbrev.build_vocab()
        return self._vocab

    def _greek_vocab(self):
        from ingest import greek
        return greek.build_vocab()
