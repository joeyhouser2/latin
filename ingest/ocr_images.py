"""OCR page images with Tesseract, and a connector that ingests the result.

This is the piece that turns *scans* into corpus text. It is deliberately
generic -- it knows nothing about any one library -- so the same engine serves
the MDZ connector's fallback path (``ingest.mdz``), a folder of images from
``iiif_downloader.py``, or a IIIF manifest URL from any repository.

Engine facts worth knowing before trusting the output:

* It shells out to the ``tesseract`` binary and uses the ``lat`` model from
  ``models/tessdata`` (falling back to ``$TESSDATA_PREFIX`` / the install
  default). No Python OCR dependency is needed.
* Results are cached per page in a JSON file and the run is resumable: a
  400-page folio costs an hour of CPU, so an interrupted run must not restart.
* Tesseract's Latin model is trained on print. On **handwritten manuscripts it
  returns confident nonsense** -- that needs an HTR engine (Kraken/eScriptorium/
  Transkribus; see README "Manuscripts"). The engine can't tell print from
  handwriting, so the guard is a Latin function-word check: ``fetch`` raises
  when the OCR output does not look like Latin at all (``strict=True``, the
  default), which is what handwriting run through Tesseract produces.

Identifier forms for the ``ocrimages`` connector:
    path/to/dir                      images (jpg/png/tif) sorted by filename
    https://.../manifest.json        a IIIF manifest (2.x or 3.x); pages are
                                     downloaded to ``data/raw/ocr_<hash>/``
Options go after a ``#`` as ``key=value&key=value``: ``lang`` (tesseract
language, default ``lat``), ``psm`` (page segmentation mode, default 3),
``pages`` (``a-b``, 1-based inclusive), ``size`` (IIIF width, default 2000),
``workers``.

Usage:
    python scripts/ingest.py ocrimages data/raw/vat_lat_3773 --title "..."
    python scripts/ingest.py ocrimages "https://.../manifest.json#pages=1-40"
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .base import Connector, RawWork
from .pagetext import join_pages, latin_function_word_rate
from .treatises import TreatisesConnector

_REPO = Path(__file__).resolve().parent.parent
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
_WIN_DEFAULT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


class OCRUnavailable(RuntimeError):
    pass


def find_tesseract() -> str:
    exe = shutil.which("tesseract")
    if exe:
        return exe
    if _WIN_DEFAULT.exists():
        return str(_WIN_DEFAULT)
    raise OCRUnavailable("tesseract binary not found on PATH")


def find_tessdata(lang: str = "lat") -> Optional[str]:
    """Directory holding ``<lang>.traineddata`` for each requested language.

    Prefers the repo's ``models/tessdata`` (where the Latin/Greek models live);
    returns None to let tesseract use its own default.
    """
    first = lang.split("+")[0]
    for cand in (_REPO / "models" / "tessdata", os.environ.get("TESSDATA_PREFIX")):
        if cand and (Path(cand) / f"{first}.traineddata").exists():
            return str(cand)
    return None


def ocr_image(path: str, lang: str = "lat", psm: int = 3,
              tessdata: Optional[str] = None) -> str:
    """OCR one image file; returns the page text."""
    env = dict(os.environ)
    tessdata = tessdata or find_tessdata(lang)
    if tessdata:
        env["TESSDATA_PREFIX"] = tessdata
    proc = subprocess.run(
        [find_tesseract(), str(path), "stdout", "-l", lang, "--psm", str(psm)],
        env=env, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"tesseract failed on {path}: {proc.stderr.decode('utf-8', 'replace')[:300]}")
    return proc.stdout.decode("utf-8", "replace")


def ocr_image_bytes(data: bytes, suffix: str = ".jpg", **kw) -> str:
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return ocr_image(tmp, **kw)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def ocr_pages(items: Sequence[Tuple[str, Callable[[], str]]],
              cache_path: Optional[str] = None, workers: int = 4,
              log: Callable[[str], None] = lambda m: print(m, file=sys.stderr)
              ) -> Dict[str, str]:
    """OCR many pages in parallel with a resumable on-disk cache.

    ``items`` is ``[(page_key, produce_text), ...]`` where ``produce_text`` is
    a zero-arg callable doing the actual OCR (so callers decide where the image
    comes from -- a file, or a download). Returns ``{page_key: text}`` for
    every item. The cache is rewritten every few pages so a crash loses little.
    """
    cache: Dict[str, str] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    todo = [(k, fn) for k, fn in items if k not in cache]
    if cache:
        log(f"  OCR cache: {len(items) - len(todo)}/{len(items)} pages already done")
    if not todo:
        return {k: cache[k] for k, _ in items}

    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def save():
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            tmp = cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            os.replace(tmp, cache_path)

    def work(item):
        key, fn = item
        text = fn()
        with lock:
            cache[key] = text
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(todo):
                save()
                rate = done[0] / max(time.time() - t0, 1e-6)
                log(f"  OCR {done[0]}/{len(todo)} pages ({rate:.2f}/s)")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, todo))
    save()
    return {k: cache[k] for k, _ in items}


def parse_options(identifier: str) -> Tuple[str, Dict[str, str]]:
    """Split ``thing#k=v&k=v`` into (thing, {k: v})."""
    if "#" not in identifier:
        return identifier.strip(), {}
    base, frag = identifier.split("#", 1)
    opts = {}
    for part in frag.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            opts[k.strip()] = v.strip()
    return base.strip(), opts


def parse_page_range(spec: Optional[str], total: int) -> range:
    """'a-b' (1-based, inclusive) / 'a-' / '-b' / 'n' -> a 0-based range."""
    if not spec:
        return range(total)
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        lo = int(a) if a else 1
        hi = int(b) if b else total
    else:
        lo = hi = int(spec)
    return range(max(lo, 1) - 1, min(hi, total))


def assemble(pages: List[str], section_words: int = 1200) -> List[Tuple[str, str]]:
    return TreatisesConnector.split_sections(join_pages(pages), section_words)


def check_latin(text: str, what: str, strict: bool, language: str = "la") -> Tuple[float, str]:
    """Return (rate, warning). Raises if ``strict`` and text isn't Latin-like."""
    rate = latin_function_word_rate(text)
    if language != "la" or len(text.split()) < 200:
        return rate, ""
    if rate < 0.02:
        msg = (f"{what}: only {rate:.1%} of OCR tokens are common Latin function "
               f"words -- this is not Latin-like text. Wrong language model, "
               f"handwriting (needs HTR, not Tesseract), or a damaged scan.")
        if strict:
            raise ValueError(msg)
        return rate, msg
    if rate < 0.06:
        return rate, (f"{what}: low Latin function-word rate ({rate:.1%}); "
                      f"expect heavy OCR damage (long s, abbreviations, Fraktur).")
    return rate, ""


class ImageOCRConnector(Connector):
    """Ingest a folder of page images, or a IIIF manifest, via Tesseract."""

    name = "ocrimages"

    def __init__(self, cache_dir: Optional[str] = None, strict: bool = True):
        self.cache_dir = Path(cache_dir) if cache_dir else _REPO / "data" / "raw"
        self.strict = strict

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        base, opts = parse_options(identifier)
        lang = opts.get("lang", "lat")
        psm = int(opts.get("psm", 3))
        workers = int(opts.get("workers", 4))

        if base.lower().startswith(("http://", "https://")):
            key = hashlib.sha1(base.encode()).hexdigest()[:10]
            work_dir = self.cache_dir / f"ocr_{key}"
            size = opts.get("size", "2000")
            images, label = self._download_manifest(
                base, work_dir, size + "," if size.isdigit() else size, opts.get("pages"))
            title = label or base
        else:
            work_dir = Path(base)
            if not work_dir.is_dir():
                raise FileNotFoundError(f"not a directory or manifest URL: {base}")
            files = sorted(p for p in work_dir.iterdir()
                           if p.suffix.lower() in _IMAGE_EXT)
            if not files:
                raise FileNotFoundError(f"no page images in {work_dir}")
            sel = parse_page_range(opts.get("pages"), len(files))
            images = [str(files[i]) for i in sel]
            title = work_dir.name

        items = [(Path(p).name, (lambda p=p: ocr_image(p, lang=lang, psm=psm)))
                 for p in images]
        texts = ocr_pages(items, cache_path=str(work_dir / f"ocr_{lang}.json"),
                          workers=workers)
        pages = [texts[k] for k, _ in items]

        language = meta_overrides.get("language", "la")
        rate, warn = check_latin("\n".join(pages), title, self.strict, language)
        if warn:
            print(f"  WARNING {warn}", file=sys.stderr)

        meta = {
            "title": title,
            "source": f"Page images OCR'd with Tesseract ({lang}): {base}",
            "language": "la",
            "language_stage": "unknown",
            "license": "Depends on the image source -- check before redistributing",
            "has_existing_translation": False,
            "_ocr_latin_rate": rate,
        }
        meta.update(meta_overrides)
        return meta, assemble(pages)

    def _download_manifest(self, url: str, work_dir: Path, size: str,
                           pages: Optional[str]) -> Tuple[List[str], Optional[str]]:
        sys.path.insert(0, str(_REPO))
        from iiif_downloader import IIIFDownloader
        dl = IIIFDownloader()
        manifest = dl.get_manifest(url)
        infos = dl.extract_image_urls(manifest)
        if not infos:
            raise ValueError(f"no page images found in manifest {url}")
        sel = parse_page_range(pages, len(infos))
        work_dir.mkdir(parents=True, exist_ok=True)
        paths: List[str] = []
        for i in sel:
            info = infos[i]
            dest = work_dir / f"page_{info['index']:04d}.jpg"
            if not dest.exists():
                if not dl.download_image(dl.build_image_url(info, size), dest):
                    print(f"  could not download page {info['index']}", file=sys.stderr)
                    continue
                time.sleep(dl.delay)
            paths.append(str(dest))
        label = manifest.get("label")
        if isinstance(label, dict):                     # IIIF 3: {"en": ["..."]}
            vals = next(iter(label.values()), [])
            label = vals[0] if vals else None
        elif isinstance(label, list):
            label = label[0] if label else None
            if isinstance(label, dict):
                label = label.get("@value")
        return paths, (str(label) if label else None)
