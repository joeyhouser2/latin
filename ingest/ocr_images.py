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
from .iiif_meta import download_manifest_pages, manifest_label
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


_GERMAN_WORDS = frozenset(
    "und der die das den dem des ein eine einer eines ist nicht auf fur für sich auch als "
    "wie bey bei von mit zu zum zur im am dass daß oder aber wird werden hat haben "
    "sind sein ihr seine nach vor über auch noch nur wenn dann so er sie es wir "
    "ich ihm ihn dieser diese dieses einem welche welcher denen unter durch gegen".split())
_TOK = __import__("re").compile(r"[^\W\d_]{2,}", __import__("re").UNICODE)


def german_function_word_rate(text: str) -> float:
    toks = [t.lower().replace("ſ", "s") for t in _TOK.findall(text)]
    return sum(t in _GERMAN_WORDS for t in toks) / len(toks) if toks else 0.0


def spread(items: List[str], n: int) -> List[str]:
    """n items spread evenly through the list, skipping covers and flyleaves."""
    if len(items) <= n:
        return list(items)
    lo, hi = min(2, len(items) // 4), len(items) - 1 - min(2, len(items) // 4)
    step = (hi - lo) / (n - 1) if n > 1 else 0
    return [items[round(lo + i * step)] for i in range(n)]


def word_hit_rate(text: str, vocab) -> float:
    """Share of alphabetic tokens that are known Latin or German words.

    Used to tell which Tesseract model reads a book's type: the right model
    produces real words, the wrong one produces letter-soup that misses both.
    """
    toks = [t.lower().replace("ſ", "s").translate(_UV) for t in _TOK.findall(text) if len(t) >= 3]
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if vocab.get(t, 0) > 0 or t in _GERMAN_WORDS)
    return hits / len(toks)


_UV = str.maketrans({"v": "u", "j": "i"})
PRINT_LANGS = ("lat", "Fraktur")


def resolve_print_lang(paths: Sequence[str], work_dir: Path, psm: int = 3,
                       workers: int = 4, log=lambda m: print(m, file=sys.stderr)
                       ) -> Tuple[str, Dict[str, float]]:
    """Pick the Tesseract model for a book by trying each on a few sample pages.

    Roman type wants ``lat``; German-region prints of the 16th-18th c. are often
    in Fraktur/Schwabacher (even their Latin passages), which ``lat`` reads as
    garbage. Returns ``(lang, stats)``; ``stats`` carries the per-model word-hit
    rate plus the Latin and German function-word rates of the winning text.
    OCR of the sample is cached, so choosing costs nothing on the real run.
    """
    from . import abbrev
    vocab = abbrev.build_vocab()
    sample = spread(list(paths), 3)
    scores: Dict[str, float] = {}
    texts: Dict[str, str] = {}
    for lang in PRINT_LANGS:
        got = ocr_pages(
            [(Path(p).name, (lambda p=p, lang=lang: ocr_image(p, lang=lang, psm=psm)))
             for p in sample],
            cache_path=str(work_dir / f"ocr_{lang}.json"), workers=workers, log=lambda m: None)
        texts[lang] = "\n".join(got.values())
        scores[lang] = word_hit_rate(texts[lang], vocab)
    best = max(PRINT_LANGS, key=lambda l: scores[l])
    stats = {f"hit_{l}": round(scores[l], 3) for l in PRINT_LANGS}
    stats["latin_rate"] = max(latin_function_word_rate(t) for t in texts.values())
    stats["german_rate"] = max(german_function_word_rate(t) for t in texts.values())
    log(f"  print model: {best}  (word-hit rates {stats['hit_lat']:.0%} lat vs "
        f"{stats['hit_Fraktur']:.0%} Fraktur; Latin words {stats['latin_rate']:.1%}, "
        f"German words {stats['german_rate']:.1%})")
    return best, stats


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


def german_or_latin(stats: Dict[str, float]) -> str:
    """'de' when a book's sampled text is clearly German rather than Latin."""
    if stats.get("german_rate", 0) > 1.5 * stats.get("latin_rate", 0) and             stats.get("german_rate", 0) >= 0.05:
        return "de"
    return "la"


def check_latin(text: str, what: str, strict: bool, language: str = "la") -> Tuple[float, str]:
    """Return (rate, warning). Raises if ``strict`` and text isn't Latin-like."""
    rate = latin_function_word_rate(text)
    n = len(text.split())
    if language == "la" and strict and (n == 0 or (n >= 3 and rate == 0.0)):
        # too short for the statistics below, but nothing Latin at all: blank or noise
        raise ValueError(f"{what}: OCR found {'no text' if n == 0 else 'no Latin words'} "
                         f"({n} tokens). Blank pages, wrong language model or a damaged scan.")
    if language != "la" or n < 200:
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


def render_pdf_pages(pdf: Path, out_dir: Path, dpi: int = 250,
                     pages: Optional[str] = None) -> List[str]:
    """Rasterise a PDF to ``page_NNNN.jpg`` files (skipping ones already there).

    Hand-downloaded scans are usually PDFs. ``pages`` is the usual 1-based
    ``a-b`` range. Uses PyMuPDF, which is already a dependency of the Greek
    re-OCR script.
    """
    import fitz  # PyMuPDF
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf))
    paths: List[str] = []
    for i in parse_page_range(pages, len(doc)):
        dest = out_dir / f"page_{i + 1:04d}.jpg"
        if not dest.exists():
            doc[i].get_pixmap(dpi=dpi).save(str(dest))
        paths.append(str(dest))
    return paths


class ImageOCRConnector(Connector):
    """Ingest scans you downloaded by hand: a PDF, a folder of images, or a manifest.

    For sources we cannot fetch automatically (see ``ingest.ocr_notes``): save the
    PDF or images, then point this connector at the file. It uses the same engine
    selection as ``iiif`` -- Tesseract Latin or Fraktur, a Kraken print model, or
    handwriting recognition, whichever reads the pages best.

    Identifiers:  ``path/to/book.pdf``, ``path/to/folder``, or a manifest URL.
    Options (after ``#``): ``pages=1-40``, ``mode=print|htr``, ``lang=lat|Fraktur``,
    ``dpi=250`` (PDF rendering), ``force=1`` to ingest despite a quality refusal.
    """

    name = "ocrimages"

    def __init__(self, cache_dir: Optional[str] = None, strict: bool = True):
        self.cache_dir = Path(cache_dir) if cache_dir else _REPO / "data" / "raw"
        self.strict = strict

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        from .iiif import IIIFConnector
        base, opts = parse_options(identifier)
        iiif = IIIFConnector(cache_dir=str(self.cache_dir), strict=self.strict)
        if base.lower().startswith(("http://", "https://")):
            return iiif.fetch(identifier, **meta_overrides)

        path = Path(base)
        strict = self.strict and opts.get("force") not in ("1", "true")
        if path.is_file() and path.suffix.lower() == ".pdf":
            key = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
            work_dir = self.cache_dir / f"pdf_{key}"
            images = render_pdf_pages(path, work_dir, int(opts.get("dpi", 250)),
                                      opts.get("pages"))
            title = path.stem
        elif path.is_dir():
            work_dir = path
            files = sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_EXT)
            if not files:
                raise FileNotFoundError(f"no page images in {path}")
            images = [str(files[i]) for i in parse_page_range(opts.get("pages"), len(files))]
            title = path.name
        else:
            raise FileNotFoundError(f"not a PDF, folder or manifest URL: {base}")
        if not images:
            raise ValueError(f"no pages selected from {base}")

        from .iiif import write_scan_info
        write_scan_info(work_dir, identifier, "ocrimages")
        meta = {
            "title": title,
            "source": f"Local scan: {path.name}",
            "language": "la",
            "language_stage": "unknown",
            "license": "Depends on the image source -- check before redistributing",
            "has_existing_translation": False,
        }
        return iiif.process_pages(images, work_dir, meta, opts, meta_overrides, strict)
