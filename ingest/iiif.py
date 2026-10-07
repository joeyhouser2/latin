"""Generic IIIF connector: manuscripts and prints from any IIIF repository.

One connector for every library that publishes IIIF Presentation manifests --
DigiVatLib (Vatican), e-codices (Switzerland), the British Library, the Parker
Library, Bodleian, BnF, and the many collections that Biblissima aggregates.
It reads the manifest for bibliographic metadata, downloads the page images,
and turns them into text with whichever engine fits what is on the page:

``print``   Tesseract (Latin model) -- for printed books.
``htr``     handwritten-text recognition (``ingest.htr``) -- for manuscripts.
``auto``    (default) OCR three sample pages with Tesseract; if they come out
            as Latin it is print, otherwise it is treated as handwriting.

Tesseract on a manuscript yields confident gibberish, and an HTR model on print
is slow for no gain, so choosing by evidence matters. If a manuscript is
detected and no HTR engine is installed, ``fetch`` raises ``HTRUnavailable``
with setup instructions rather than ingesting noise.

Identifiers:
    https://.../manifest.json         any IIIF manifest URL
    vatlib:Vat.lat.3773               DigiVatLib shelfmark
    ecodices:csg-0390                 e-codices <library>-<number>
    bnf:ark:/12148/btv1b8432895r      Gallica ark (images may be bot-checked)
    bodleian:<id>
Options after ``#`` as ``key=value&key=value``: ``mode`` (auto|print|htr),
``pages`` (``a-b``, 1-based), ``size`` (IIIF width, default 2000 for print /
3000 for htr), ``lang``, ``psm``, ``workers``.

Honest limits: handwriting output from the default CATMuS model is *graphematic*
(abbreviations are not expanded), so it needs ``ingest.abbrev`` expansion before
it translates well; and ``discover()`` is unsupported -- find manifests in the
library's own catalogue.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import abbrev
from .base import Connector, RawWork
from .iiif_meta import bibliographic_meta, download_manifest_pages
from .ocr_images import (assemble, check_latin, german_function_word_rate,
                         german_or_latin, ocr_image, ocr_pages, parse_options, word_hit_rate,
                         parse_page_range, resolve_print_lang, spread)
from .pagetext import latin_function_word_rate
from .treatises import _stage_for

_REPO = Path(__file__).resolve().parent.parent
_PRINT_THRESHOLD = 0.03      # sample Latin-function-word rate above which it's print


class HTRUnavailable(RuntimeError):
    pass


class HTRQualityError(RuntimeError):
    """The recogniser's output is not Latin-like enough to ingest."""


_HTR_REFUSE_BELOW = 0.50     # vocabulary hit rate of the expanded text
_HTR_WARN_BELOW = 0.65


def resolve_manifest_url(identifier: str) -> str:
    """Expand a ``scheme:value`` shortcut into a manifest URL."""
    sys.path.insert(0, str(_REPO))
    import iiif_downloader as dl
    ident = identifier.strip()
    if ident.lower().startswith(("http://", "https://")):
        return ident
    scheme, _, value = ident.partition(":")
    scheme = scheme.lower()
    if scheme == "vatlib":
        return dl.get_vatican_manifest(value)
    if scheme == "bodleian":
        return dl.get_bodleian_manifest(value)
    if scheme == "bnf":
        return dl.get_bnf_manifest(value)
    if scheme == "ecodices":
        lib, _, num = value.partition("-")
        return dl.get_ecodices_manifest(lib, num)
    raise ValueError(
        f"unknown IIIF shortcut {identifier!r}; use a manifest URL or "
        "vatlib:/ecodices:/bnf:/bodleian: <id>")


class IIIFConnector(Connector):
    name = "iiif"

    def __init__(self, cache_dir: Optional[str] = None, strict: bool = True):
        self.cache_dir = Path(cache_dir) if cache_dir else _REPO / "data" / "raw"
        self.strict = strict

    def discover(self, query: str, limit: int = 50) -> List[str]:
        raise NotImplementedError(
            "iiif has no catalogue to search; find manifest URLs in the "
            "library's own catalogue (or via vatlib/ecodices shortcuts)")

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        base, opts = parse_options(identifier)
        url = resolve_manifest_url(base)
        mode = opts.get("mode", "auto")
        if mode not in ("auto", "print", "htr"):
            raise ValueError(f"mode must be auto|print|htr, not {mode!r}")
        lang = opts.get("lang", "auto")        # auto: choose lat vs Fraktur by trial
        psm = int(opts.get("psm", 3))
        workers = int(opts.get("workers", 4))
        want_size = opts.get("size")
        strict = self.strict and opts.get("force") not in ("1", "true")

        work_dir = self.cache_dir / ("iiif_" + hashlib.sha1(url.encode()).hexdigest()[:10])
        # printed pages need ~2000px; handwriting benefits from more line detail
        size = want_size or {"htr": "3000", "auto": "2500"}.get(mode, "2000")
        size = size + "," if size.isdigit() else size

        paths, manifest = download_manifest_pages(
            url, work_dir, size,
            lambda n: parse_page_range(opts.get("pages"), n))
        if not paths:
            raise ValueError(f"no page images could be downloaded from {url}")

        meta = {
            **{k: v for k, v in bibliographic_meta(manifest).items() if v},
            "source": f"IIIF manifest {url}",
            "has_existing_translation": False,
        }
        meta.setdefault("title", url)

        engine, pages, rate, stats = self._transcribe(
            paths, work_dir, mode, lang, psm, workers, strict)
        meta["_ocr_engine"] = engine
        meta["_ocr_latin_rate"] = rate
        meta["source"] += f" [OCR: {engine}]"
        if engine == "tesseract":
            meta["language"] = german_or_latin(stats) if "latin_rate" in stats \
                else meta.get("language", "la")
        if engine == "htr":
            meta["language_stage"] = "medieval"
        else:
            meta["language_stage"] = _stage_for(
                (meta["century"] * 100 - 50) if meta.get("century") else None)
        meta.update(meta_overrides)

        _, warn = check_latin("\n".join(pages), meta["title"], self.strict,
                              meta.get("language", "la"))
        if warn:
            print(f"  WARNING {warn}", file=sys.stderr)
        return meta, assemble(pages)

    # ---- engine choice ----------------------------------------------------
    def _transcribe(self, paths: List[str], work_dir: Path, mode: str, lang: str,
                    psm: int, workers: int, strict: bool = True
                    ) -> Tuple[str, List[str], float, dict]:
        """Returns (engine, per-page text, latin rate, print-model stats)."""
        key = lambda p: Path(p).name
        stats: dict = {}

        if mode == "auto":
            # try the print models on a few pages: Latin *or German* words mean
            # print (Tesseract); neither means handwriting (HTR)
            if lang == "auto":
                lang, stats = resolve_print_lang(paths, work_dir, psm, workers)
            else:
                sample = spread(paths, 3)
                got = ocr_pages(
                    [(key(p), (lambda p=p: ocr_image(p, lang=lang, psm=psm))) for p in sample],
                    cache_path=str(work_dir / f"ocr_{lang}.json"), workers=workers,
                    log=lambda m: None)
                text = "\n".join(got.values())
                stats = {"latin_rate": latin_function_word_rate(text),
                         "german_rate": german_function_word_rate(text)}
            rate = max(stats["latin_rate"], stats["german_rate"])
            mode = "print" if rate >= _PRINT_THRESHOLD else "htr"
            print(f"  sample word rate {rate:.1%} -> treating as "
                  f"{'print' if mode == 'print' else 'handwriting'}", file=sys.stderr)
        elif mode == "print" and lang == "auto":
            lang, stats = resolve_print_lang(paths, work_dir, psm, workers)

        from . import htr
        vocab = abbrev.build_vocab()

        if mode == "print":
            if lang == "auto":
                lang = "lat"
            tess_items = [(key(p), (lambda p=p: ocr_image(p, lang=lang, psm=psm))) for p in paths]
            tess_cache = str(work_dir / f"ocr_{lang}.json")
            kraken = htr.installed(htr.PRINT_MODELS) if htr.available() else []
            if kraken:
                # Trained print models usually out-read Tesseract on early type
                # (0.88 vs 0.68 known words on a 1744 Latin print): try both on a
                # few pages and keep the better.
                sample = spread(paths, 3)
                got = ocr_pages([(key(p), (lambda p=p: ocr_image(p, lang=lang, psm=psm)))
                                 for p in sample], cache_path=tess_cache,
                                workers=workers, log=lambda m: None)
                score = lambda t: word_hit_rate(t, vocab)
                tess_score = score("\n".join(got.values()))
                best, scores = htr.pick_model(sample, kraken, score, work_dir)
                stats["print_scores"] = {"tesseract": tess_score, **scores}
                if scores.get(best.name, 0.0) > tess_score:
                    print(f"  print engine: {best.stem} ({scores[best.name]:.0%}) beats "
                          f"Tesseract {lang} ({tess_score:.0%})", file=sys.stderr)
                    texts = htr.transcribe_images(
                        paths, cache_path=str(work_dir / f"htr_{best.stem}.json"),
                        model=str(best))
                    pages = [texts[key(p)] for p in paths]
                    stats["model"] = best.name
                    return ("kraken-print", pages,
                            latin_function_word_rate("\n".join(pages)), stats)
            texts = ocr_pages(tess_items, cache_path=tess_cache, workers=workers)
            pages = [texts[k] for k, _ in tess_items]
            return "tesseract", pages, latin_function_word_rate("\n".join(pages)), stats

        if not htr.available():
            raise HTRUnavailable(htr.SETUP_HINT)
        models = htr.installed(htr.HAND_MODELS) or [htr.MODEL]
        # which hand model reads this book best? try each on a few pages
        best, scores = htr.pick_model(
            spread(paths, 3), models,
            lambda t: abbrev.vocab_hit_rate(abbrev.expand_text(t, vocab), vocab), work_dir)
        stats["model"] = best.name
        stats["model_scores"] = scores
        cache = work_dir / f"htr_{best.stem}.json"
        texts = htr.transcribe_images(paths, cache_path=str(cache), model=str(best))
        # HTR is graphematic; expand abbreviations so the text can be translated
        # (the raw transcription stays in the cache file beside the page images)
        pages = [abbrev.expand_text(texts[key(p)], vocab) for p in paths]
        hit = abbrev.vocab_hit_rate("\n".join(pages), vocab)
        stats["htr_hit_rate"] = hit
        if hit < _HTR_REFUSE_BELOW and strict:
            raise HTRQualityError(
                f"HTR output is {hit:.0%} known Latin words (needs >= "
                f"{_HTR_REFUSE_BELOW:.0%}): none of the installed models "
                f"({', '.join(m.stem for m in models)}) reads this hand. They are "
                f"trained on Carolingian-and-later minuscule; uncial, rustic "
                f"capitals and papyrus defeat them (see scripts/htr_benchmark.py). "
                f"Raw transcription kept in {cache}. Re-run with #force=1 to "
                f"ingest anyway.")
        if hit < _HTR_WARN_BELOW:
            print(f"  WARNING HTR output is only {hit:.0%} known words -- expect "
                  f"many misreadings.", file=sys.stderr)
        return "htr", pages, latin_function_word_rate("\n".join(pages)), stats
