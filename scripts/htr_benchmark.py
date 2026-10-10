"""Compare recognition models on the same pages, scored by known-word rate.

    python scripts/htr_benchmark.py --out data/htr_bench.json

Each test set is a few page images of one script/style; each model transcribes
all of them and the expanded text is scored with ``abbrev.vocab_hit_rate`` (the
share of tokens that are real Latin words -- 0.75+ is readable, ~0.3 is noise).
The point is to learn which installed model suits which hand, so the IIIF
connector can pick by trial instead of guessing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8")

from ingest import abbrev, greek, htr                                   # noqa: E402

RAW = REPO / "data" / "raw"
SETS = {   # name -> (directory, page file names)
    "uncial-papyrus csg-226": ("iiif_dead7507bf", ["page_0012.jpg", "page_0020.jpg", "page_0030.jpg"]),
    "carolingian csg-195": ("iiif_f92189c0dd", ["page_0020.jpg", "page_0021.jpg", "page_0022.jpg"]),
    "carolingian csg-390": ("iiif_56e504f490", ["page_0030.jpg", "page_0031.jpg", "page_0032.jpg"]),
    "gothic-14c csg-192": ("bench_gothic_csg192", ["page_0060.jpg", "page_0061.jpg", "page_0062.jpg"]),
    "greek-10c pal-gr-23": ("bench_greek_pal23", ["page_0150.jpg", "page_0151.jpg", "page_0152.jpg"]),
    "print-1744 bsb": ("bench_print_bsb", ["page_0008.jpg", "page_0009.jpg", "page_0010.jpg"]),
}
PRINT_MODELS = ["catmus-print-fondue-large.mlmodel", "reichenau_lat_cat_099218.mlmodel",
                "catmus-medieval-1.6.0.mlmodel"]
GREEK_MODELS = ["greek_minuscule_s9-12_NFC.mlmodel", "catmus-medieval-1.6.0.mlmodel",
                "manicule-2026-latin_medieval.mlmodel"]
MODELS = ["catmus-medieval.mlmodel", "catmus-medieval-1.6.0.mlmodel",
          "manicule-2026-latin_medieval.mlmodel", "frolat_medieval_expan.mlmodel",
          "frolat_medieval_abbr.mlmodel"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/htr_bench.json")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    args = ap.parse_args()

    vocab = abbrev.build_vocab()
    results = {}
    for sname in args.sets:
        d, names = SETS[sname]
        paths = [str(RAW / d / n) for n in names if (RAW / d / n).exists()]
        if not paths:
            print(f"skip {sname}: no images"); continue
        is_greek = sname.startswith("greek")
        for m in (PRINT_MODELS if sname.startswith("print") else GREEK_MODELS if is_greek else args.models):
            model = REPO / "models" / "htr" / m
            cache = RAW / "bench" / f"{sname.split()[0]}_{sname.split()[-1]}_{m}.json"
            out = htr.transcribe_images(paths, cache_path=str(cache), model=str(model),
                                        log=lambda s: None)
            text = abbrev.expand_text("\n".join(out.values()), vocab)
            rate = (greek.hit_rate(text, greek.build_vocab()) if is_greek
                    else abbrev.vocab_hit_rate(text, vocab))
            results[f"{sname} | {m}"] = {"hit_rate": round(rate, 3), "sample": text[:160].replace("\n", " | ")}
            print(f"{sname:26s} {m:40s} {rate:.2f}  {text[:90]!r}", flush=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
