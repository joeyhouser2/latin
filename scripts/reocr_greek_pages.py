"""Render specific PDF pages at high DPI and OCR them with Greek+Latin
Tesseract, caching results to JSON. Resumable -- skips pages already cached.

Usage:
    python scripts/reocr_greek_pages.py \
        --pdf data/raw/bub_gb_dSihnyKx6hgC_scan/page_scan.pdf \
        --pages data/raw/bub_gb_dSihnyKx6hgC_scan/pages_to_ocr.json \
        --out data/raw/bub_gb_dSihnyKx6hgC_scan/page_ocr_grclat.json \
        --tessdata models/tessdata
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")


def ocr_page(pix_path: str, tessdata_dir: str) -> str:
    out_base = pix_path.rsplit(".", 1)[0]
    env = dict(os.environ, TESSDATA_PREFIX=tessdata_dir)
    subprocess.run(
        ["tesseract", pix_path, out_base, "-l", "grc+lat", "--psm", "6"],
        env=env, check=True, capture_output=True,
    )
    with open(out_base + ".txt", encoding="utf-8") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pages", required=True, help="JSON list of 0-indexed page numbers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tessdata", default="models/tessdata")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    import fitz  # PyMuPDF

    with open(args.pages, encoding="utf-8") as f:
        pages = json.load(f)

    results = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} pages already cached")

    todo = [p for p in pages if str(p) not in results]
    print(f"=== OCR {len(todo)}/{len(pages)} pages (grc+lat) ===")

    doc = fitz.open(args.pdf)
    tessdata_abs = os.path.abspath(args.tessdata)

    with tempfile.TemporaryDirectory() as tmp:
        for i, p in enumerate(todo):
            pix = doc[p].get_pixmap(dpi=args.dpi)
            img_path = os.path.join(tmp, f"p{p}.png")
            pix.save(img_path)
            try:
                text = ocr_page(img_path, tessdata_abs)
            except subprocess.CalledProcessError as e:
                print(f"  [{p}] OCR FAILED: {e.stderr.decode(errors='replace')[:200]}")
                text = ""
            results[str(p)] = text
            if (i + 1) % 10 == 0 or i == len(todo) - 1:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(results, f)
                print(f"  {i+1}/{len(todo)} done, saved checkpoint", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"Done. {len(results)} pages total in {args.out}")


if __name__ == "__main__":
    main()
