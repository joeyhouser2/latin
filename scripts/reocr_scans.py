"""Re-OCR an archive.org scan whose own OCR lost the Greek, with Tesseract's Ancient Greek model.

    python scripts/reocr_scans.py diefragmentederv01diel_0 --lang grc+deu
    python scripts/reocr_scans.py aristotelisquif00arisgoog --lang grc+lat --workers 12

archive.org's OCR read the Greek of many 19th-century editions as Latin letters (Rose's
Aristotle fragments, Diels's Vorsokratiker, Usener's Epicurea). Tesseract's tessdata_best "grc"
model reads the same pages well. Page images are fetched one at a time from archive.org and
discarded after OCR; only the text is kept, one file per page, so a rerun resumes.

Models: models/tessdata/{grc,lat,deu}.traineddata from github.com/tesseract-ocr/tessdata_best
(models/ is gitignored). Output: data/raw/quotations/philosophy/reocr/<id>/NNNN.txt and the
joined <id>.txt beside it, with "=== page N" lines between pages.

PRIVATE RESEARCH USE: the text is for this library's baselines and is not redistributed.
"""
import argparse
import concurrent.futures as cf
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESSDATA = ROOT / "models" / "tessdata"
OUT = ROOT / "data" / "raw" / "quotations" / "philosophy" / "reocr"
HEADERS = {"User-Agent": "latin-research"}


def page_count(ident):
    meta = requests.get(f"https://archive.org/metadata/{ident}", headers=HEADERS, timeout=60).json()["metadata"]
    if meta.get("access-restricted-item") in (True, "true"):
        sys.exit(f"{ident} is lending-only on archive.org: not used.")
    return int(meta["imagecount"])


def ocr_page(ident, n, lang, directory):
    out = directory / f"{n:04d}.txt"
    if out.exists():
        return n, "kept"
    for attempt in range(4):
        try:
            r = requests.get(f"https://archive.org/download/{ident}/page/n{n}.jpg", headers=HEADERS, timeout=180)
            r.raise_for_status()
            break
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    else:
        return n, "download failed"
    with tempfile.TemporaryDirectory() as tmp:
        image = pathlib.Path(tmp) / "page.jpg"
        image.write_bytes(r.content)
        env = dict(os.environ, OMP_THREAD_LIMIT="1")
        done = subprocess.run(["tesseract", str(image), str(pathlib.Path(tmp) / "page"), "--tessdata-dir", str(TESSDATA),
                               "-l", lang, "--psm", "3"], capture_output=True, env=env, timeout=600)
        text = pathlib.Path(tmp) / "page.txt"
        if done.returncode or not text.exists():
            return n, f"tesseract failed: {done.stderr[-200:]!r}"
        out.write_text(text.read_text(encoding="utf-8"), encoding="utf-8")
    return n, "done"


def join(ident, directory, pages):
    parts = []
    for n in range(pages):
        page = directory / f"{n:04d}.txt"
        if page.exists():
            parts.append(f"=== page {n}\n" + page.read_text(encoding="utf-8"))
    (OUT / f"{ident}.txt").write_text("\n".join(parts), encoding="utf-8")
    return len(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ident", help="archive.org identifier")
    ap.add_argument("--lang", default="grc+lat", help="Tesseract languages, e.g. grc+deu")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, help="default: the last page")
    args = ap.parse_args()
    pages = page_count(args.ident)
    directory = OUT / args.ident
    directory.mkdir(parents=True, exist_ok=True)
    todo = range(args.first, min(pages, (args.last if args.last is not None else pages - 1) + 1))
    print(f"{args.ident}: {pages} pages, OCR {len(todo)} with {args.lang} on {args.workers} workers", flush=True)
    failed = []
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for i, (n, status) in enumerate(pool.map(lambda n: ocr_page(args.ident, n, args.lang, directory), todo), 1):
            if status not in ("done", "kept"):
                failed.append((n, status))
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} pages | overall {i}/{len(todo)}", flush=True)
    print(f"joined {join(args.ident, directory, pages)} pages into {OUT / (args.ident + '.txt')}; failed: {failed}")


if __name__ == "__main__":
    main()
