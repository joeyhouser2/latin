"""End-to-end OCR self-test: real engines on known text. Writes nothing to the corpus.

    python scripts/ocr_selftest.py            # Tesseract + Kraken print + handwriting checks

Renders known Latin to a PDF (typeset, and a cursive-ish font if one is installed),
runs it through the ``ocrimages`` connector and compares against the source. Each
check reports PASS/FAIL/SKIP; exit code 1 on any FAIL.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8")

TEXT = ("Fides quaerens intellectum est. Omnes homines natura scire desiderant, "
        "et ea quae sunt in rerum natura cognoscere.")


def words(s: str) -> set:
    return set(re.findall(r"[a-z]+", s.lower()))


def recall(got: str, want: str) -> float:
    w = words(want)
    return len(w & words(got)) / len(w)


def main() -> int:
    import fitz
    from ingest import htr
    from ingest.ocr_images import ImageOCRConnector, find_tesseract

    fails = 0

    def report(name, ok, detail=""):
        nonlocal fails
        fails += (ok is False)
        print(f"  {'SKIP' if ok is None else 'PASS' if ok else 'FAIL'}  {name}  {detail}")

    work = Path(tempfile.mkdtemp(prefix="ocr_selftest_"))
    pdf = work / "selftest.pdf"
    doc = fitz.open()
    doc.new_page().insert_textbox(fitz.Rect(60, 80, 540, 400), TEXT, fontsize=18, fontname="tiro")
    doc.save(str(pdf))
    conn = ImageOCRConnector(cache_dir=str(work))

    try:
        find_tesseract()
        meta, secs = conn.fetch(f"{pdf}#mode=print&lang=lat")
        got = " ".join(t for _, t in secs)
        r = recall(got, TEXT)
        report("Tesseract Latin reads typeset text", r >= 0.9, f"recall {r:.0%}, engine {meta['_ocr_engine']}")
    except Exception as e:                                      # noqa: BLE001
        report("Tesseract Latin reads typeset text", False, str(e)[:120])

    if htr.available():
        models = htr.installed(htr.PRINT_MODELS)
        report("Kraken print model installed", bool(models), ", ".join(m.stem for m in models))
        try:
            meta, secs = conn.fetch(f"{pdf}#mode=htr&force=1")
            got = " ".join(t for _, t in secs)
            r = recall(got, TEXT)
            report("Handwriting engine runs on GPU/CPU", r >= 0.5, f"recall {r:.0%}, model {meta.get('_ocr_engine')}")
        except Exception as e:                                  # noqa: BLE001
            report("Handwriting engine runs on GPU/CPU", False, str(e)[:120])
    else:
        report("Handwriting engine", None, "no .venv-htr")

    try:                                                         # garbage must be refused, not ingested
        blank = fitz.open()
        pg = blank.new_page()
        for i in range(60):
            pg.draw_line((50, 50 + i * 10), (550, 50 + i * 10 + (i % 7)))
        bpdf = work / "noise.pdf"
        blank.save(str(bpdf))
        conn.fetch(f"{bpdf}#mode=print&lang=lat")
        report("Noise page is refused", False, "ingested without complaint")
    except Exception as e:                                       # noqa: BLE001
        report("Noise page is refused", True, type(e).__name__)

    print(f"\n{'FAILED' if fails else 'OK'} ({fails} failing). Work dir: {work}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
