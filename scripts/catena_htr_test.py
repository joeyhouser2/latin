"""Phase 6 test: can the installed Greek minuscule model read catena manuscripts well enough to mine?

    python scripts/catena_htr_test.py                    # download pages, run the models, score

The Patristic Text Archive transcribed one Severian excerpt (De fide et lege naturae) as copied in
five manuscripts of the Catena on Acts (data/raw/quotations/pta/*MsCat*.xml, CC BY-SA 4.0). Four
link each page to a IIIF crop of the manuscript image (Paris Coislin 25 and Grec 221 on Gallica,
Jerusalem Taphou 25 at the Library of Congress). The transcriptions are diplomatic, with line
breaks, so they are ground truth for the lines they cover; the crops also hold other catena text,
which is ignored.

Two scores per model:
  line CER   each transcribed line is matched to the model line that reads most like it;
             character error rate over all transcribed lines, accents and case removed
  coverage   the quotation matcher's question: what share of the transcription's words does the
             3-word shingle matcher find in the model's text (ingest.quotations.score)?

The pipeline needs coverage of about 0.5 or more on a known passage (phase 1's threshold) to find
quotations in a text. Images go to data/raw/quotations/pta/images/ (gitignored), the model output
to data/raw/bench/.
"""
import argparse
import json
import pathlib
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from ingest import htr                                   # noqa: E402
from ingest.quotations import RAW, Witness, norm, score, tokenize   # noqa: E402

PTA = RAW / "pta"
IMAGES = PTA / "images"
BENCH = ROOT / "data" / "raw" / "bench"
TEI = "{http://www.tei-c.org/ns/1.0}"
MODELS = ["greek_minuscule_s9-12_NFC.mlmodel", "catmus-medieval-1.6.0.mlmodel"]


def pages(path):
    """[(facs url, [line text])] for each page break with an image link."""
    root = ET.parse(path).getroot()
    out, current = [], None

    def walk(el):
        nonlocal current
        tag = el.tag.replace(TEI, "")
        if tag == "pb":
            current = None
            if el.get("facs"):
                current = (el.get("facs"), [""])
                out.append(current)
        elif tag == "lb" and current:
            current[1].append("")
        elif tag in ("note", "teiHeader"):   # marginal numbers and the header are not the text
            return el.tail or ""
        if current and el.text and tag not in ("pb", "lb"):
            current[1][-1] += el.text
        for child in el:
            tail = walk(child)
            if current and tail:
                current[1][-1] += tail
        return el.tail or ""

    walk(root.find(f"{TEI}text"))
    return [(url, [" ".join(l.split()) for l in lines if len(l.split()) >= 2]) for url, lines in out]


def fetch(url, dest):
    if dest.exists():
        return True
    req = urllib.request.Request(url, headers={"User-Agent": "latin-corpus-research/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                dest.write_bytes(r.read())
            return True
        except Exception as e:                                     # noqa: BLE001
            print(f"  download failed ({e}), retrying", flush=True)
            time.sleep(5 * (attempt + 1))
    return False


def bare(text):
    return " ".join(n for _, n in tokenize(text))


def cer(a, b):
    """Levenshtein distance / len(a)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)
    truth = {}                                      # image file -> (manuscript, transcribed lines)
    for path in sorted(PTA.glob("*MsCat*.xml")):
        ms = path.stem.split("-")[-1]
        for url, lines in pages(path):
            name = f"{ms}_{re.sub(r'[^0-9A-Za-z]+', '_', url.split('/iiif/')[-1])[:60]}.jpg"
            if lines and fetch(url, IMAGES / name):
                truth.setdefault(name, (ms, []))[1].extend(lines)
    print(f"{len(truth)} page images, {sum(len(v[1]) for v in truth.values())} transcribed lines")
    results = {}
    for m in args.models:
        out = htr.transcribe_images([str(IMAGES / n) for n in truth], cache_path=str(BENCH / f"catena_{m}.json"),
                                    model=str(ROOT / "models" / "htr" / m), log=lambda s: None)
        rows = []
        for name, (ms, lines) in truth.items():
            got = [bare(l) for l in out.get(name, "").splitlines() if bare(l)]
            errors = chars = 0
            for line in map(bare, lines):
                best = min((cer(line, g) for g in got), default=1.0)
                errors += best * len(line)
                chars += len(line)
            gt_tokens = [n for l in lines for _, n in tokenize(l)]
            coverage = score(gt_tokens, Witness("htr", [(out.get(name, ""), "1")]))[0]
            rows.append((ms, name, errors / max(1, chars), coverage, len(lines), len(got)))
            print(f"{m[:28]:28} {ms:10} line CER {errors / max(1, chars):5.0%}  coverage {coverage:4.2f}  "
                  f"({len(lines)} transcribed lines, {len(got)} read)", flush=True)
        total = sum(r[4] for r in rows)
        mean_cer = sum(r[2] * r[4] for r in rows) / max(1, total)
        mean_cov = sum(r[3] * r[4] for r in rows) / max(1, total)
        print(f"{m}: line CER {mean_cer:.0%}, coverage {mean_cov:.2f} (weighted by lines)\n")
        results[m] = {"line_cer": round(mean_cer, 3), "coverage": round(mean_cov, 3),
                      "pages": [{"ms": r[0], "image": r[1], "line_cer": round(r[2], 3), "coverage": r[3]}
                                for r in rows]}
    (ROOT / "data" / "htr_bench_catena.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")


if __name__ == "__main__":
    main()
