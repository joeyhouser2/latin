"""Phase 6 run: read a whole catena manuscript and find the text no printed edition has.

    python scripts/catena_manuscript.py fetch      # page images (IIIF, half size), split into pages
    python scripts/catena_manuscript.py read       # Kraken, greek_minuscule_s9-12, on the GPU
    python scripts/catena_manuscript.py map        # coverage of every stretch by the printed texts

The manuscript is Jerusalem, Patriarchal Library, Hagios Stavros (Taphou) 25: catena commentary
on Acts and the Catholic Epistles, 9th/10th century, 347 folios, filmed for the Library of Congress
(public domain; credit "Library of Congress Collection of Manuscripts in the Greek Orthodox
Patriarchate of Jerusalem"). scripts/catena_htr_test.py measured the model on four of its pages:
9% line CER, matcher coverage 0.71 at half resolution.

``map`` slides a 50-word window (stride 25) over the transcription and asks, as the quotation
matcher does, what share of the window's words the printed texts have together: Cramer's catenae
(all of tlg4102, Acts and the Catholic Epistles among them), the open Fathers of phase 4, and the
biblical text (SBLGNT, for the lemmas). A known passage read by this model scores about 0.7; a
window under LOW that its neighbours share is a stretch no printed text here contains. Each such
stretch gets the nearest author label before it. Output: data/quotations/catena_ms_<id>.json.

Each frame of the film is an opening of two pages; ``fetch`` cuts it at the gutter.
"""
import argparse
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from ingest import htr                                                         # noqa: E402
from ingest.catenae import author_of                                           # noqa: E402
from ingest.quotations import RAW, Witness, norm, read_tei, score, tokenize    # noqa: E402

MS = "00279395529-jo"
NAME = "taphou25"
FRAMES = 360
IIIF = "https://tile.loc.gov/image-services/iiif/service:amed:amedmonastery:{ms}:{n:04d}/full/pct:50/0/default.jpg"
DIR = RAW / "manuscripts" / NAME
CACHE = DIR / "htr_greek_minuscule.json"
MODEL = ROOT / "models" / "htr" / "greek_minuscule_s9-12_NFC.mlmodel"
NT_DIR = RAW / "nt"
NT_BOOKS = ["65-Ac", "80-Jas", "81-1Pe", "82-2Pe", "83-1Jn", "84-2Jn", "85-3Jn", "86-Jud"]
NT_URL = "https://raw.githubusercontent.com/morphgnt/sblgnt/master/{book}-morphgnt.txt"
WINDOW, STRIDE, LOW = 50, 25, .25
UA = {"User-Agent": "latin-corpus-research/1.0"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def gutter(im):
    """x of the darkest column band in the middle fifth of an opening: the fold between the pages."""
    gray = im.convert("L")
    w, h = gray.size
    band = gray.crop((int(w * .4), int(h * .1), int(w * .6), int(h * .9))).resize((int(w * .2), 50))
    px = band.load()
    cols = [sum(px[x, y] for y in range(50)) for x in range(band.size[0])]
    smooth = [sum(cols[max(0, i - 3):i + 4]) for i in range(len(cols))]
    return int(w * .4) + min(range(len(smooth)), key=smooth.__getitem__)


def fetch(args):
    from io import BytesIO
    from PIL import Image
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "NOTICE.txt").write_text(
        "Jerusalem, Hagios Stavros 25, from the Library of Congress Collection of Manuscripts in the Greek "
        "Orthodox Patriarchate of Jerusalem (public domain). Half-size IIIF images, each film frame cut into "
        "its two pages.\n", encoding="utf-8")
    for n in range(args.first, args.last + 1):
        left, right = DIR / f"f{n:04d}_a.jpg", DIR / f"f{n:04d}_b.jpg"
        if left.exists() and right.exists():
            continue
        for attempt in range(3):
            try:
                im = Image.open(BytesIO(get(IIIF.format(ms=MS, n=n))))
                break
            except Exception as e:                                             # noqa: BLE001
                print(f"  frame {n}: {e}", flush=True)
                im = None
        if im is None:
            continue
        x = gutter(im)
        im.crop((0, 0, x, im.size[1])).save(left, quality=92)
        im.crop((x, 0, im.size[0], im.size[1])).save(right, quality=92)
        print(f"  frame {n}/{args.last}: gutter at {x}/{im.size[0]} | overall {n - args.first + 1}/{args.last - args.first + 1}",
              flush=True)
    for book in NT_BOOKS:
        path = NT_DIR / f"{book}.txt"
        if not path.exists():
            NT_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(get(NT_URL.format(book=book)))


def read(args):
    pages = sorted(str(p) for p in DIR.glob("f*_[ab].jpg"))
    print(f"{len(pages)} pages", flush=True)
    htr.transcribe_images(pages, cache_path=str(CACHE), model=str(MODEL), log=lambda s: print(s, flush=True))


def nt_text():
    words = []
    for book in NT_BOOKS:
        for line in (NT_DIR / f"{book}.txt").read_text(encoding="utf-8").splitlines():
            cols = line.split()
            if len(cols) >= 6:
                words.append(cols[5])                     # the word without punctuation
    return " ".join(words)


def printed():
    """One witness of everything printed and open: Cramer's catenae, the Fathers, the NT text."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mine_catenae", ROOT / "scripts" / "mine_catenae.py")
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    parts = read_tei(sorted((RAW / "witnesses").glob("tlg4102.*.xml")))
    for group in mc.FATHERS.values():
        parts += read_tei(sorted(mc.FATHERS_DIR.glob(f"{group}.*.xml")))
    parts.append((nt_text(), "NT"))
    return Witness("printed", parts)


def mapping(args):
    text = json.loads(CACHE.read_text(encoding="utf-8"))
    words, where = [], []                        # normalized words, and (page, line) of each
    originals = []
    for page in sorted(text):
        for li, line in enumerate(text[page].splitlines()):
            for original, normalized in tokenize(line):
                words.append(normalized)
                originals.append(original)
                where.append((page, li))
    print(f"{len(text)} pages, {len(words):,} words read", flush=True)
    base = printed()
    print(f"printed texts: {len(base.tok):,} words", flush=True)
    windows = []
    for start in range(0, max(1, len(words) - WINDOW + 1), STRIDE):
        windows.append((start, score(words[start:start + WINDOW], base)[0]))
    # Labels: an author's genitive read by the model (majuscule labels read by a minuscule model are
    # often garbled; author_of allows one wrong letter).
    labels = [(i, author_of(o)) for i, o in enumerate(originals) if author_of(o)]
    low = [(s, c) for s, c in windows if c < LOW]
    stretches = []
    for s, c in low:
        if stretches and s <= stretches[-1]["end"]:
            stretches[-1]["end"] = s + WINDOW
            stretches[-1]["coverage"] = min(stretches[-1]["coverage"], c)
        else:
            stretches.append({"start": s, "end": s + WINDOW, "coverage": c})
    for st in stretches:
        before = [a for i, a in labels if i <= st["start"] + 10]
        st["label"] = before[-1] if before else None
        st["page"], st["line"] = where[st["start"]]
        st["words"] = st["end"] - st["start"]
        st["text"] = " ".join(originals[st["start"]:st["end"]])
    covered = sum(c >= LOW for _, c in windows)
    print(f"windows: {len(windows):,}; covered by a printed text (>= {LOW}): {covered:,} ({covered / len(windows):.0%})")
    print(f"uncovered stretches: {len(stretches)}, {sum(s['words'] for s in stretches):,} words")
    out = ROOT / "data" / "quotations" / f"catena_ms_{NAME}.json"
    out.write_text(json.dumps({"windows": windows, "stretches": stretches}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--first", type=int, default=1)
    f.add_argument("--last", type=int, default=FRAMES)
    sub.add_parser("read")
    sub.add_parser("map")
    args = ap.parse_args()
    {"fetch": fetch, "read": read, "map": mapping}[args.cmd](args)


if __name__ == "__main__":
    main()
