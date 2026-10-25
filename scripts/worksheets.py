"""Printable line-correction worksheets, and loading the typed-up results back.

    python scripts/worksheets.py pool                                        # what material exists
    python scripts/worksheets.py make --diverse 24 [--analyze] [--blind] [--lines-per-page 12] [--name S1]
    python scripts/worksheets.py make --doc 13402 --pages 20-29 [--analyze] [--blind] [--name S1]
    python scripts/worksheets.py import --name S1 results.docx [--apply]

``make`` writes ``data/worksheets/<name>/sheet.html`` (open it, Print -> Save as PDF or print
on paper) and ``manifest.json``. Each sheet page has a reduced picture of the manuscript page,
then one row per line: an ID like ``P007-L03``, the cropped line image, the model's reading
(omitted with ``--blind``) and a blank to write in.

``--diverse N`` takes N pages mixed from *every* scan folder on this computer: script classes
(Latin hand, Latin print, Greek hand, Fraktur) are interleaved, no single book supplies more
than a third, and within each book half the pages are spread evenly and half are its
least-confident pages. Sheets are ordered so a printed stack alternates hands.
``--analyze`` reads pages that have no line detail yet (handwriting wants a free GPU).

Type the results into Word (or any text file), one line per row:

    P007-L03 =                    the model's reading is right as printed
    P007-L04 quae sunt in rerum   corrected text (the whole line)
    P007-L05 M: uesti             a margin note, with its text
    P007-L06 G: tubas             an interlinear gloss, with its text
    P007-L07 X                    not text (neumes, ornament, bleed-through)
    P007-L08 ?                    can't read it / skip (nothing is recorded)

``import`` reads a .docx (or .txt) and records every row in the scan folders'
``line_corrections.json`` -- the store the viewer's Lines tab uses, so rows show up there and
in ``python -m training.htr_export``. Without ``--apply`` it only reports what it would do.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import sqlite3
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = REPO / "data" / "worksheets"
KINDS = {"M": "margin", "G": "gloss", "X": "noise"}
ROW = re.compile(r"^\s*(P\d+-L\d+)\s*(.*?)\s*$", re.I)
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

KEY_HTML = ("<p><b>How to fill in</b> &mdash; one line per row, e.g. <code>P001-L03 =</code> (reading is right), "
            "<code>P001-L04 corrected text</code>, <code>P001-L05 M: margin text</code>, "
            "<code>P001-L06 G: gloss text</code>, <code>P001-L07 X</code> (not text), "
            "<code>P001-L08 ?</code> (unreadable).</p>")
STYLE = """
 @page { size: A4; margin: 12mm; }
 body { font: 11pt Georgia, serif; color: #000; }
 h2 { margin: 0 0 2mm; } h3 { margin: 0 0 3mm; }
 .meta { margin: 0 0 4mm; color: #555; font-size: 9pt; }
 section { break-after: page; }
 .thumb { max-width: 100%; max-height: 225mm; }
 table { width: 100%; border-collapse: collapse; }
 td { border-bottom: 1px solid #999; padding: 2mm 1mm; vertical-align: top; }
 td.id { width: 22mm; font: bold 10pt monospace; }
 td img { max-width: 100%; max-height: 14mm; display: block; }
 .model { font-size: 10pt; color: #333; margin-top: 1mm; }
 .blank { height: 9mm; border-bottom: 1px dotted #777; margin-top: 2mm; }
 .key { font-size: 9pt; border: 1px solid #999; padding: 2mm; margin-bottom: 4mm; }
 tr { break-inside: avoid; }
 .card h1 { font-size: 20pt; margin: 0 0 2mm; }
 .card .samples img { display: block; width: 100%; max-height: 28mm; object-fit: contain; margin: 2mm 0;
                      border-bottom: 1px solid #ccc; }
 .card li { margin-bottom: 2mm; }
"""


def _doc_row(doc_id: int):
    con = sqlite3.connect(f"file:{REPO / 'data' / 'corpus.db'}?mode=ro", uri=True)
    try:
        row = con.execute("select source, language, title from documents where id=?", (doc_id,)).fetchone()
    finally:
        con.close()
    if not row:
        raise SystemExit(f"no document {doc_id}")
    return row



# ---------------------------------------------------------------------------
# writing styles: a card per style, and which folder belongs to which
# ---------------------------------------------------------------------------
STYLE_HINTS = [                      # substring of folder name / manifest identifier -> style
    ("dead7507bf", "uncial"), ("csg-0226", "uncial"),
    ("f92189c0dd", "carolingian"), ("csg-0195", "carolingian"),
    ("bench_gothic", "gothic"), ("csg-0192", "gothic"),
    ("pal23", "greek-min"), ("cpgraec23", "greek-min"),
]
CLASS_STYLE = {"latin-print": "print", "fraktur-print": "fraktur", "greek-hand": "greek-min",
               "greek-print": "greek-print", "latin-hand": "hand-other"}
STYLE_ORDER = ["uncial", "carolingian", "gothic", "hand-other", "greek-min", "print", "fraktur", "greek-print"]
STYLES: Dict[str, Dict[str, Any]] = {
    "uncial": {
        "title": "Uncial / half-uncial on papyrus \u2014 St Gall Cod. Sang. 226",
        "when": "written after 650, southern France; the only papyrus codex at St Gall",
        "blind": True,
        "look": ["A formal-uncial base (rounded capitals, little ascender/descender contrast) mixed with minuscule-looking e, r, h, n: expect letters that are neither one script nor the other.",
                 "Rounded forms: A like a rounded, open u-shape; D with a curved back; round E with a short bar; "
                 "M with curved sides; U/V are one letter.",
                 "Words run together with little or no spacing; a bar or colon-like mark stands for -bus (B;) and -que (Q;).",
                 "Papyrus: fibre stripes, cracks and holes look like strokes. Lines follow the horizontal fibres.",
                 "The reading shown on other sheets is useless here (about 30% of words are real), so rows are blank to transcribe."],
    },
    "carolingian": {
        "title": "Carolingian minuscule \u2014 St Gall Cod. Sang. 195",
        "when": "9th century (Ferrandus, De duce religioso)",
        "look": ["Clear, rounded minuscule; ascenders (b d h l) end in small clubs or wedges.",
                 "Words are separated; the model reads this hand best (about 80% known words).",
                 "Watch for the ct and st ligatures, the Tironian \u2018et\u2019 (a 7-like sign) and suspension strokes over a letter.",
                 "Interlinear glosses and marginal notes are common and are not part of the text."],
    },
    "gothic": {
        "title": "Gothic (textualis) hand",
        "when": "later medieval; folio sample from a St Gall manuscript used as the Gothic benchmark",
        "look": ["Compressed, angular letters with broken strokes; i, m, n, u are runs of identical upright strokes (minims).",
                 "Heavy abbreviation: signs for con-/cum-, per, pro, quod, and a bar over a vowel for a nasal.",
                 "Round s at word end, long s inside the word; r after o is often a 2-shape.",
                 "Two columns and rubrics (red headings) are likely; small initials may be skipped by the segmenter."],
    },
    "hand-other": {
        "title": "Other Latin hands",
        "when": "samples in the library that are not one of the above",
        "look": ["Treat each page on its own: note the hand, the abbreviations and what the model gets wrong."],
    },
    "greek-min": {
        "title": "Greek minuscule \u2014 Palatine Anthology (Heidelberg Pal. gr. 23)",
        "when": "10th century",
        "look": ["Connected minuscule with many ligatures; accents and breathings are tiny and easily dropped.",
                 "Common abbreviations: suspension strokes, and symbols for -\u03bd, \u03ba\u03b1\u03af, -\u03bf\u03c5.",
                 "The model reads this book at about 40% known Greek words, so expect many fixes.",
                 "Epigram headings and marginal scholia are in different sizes and are not the main text."],
    },
    "print": {
        "title": "Early printed Latin (roman type)",
        "when": "printed books; reading is by the print models or Tesseract",
        "look": ["Long s (\u017f) against f: f has a crossbar through the stem, \u017f only a nub on the left.",
                 "Ligatures: ct, st, ff, fi, \u00e6, \u0153; u/v and i/j used interchangeably.",
                 "Abbreviations (q\u0304, \u1d1b) are still common in early prints; line-end hyphenation splits words.",
                 "Running heads, page numbers and catchwords are not text."],
    },
    "fraktur": {
        "title": "Fraktur print (German)",
        "when": "German printed books",
        "look": ["Broken letterforms; long s and f, and round r after o, are the usual confusions.",
                 "Capitals are ornate and easily mistaken for each other (I/J, U/V, S/G).",
                 "ch, ck, tz and \u017ft are ligatures; umlauts appear as a small e above the letter in older prints."],
    },
    "greek-print": {"title": "Printed Greek", "when": "printed books", "look": ["Accents and breathings; ligatures in older types."]},
}


def style_of(entry: Dict[str, Any]) -> str:
    hay = f"{entry['dir'].name} {entry.get('label', '')}".lower()
    for needle, style in STYLE_HINTS:
        if needle in hay:
            return style
    return CLASS_STYLE.get(entry["class"], "hand-other")


def style_card(style: str, n_pages: int, sources: List[str], samples: List[str]) -> str:
    st = STYLES.get(style, {"title": style, "when": "", "look": []})
    look = "".join(f"<li>{html.escape(x)}</li>" for x in st["look"])
    imgs = "".join(f'<img src="data:image/jpeg;base64,{b}">' for b in samples)
    return (f'<section class="card"><h1>{html.escape(st["title"])}</h1>'
            f'<p class="meta">{html.escape(st["when"])} &middot; {n_pages} page(s) follow &middot; '
            f'{html.escape(", ".join(sources))}</p><div class="samples">{imgs}</div>'
            f'<h3>What to look for</h3><ul>{look}</ul>'
            f'<p class="meta">Then, on the next pages: type what you read in the form shown on the first page.</p>'
            f'</section>')


# ---------------------------------------------------------------------------
# the pool: every scan folder on disk
# ---------------------------------------------------------------------------
def _script_class(cache_name: str) -> str:
    n = cache_name.lower()
    if n.startswith("ocr_fraktur"):
        return "fraktur-print"
    if n.startswith("ocr_") or any(k in n for k in ("print", "reichenau", "gothic")):
        return "latin-print"
    return "latin-hand"


_VOCABS: Dict[str, Any] = {}


def _script_scores(text: str):
    """(Latin, Greek) known-word shares of a transcription: which script it really is."""
    from ingest import abbrev, greek
    if not _VOCABS:
        _VOCABS["la"] = abbrev.build_vocab()
        _VOCABS["grc"] = greek.build_vocab()
    lat = abbrev.vocab_hit_rate(abbrev.expand_text(text, _VOCABS["la"]), _VOCABS["la"])
    return lat, greek.hit_rate(text, _VOCABS["grc"])


def _model_for(cache: Path) -> Path:
    from ingest import htr
    return htr.MODEL_DIR / (re.sub(r"\.mlmodel$", "", cache.stem[len("htr_"):]) + ".mlmodel")


def _thumb(path: Path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        return im.size, list(im.convert("L").resize((24, 32)).tobytes())


def _is_duplicate(folder: Path, imgs: List[str], seen: List[Any]) -> bool:
    """Same page names and visually the same first image (different resolution is fine)."""
    for other, other_imgs in seen:
        if other_imgs != imgs:
            continue
        (sa, ta), (sb, tb) = _thumb(folder / imgs[0]), _thumb(other / imgs[0])
        if abs(sa[0] / sa[1] - sb[0] / sb[1]) > 0.02:
            continue
        if sum(abs(x - y) for x, y in zip(ta, tb)) / len(ta) < 12:
            return True
    return False


def pool() -> List[Dict[str, Any]]:
    """One entry per scan folder: dir, images, chosen text cache, script class, label."""
    from ingest import htr
    from web.scans import RAW, _read_json, locate
    folders: Dict[str, Dict[str, Any]] = {}
    con = sqlite3.connect(f"file:{REPO / 'data' / 'corpus.db'}?mode=ro", uri=True)
    try:
        docs = con.execute("select id,title,language,source from documents "
                           "where source like '%[scan:%' or source like '%IIIF manifest%'").fetchall()
    finally:
        con.close()
    for doc_id, title, lang, src in docs:
        loc = locate(src)
        if loc:
            folders.setdefault(str(loc["dir"].resolve()), {"title": title, "language": lang, "doc": doc_id})
    for root in (RAW, REPO / "data" / "manuscripts"):
        if root.is_dir():
            for d in sorted(root.iterdir()):
                if d.is_dir() and any(f.suffix.lower() in IMG_EXT for f in d.iterdir()):
                    folders.setdefault(str(d.resolve()), {"title": None, "language": None, "doc": None})
    out: List[Dict[str, Any]] = []
    seen: List[Any] = []
    for d, meta in sorted(folders.items(), key=lambda kv: kv[1]["doc"] is None):   # documents first
        folder = Path(d)
        imgs = sorted(f.name for f in folder.iterdir() if f.suffix.lower() in IMG_EXT)
        if imgs and _is_duplicate(folder, imgs, seen):
            continue                                            # the same scans fetched twice
        seen.append((folder, imgs))
        caches = [c for c in folder.glob("*.json")
                  if re.match(r"(htr|ocr)_", c.name) and not c.name.endswith(".lines.json")]
        if imgs and not caches and any(n in folder.name.lower() for n, st in STYLE_HINTS if st == "uncial"):
            ident = _read_json(folder / "scan.json").get("identifier") or folder.name
            out.append({**meta, "dir": folder, "images": imgs, "class": "latin-hand", "language": "la",
                        "cache": folder / "htr_catmus-medieval-1.6.0.json", "weak": True, "has_lines": 0,
                        "label": f"{folder.name} ({str(ident)[:60]})"})            # read on --analyze
            continue
        if not imgs or not caches:
            out.append({**meta, "dir": folder, "images": imgs, "cache": None, "class": "no-text", "label": folder.name})
            continue

        scored = []
        for c in caches:
            data = _read_json(c)
            lat, grc = _script_scores("\n".join(str(v) for v in data.values()))
            usable = c.name.startswith("ocr_") or _model_for(c).exists()
            scored.append(((usable, round(max(lat, grc), 1), Path(htr.lines_path(str(c))).exists(), len(data)),
                           c, lat, grc))
        _key, cache, lat, grc = max(scored, key=lambda t: t[0])
        cls = _script_class(cache.name)
        if grc > lat:                                         # which script the text actually is
            cls = "greek-print" if cache.name.startswith("ocr_") else "greek-hand"
        label = meta["title"] or _read_json(folder / "scan.json").get("identifier") or folder.name
        lp = Path(htr.lines_path(str(cache)))
        has = len(json.loads(lp.read_text(encoding="utf-8"))) if lp.exists() else 0
        out.append({**meta, "dir": folder, "images": imgs, "cache": cache, "class": cls,
                    "label": str(label)[:70], "has_lines": has,
                    "weak": max(lat, grc) < 0.2 and not cache.name.startswith("ocr_"),   # cache is noise
                    "language": meta["language"] or {"greek-hand": "grc", "greek-print": "grc",
                                                    "fraktur-print": "de"}.get(cls, "la")})
    return out


def source_for(entry: Dict[str, Any]) -> str:
    engine = "htr-greek" if entry["class"] == "greek-hand" else (
        "tesseract" if entry["cache"].name.startswith("ocr_") else "htr")
    return f"[scan: {entry['dir']}|{entry['cache'].name}] [OCR: {engine}]"


def show_pool() -> int:
    for e in pool():
        if e["cache"] is None:
            print(f"  {'(no text cache)':14s} {len(e['images']):4d} images                      {e['dir'].name}")
        else:
            print(f"  {e['class']:14s} {len(e['images']):4d} images, {e['has_lines']:3d} with lines  "
                  f"{e['dir'].name:24s} {e['label']}")
    return 0


# ---------------------------------------------------------------------------
# choosing pages
# ---------------------------------------------------------------------------
def choose_diverse(entries: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    from ingest import htr
    usable = [e for e in entries if e["cache"] is not None]
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for e in usable:
        by_class.setdefault(e["class"], []).append(e)
    cap = max(2, -(-n // 3))                                  # one book supplies at most a third
    quota = {id(e): 0 for e in usable}
    order: List[Dict[str, Any]] = []
    queues = [list(v) for v in by_class.values()]
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    left = n
    while left > 0 and any(quota[id(e)] < min(cap, len(e["images"])) for e in usable):
        for e in order:
            if left and quota[id(e)] < min(cap, len(e["images"])):
                quota[id(e)] += 1
                left -= 1
    per_folder = []
    for e in order:
        q = quota[id(e)]
        if not q:
            continue
        names = e["images"]
        step = len(names) / q
        pages = list(dict.fromkeys(names[min(len(names) - 1, int(step * (k + 0.5)))] for k in range(q)))
        lp = Path(htr.lines_path(str(e["cache"])))
        if lp.exists():                                      # swap half for the least-confident pages
            det = json.loads(lp.read_text(encoding="utf-8"))
            weak = sorted((p for p in det if p in names and p not in pages),
                          key=lambda p: sum(x.get("c", 1) for x in det[p]) / max(1, len(det[p])))
            for k in range(min(len(weak), len(pages) // 2)):
                pages[-1 - k] = weak[k]
        per_folder.append([(e, p) for p in dict.fromkeys(pages)])
    mixed: List[Dict[str, Any]] = []
    while any(per_folder):                                   # print order alternates hands
        for q in per_folder:
            if q:
                e, p = q.pop(0)
                mixed.append({"entry": e, "page": p})
    return mixed


def parse_pages(spec: str, names: List[str]) -> List[str]:
    if spec == "all":
        return names
    out: List[str] = []
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        lo, hi = int(a), int(b or a)
        out += [names[i - 1] for i in range(lo, hi + 1) if 1 <= i <= len(names)]
    return out


# ---------------------------------------------------------------------------
# building the sheet
# ---------------------------------------------------------------------------
def ensure_lines(sc, entry: Dict[str, Any], pages: List[str]) -> None:
    """Record per-line detail for chosen pages that have none (one worker run per folder)."""
    from ingest import htr
    cache = entry["cache"]
    if entry.get("weak"):
        # the only transcription here is noise (a model run on the wrong script): read it afresh
        # with the general medieval-hand model so the sheet has something worth correcting
        model = htr.MODEL_DIR / "catmus-medieval-1.6.0.mlmodel"
        newcache = entry["dir"] / "htr_catmus-medieval-1.6.0.json"
        print(f"  {entry['dir'].name}: existing text is noise, reading {len(pages)} page(s) "
              f"with {model.name} ...")
        htr.transcribe_images([str(entry["dir"] / p) for p in pages], cache_path=str(newcache),
                              model=str(model), log=lambda m: None)
        entry.update(cache=newcache, weak=False, **{"class": "latin-hand", "language": "la"})
        return
    lp = Path(htr.lines_path(str(cache)))
    have = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}
    missing = [p for p in pages if p not in have]
    if not missing:
        return
    if cache.name.startswith("ocr_"):
        for p in missing:
            sc.analyze(source_for(entry), p, entry["language"])
        return
    model = _model_for(cache)
    if not model.exists():
        print(f"  {entry['dir'].name}: model {model.name} is not installed, skipped")
        return
    print(f"  {entry['dir'].name}: reading {len(missing)} page(s) with {model.name} ...")
    htr.transcribe_images([str(entry["dir"] / p) for p in missing], cache_path=str(cache),
                          model=str(model), log=lambda m: None, detail_only=True)


def pick_lines(lines: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """k lines of a page: half the least-confident, half spread evenly; reading order kept."""
    lines = [x for x in lines if x["orig"].strip()]
    if k <= 0 or len(lines) <= k:
        return lines
    weak = {x["i"] for x in sorted(lines, key=lambda x: x["c"])[: k // 2]}
    rest = [x for x in lines if x["i"] not in weak]
    need = k - len(weak)
    step = len(rest) / need
    spread = {rest[min(len(rest) - 1, int(step * (j + 0.5)))]["i"] for j in range(need)}
    keep = weak | spread
    return [x for x in lines if x["i"] in keep]


def build(name: str, mixed: List[Dict[str, Any]], blind: bool, analyze: bool, diverse: bool,
          per_page: int = 12, by_style: bool = False) -> int:
    from PIL import Image
    from web.scans import Scans
    Image.MAX_IMAGE_PIXELS = None
    sc = Scans()
    if analyze:
        done = set()
        for m in mixed:
            e = m["entry"]
            if id(e) not in done:
                done.add(id(e))
                ensure_lines(sc, e, [x["page"] for x in mixed if x["entry"] is e])
    if by_style:                                           # one block per writing style, uncial first
        mixed = sorted(mixed, key=lambda m: STYLE_ORDER.index(style_of(m["entry"]))
                       if style_of(m["entry"]) in STYLE_ORDER else 99)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"name": name, "diverse": True, "blind": blind, "sources": [],
                                "languages": [], "rows": {}}
    body: List[str] = []
    skipped: List[str] = []
    classes: Dict[str, int] = {}
    last_style = None
    for e_page in mixed:
        e, page = e_page["entry"], e_page["page"]
        src = source_for(e)
        style = style_of(e)
        row_blind = blind or bool(STYLES.get(style, {}).get("blind"))
        got = sc.line_texts(src, page)
        if got is None:
            skipped.append(f"{e['dir'].name}/{page}")
            continue
        got["lines"] = pick_lines(got["lines"], per_page)
        if len(got["lines"]) < 3:
            skipped.append(f"{e['dir'].name}/{page} (too few lines)")
            continue
        if src not in manifest["sources"]:
            manifest["sources"].append(src)
            manifest["languages"].append(e["language"])
        si = manifest["sources"].index(src)
        pid = f"P{len(body) + 1:03d}"
        classes[e["class"]] = classes.get(e["class"], 0) + 1
        with Image.open(e["dir"] / page) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((900, 1200))
            tb = io.BytesIO()
            thumb.save(tb, format="JPEG", quality=70)
        if by_style and style != last_style:
            last_style = style
            group = [m for m in mixed if style_of(m["entry"]) == style]
            samples = []
            for x in sorted(got["lines"], key=lambda x: -len(x["orig"]))[:3]:
                cr = Image.open(io.BytesIO(sc.line_image(src, page, x["i"])))
                cr.thumbnail((1400, 220))
                cbuf = io.BytesIO()
                cr.convert("L").save(cbuf, format="JPEG", quality=80)
                samples.append(base64.b64encode(cbuf.getvalue()).decode())
            body.append(style_card(style, len(group), sorted({m["entry"]["dir"].name for m in group}), samples))
        rows = []
        for x in got["lines"]:
            rid = f"{pid}-L{x['i'] + 1:02d}"
            crop = Image.open(io.BytesIO(sc.line_image(src, page, x["i"])))
            crop.thumbnail((1100, 160))                      # paper does not need the full scan size
            cb = io.BytesIO()
            crop.convert("L").save(cb, format="JPEG", quality=75)
            manifest["rows"][rid] = {"s": si, "page": page, "line": x["i"], "orig": x["orig"], "b": x["b"]}
            model = "" if row_blind else f'<div class="model">{html.escape(x["orig"])}</div>'
            rows.append(f'<tr><td class="id">{rid}</td><td><img src="data:image/jpeg;base64,'
                        f'{base64.b64encode(cb.getvalue()).decode()}">{model}<div class="blank"></div></td></tr>')
        body.append(
            f'<section><h2>{pid} &mdash; {html.escape(e["label"])}</h2>'
            f'<p class="meta">{html.escape(style)} &middot; {html.escape(page)} &middot; '
            f'read by {html.escape(e["cache"].name)}</p>'
            f'<img class="thumb" src="data:image/jpeg;base64,{base64.b64encode(tb.getvalue()).decode()}">'
            f'</section><section class="lines"><h3>{pid}: {len(rows)} lines</h3><table>{"".join(rows)}</table></section>')
    if not body:
        raise SystemExit("none of the chosen pages has line detail yet; rerun with --analyze "
                         "(handwriting wants a free GPU)")
    (out / "sheet.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>{html.escape(name)}</title><style>{STYLE}</style>'
        f'<div class="key">{KEY_HTML}</div>{"".join(body)}', encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out / 'sheet.html'}  ({len(manifest['rows'])} lines on {len(body)} pages)")
    print("mix: " + ", ".join(f"{k} {v}" for k, v in sorted(classes.items())))
    if skipped:
        print("skipped (no line detail; use --analyze): " + ", ".join(skipped))
    return 0


def make_doc(doc_id: int, pages: str, name: Optional[str], blind: bool, analyze: bool,
             per_page: int = 12) -> int:
    _src, _lang, _title = _doc_row(doc_id)
    entry = next((e for e in pool() if e.get("doc") == doc_id and e["cache"] is not None), None)
    if entry is None:
        raise SystemExit("this document has no page images and text cache on this computer")
    chosen = parse_pages(pages, entry["images"])
    return build(name or f"doc{doc_id}-{pages.replace(',', '_')}",
                 [{"entry": entry, "page": p} for p in chosen], blind, analyze, False, per_page)


def make_diverse(n: int, name: Optional[str], blind: bool, analyze: bool, per_page: int = 12,
                 by_style: bool = False) -> int:
    mixed = choose_diverse(pool(), n)
    if not mixed:
        raise SystemExit("no scan folders with a text cache on this computer")
    return build(name or f"diverse-{n}", mixed, blind, analyze, True, per_page, by_style)


# ---------------------------------------------------------------------------
# reading the typed results back
# ---------------------------------------------------------------------------
def docx_text(path: Path) -> List[str]:
    if path.suffix.lower() != ".docx":
        return path.read_text(encoding="utf-8").splitlines()
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:(tab|br)[^>]*/>", " ", xml)
    return html.unescape(re.sub(r"<[^>]+>", "", xml)).splitlines()


def parse_row(rest: str):
    """-> (kind, text, ok) or None to skip; text None means 'as printed'."""
    r = rest.strip()
    if r in ("", "?"):
        return None
    if r == "=":
        return ("text", None, True)
    if r.upper() == "X":
        return ("noise", None, True)
    m = re.match(r"^([MG]):\s*(.*)$", r, re.I)
    if m:
        return (KINDS[m.group(1).upper()], m.group(2).strip() or None, True)
    return ("text", r, True)


def do_import(name: str, path: str, apply: bool) -> int:
    from web.scans import Scans
    manifest = json.loads((OUT / name / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sources"):
        sources, languages = manifest["sources"], manifest.get("languages") or ["la"] * len(manifest["sources"])
    else:                                                    # sheets made for a single document
        _src, language, _t = _doc_row(manifest["doc"])
        sources, languages = [manifest["source"]], [language]
    sc = Scans()
    todo, unknown, skipped = [], [], 0
    for line in docx_text(Path(path)):
        m = ROW.match(line)
        if not m:
            continue
        rid = m.group(1).upper()
        row = manifest["rows"].get(rid)
        if row is None:
            unknown.append(rid)
            continue
        parsed = parse_row(m.group(2))
        if parsed is None:
            skipped += 1
            continue
        todo.append((rid, row, *parsed))
    n_edit = sum(1 for _r, row, k, t, _ok in todo if t is not None and t != row["orig"])
    print(f"{len(todo)} rows to record ({n_edit} corrected text, "
          f"{sum(1 for x in todo if x[2] != 'text')} labelled), {skipped} skipped, "
          f"{len(unknown)} unknown IDs{': ' + ', '.join(unknown[:8]) if unknown else ''}")
    if not apply:
        print("dry run; add --apply to record them")
        return 0
    for _rid, row, kind, text, ok in todo:
        si = row.get("s", 0)
        sc.line_save(sources[si], row["page"], row["line"], text=text, kind=kind, ok=ok,
                     language=languages[si])
    print(f"recorded {len(todo)} lines")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--doc", type=int)
    m.add_argument("--diverse", type=int, metavar="N",
                   help="N pages mixed from every scan folder on disk (hands, scripts, quality)")
    m.add_argument("--pages", default="all", help="with --doc: viewer page numbers 20-29 | 5,9,12 | all")
    m.add_argument("--name")
    m.add_argument("--blind", action="store_true", help="leave the model's reading off the sheet")
    m.add_argument("--analyze", action="store_true", help="segment pages that have no line detail yet")
    m.add_argument("--by-style", action="store_true",
                   help="group pages by writing style, each group opened by a style card (uncial first)")
    m.add_argument("--lines-per-page", type=int, default=12, metavar="K",
                   help="lines to put on the sheet per page (0 = all; default 12: half the least "
                        "confident, half spread evenly)")
    sub.add_parser("pool", help="list the scan material available for sheets")
    i = sub.add_parser("import")
    i.add_argument("file")
    i.add_argument("--name", required=True)
    i.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.cmd == "pool":
        return show_pool()
    if a.cmd == "make":
        if a.diverse:
            return make_diverse(a.diverse, a.name, a.blind, a.analyze, a.lines_per_page, a.by_style)
        if a.doc is None:
            ap.error("make needs --doc or --diverse")
        return make_doc(a.doc, a.pages, a.name, a.blind, a.analyze, a.lines_per_page)
    return do_import(a.name, a.file, a.apply)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
