"""Export hand-corrected lines from the scan viewer as a Kraken training set.

The viewer's line editor stores corrections beside the page images:

    <scan dir>/line_corrections.json
        {"_meta": {"language": "la", "cache": "htr_x.json", "engine": "htr"},
         "<page image>": {"<line index>": {"t": text, "orig": model text, "b": [x0,y0,x1,y1],
                                           "kind": "text|margin|gloss|noise", "ok": true}}}

A line is *ground truth* once it was verified (``ok``) or edited. This module turns
those lines into line-image / ``.gt.txt`` pairs, the ``ketos train -f path`` format
(Kraken 7), split by page so a validation page never shares lines with training:

    data/training/htr/<name>/
        lines/<folder>__<page>__<index>.png  +  .gt.txt
        train.txt  val.txt          one image path per line (relative to the set folder)
        manifest.json               counts, split, source folders, the ketos command
        layout_labels.json          every margin/gloss/noise label with its box (for a later
                                    region classifier)

Noise lines (kind ``noise``: neumes, ornament, bleed-through) are left out of the
recogniser's training data but kept in ``layout_labels.json``. Margin and gloss lines
*are* handwriting, so their text stays in.

CLI:  python -m training.htr_export [--name NAME] [--lang la|grc|all] [--dir DIR ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
MANUSCRIPTS = REPO / "data" / "manuscripts"
OUT_ROOT = REPO / "data" / "training" / "htr"
KINDS = ("text", "margin", "gloss", "noise")
TRAINABLE = ("text", "margin", "gloss")
PAD = 6                                           # pixels of context kept around a line box
CORRECTIONS = "line_corrections.json"


def find_folders(extra: Iterable[Path] = ()) -> List[Path]:
    """Every scan folder that has line corrections (default roots plus ``extra``)."""
    seen: Dict[Path, None] = {}
    for root in (RAW, MANUSCRIPTS):
        if root.is_dir():
            for p in sorted(root.glob(f"*/{CORRECTIONS}")):
                seen[p.parent.resolve()] = None
    for d in extra:
        d = Path(d)
        if (d / CORRECTIONS).is_file():
            seen[d.resolve()] = None
    return list(seen)


def is_ground_truth(rec: Dict[str, Any]) -> bool:
    """Verified, or changed from what the model said."""
    return bool(rec.get("ok")) or (rec.get("t") or "") != (rec.get("orig") or "")


def clean_text(t: str) -> str:
    """One NFC line, no control characters or line breaks."""
    t = unicodedata.normalize("NFC", t or "")
    t = re.sub(r"[\r\n\t]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _load_lines(folder: Path, meta: Dict[str, Any]) -> Dict[str, list]:
    """The model's per-page line detail (``<cache>.lines.json``), for polygons."""
    cache = meta.get("cache") or ""
    if not cache:
        return {}
    p = folder / (cache[:-5] + ".lines.json" if cache.endswith(".json") else cache + ".lines.json")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def crop_line(im, box: List[float], polygon: Optional[list] = None):
    """Cut one line out of a page image; blank everything outside its polygon if known."""
    from PIL import Image, ImageDraw, ImageStat
    W, H = im.size
    x0, y0, x1, y1 = box
    l, t = max(0, int(x0) - PAD), max(0, int(y0) - PAD)
    r, b = min(W, int(x1) + PAD), min(H, int(y1) + PAD)
    crop = im.crop((l, t, r, b)).convert("RGB")
    if polygon and len(polygon) >= 3:
        mask = Image.new("L", crop.size, 0)
        pts = [(float(x) - l, float(y) - t) for x, y in polygon]
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.MaxFilter(5))        # keep descenders and flourishes
        fill = tuple(int(v) for v in ImageStat.Stat(crop).median)
        bg = Image.new("RGB", crop.size, fill)
        crop = Image.composite(crop, bg, mask)
    return crop


def _val_pages(keys: List[str], fraction: float) -> set:
    """Deterministic page split: hash order, at least one validation page from 3 pages up."""
    if len(keys) < 3 or fraction <= 0:
        return set()
    ranked = sorted(keys, key=lambda k: hashlib.sha1(k.encode()).hexdigest())
    n = max(1, round(len(keys) * fraction))
    return set(ranked[:n])


def export(name: Optional[str] = None, folders: Optional[Iterable[Path]] = None,
           language: str = "all", val_fraction: float = 0.1,
           out_root: Path = OUT_ROOT, log=print) -> Dict[str, Any]:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    folders = list(folders) if folders is not None else find_folders()
    name = name or f"{language}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(out_root) / name
    (out / "lines").mkdir(parents=True, exist_ok=True)

    pages: List[Dict[str, Any]] = []            # one entry per (folder, page) with usable lines
    layout: List[Dict[str, Any]] = []
    skipped = {"empty": 0, "noise": 0, "stale": 0, "missing_image": 0, "other_language": 0}
    for folder in folders:
        try:
            store = json.loads((folder / CORRECTIONS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        meta = store.get("_meta", {})
        lang = meta.get("language") or "la"
        if language != "all" and lang != language:
            skipped["other_language"] += sum(len(v) for k, v in store.items() if k != "_meta")
            continue
        detail = _load_lines(folder, meta)
        for page, lines in store.items():
            if page == "_meta" or not isinstance(lines, dict):
                continue
            img_path = folder / page
            if not img_path.is_file():
                skipped["missing_image"] += len(lines)
                continue
            current = detail.get(page) or []
            usable = []
            for idx, rec in lines.items():
                try:
                    i = int(idx)
                except ValueError:
                    continue
                text = clean_text(rec.get("t", ""))
                if rec.get("b") is None:
                    skipped["stale"] += 1
                    continue
                # the stored box is what the user saw; take the polygon only from a line that
                # still has that box (the model's lines may have been recomputed since)
                poly = None
                if 0 <= i < len(current) and current[i].get("b") == rec.get("b"):
                    poly = current[i].get("p")
                kind = rec.get("kind", "text")
                if kind != "text":
                    layout.append({"folder": folder.name, "page": page, "line": i,
                                   "kind": kind, "box": rec["b"]})
                if not is_ground_truth(rec):
                    continue                          # a label only: the text was never checked
                if kind not in TRAINABLE:
                    skipped["noise"] += 1
                    continue
                if not text:
                    skipped["empty"] += 1
                    continue
                usable.append((i, text, rec["b"], poly, kind))
            if usable:
                pages.append({"folder": folder, "page": page, "lang": lang, "lines": usable})

    val = _val_pages([f"{p['folder'].name}/{p['page']}" for p in pages], val_fraction)
    train_list: List[str] = []
    val_list: List[str] = []
    n_chars = 0
    for pg in pages:
        key = f"{pg['folder'].name}/{pg['page']}"
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{pg['folder'].name}__{Path(pg['page']).stem}")
        with Image.open(pg["folder"] / pg["page"]) as im:
            for i, text, box, poly, kind in pg["lines"]:
                rel = f"lines/{stem}__{i:03d}.png"
                crop_line(im, box, poly).save(out / rel)
                (out / rel).with_suffix(".gt.txt").write_text(text + "\n", encoding="utf-8")
                (val_list if key in val else train_list).append(rel)
                n_chars += len(text)
    (out / "train.txt").write_text("\n".join(train_list) + ("\n" if train_list else ""), encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val_list) + ("\n" if val_list else ""), encoding="utf-8")
    (out / "layout_labels.json").write_text(json.dumps(layout, ensure_ascii=False, indent=1), encoding="utf-8")
    alphabet = sorted({ch for rel in train_list + val_list
                       for ch in (out / rel).with_suffix(".gt.txt").read_text(encoding="utf-8").strip()})
    manifest = {
        "name": name, "language": language, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lines": len(train_list) + len(val_list), "train_lines": len(train_list),
        "val_lines": len(val_list), "pages": len(pages), "val_pages": sorted(val),
        "characters": n_chars, "skipped": skipped, "layout_labels": len(layout),
        "folders": sorted({p["folder"].name for p in pages}),
        "alphabet": "".join(alphabet),
        "train_command": (
            rf"$env:PYTHONUTF8=1; .venv-htr\Scripts\ketos.exe -d cuda:0 train -f path --linetype bbox "
            rf"-t {out / 'train.txt'} -e {out / 'val.txt'} -i models\htr\<base model>.mlmodel "
            rf"--resize union -B 8 -o models\htr\ft-{name}"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"exported {manifest['lines']} lines ({len(train_list)} train / {len(val_list)} val) "
        f"from {len(pages)} pages -> {out}")
    return {**manifest, "dir": str(out)}


def stats(folders: Optional[Iterable[Path]] = None) -> Dict[str, Any]:
    """How much ground truth exists right now, per folder, without writing anything."""
    folders = list(folders) if folders is not None else find_folders()
    out = {"folders": [], "lines": 0, "pages": 0, "chars": 0}
    for folder in folders:
        try:
            store = json.loads((folder / CORRECTIONS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        n = c = pg = 0
        for page, lines in store.items():
            if page == "_meta" or not isinstance(lines, dict):
                continue
            got = [r for r in lines.values()
                   if is_ground_truth(r) and r.get("kind", "text") in TRAINABLE and clean_text(r.get("t", ""))]
            if got:
                pg += 1
                n += len(got)
                c += sum(len(clean_text(r["t"])) for r in got)
        out["folders"].append({"folder": folder.name, "language": store.get("_meta", {}).get("language", "la"),
                               "pages": pg, "lines": n})
        out["lines"] += n
        out["pages"] += pg
        out["chars"] += c
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="set name (default: <lang>-<timestamp>)")
    ap.add_argument("--lang", default="all", help="la | grc | all (default all)")
    ap.add_argument("--dir", action="append", default=[],
                    help="extra scan folder with line_corrections.json (repeatable)")
    ap.add_argument("--val", type=float, default=0.1, help="share of pages held out for validation")
    ap.add_argument("--stats", action="store_true", help="only report how much ground truth exists")
    a = ap.parse_args(argv)
    extra = [Path(d) for d in a.dir]
    if a.stats:
        print(json.dumps(stats(find_folders(extra)), indent=1))
        return 0
    res = export(a.name, find_folders(extra), a.lang, a.val)
    print(json.dumps({k: v for k, v in res.items() if k != "alphabet"}, indent=1, ensure_ascii=False))
    return 0 if res["lines"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
