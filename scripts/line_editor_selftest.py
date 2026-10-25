"""Self-test of the line editor backend and the Kraken training export (no GPU, no server).

Builds a throwaway scan folder with two synthetic pages (text drawn with PIL, a fake
model transcription with a deliberate error, per-line boxes and outlines), then:

  * lists lines, edits one, verifies one, marks one as a margin note and one as noise
  * checks stale boxes are ignored, "back to untouched" removes the record
  * rebuilds the page text from the lines (noise left out)
  * exports a training set and checks the .gt.txt/png pairs, split, manifest, layout labels

    python scripts/line_editor_selftest.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image, ImageDraw                     # noqa: E402

from training import htr_export                      # noqa: E402
from web.scans import Scans                          # noqa: E402

LINES = ["Fides quaerens intellectum", "Omnes homines natura scire", "desiderant et ea quae sunt",
         "in rerum natura cognoscere", "tubas"]
FAILED: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def make_page(folder: Path, name: str, shift: int = 0) -> list:
    im = Image.new("RGB", (900, 60 + 70 * len(LINES)), (226, 214, 180))
    d = ImageDraw.Draw(im)
    detail = []
    for i, text in enumerate(LINES):
        y = 40 + 70 * i
        d.text((60, y), text, fill=(30, 20, 10))
        box = [55, y - 8 + shift, 55 + 9 * len(text), y + 24 + shift]
        shown = text.replace("intellectum", "inteliectum") if i == 0 else text   # model error
        detail.append({"t": shown, "c": 0.9, "b": box, "w": [],
                       "p": [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]})
    im.save(folder / name)
    return detail


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lines_selftest_"))
    scan = tmp / "scan"
    scan.mkdir()
    cache = {}
    detail = {}
    for k, name in enumerate(("p1.png", "p2.png", "p3.png")):
        detail[name] = make_page(scan, name)
        cache[name] = "\n".join(x["t"] for x in detail[name])
    (scan / "htr_test.json").write_text(json.dumps(cache), encoding="utf-8")
    (scan / "htr_test.lines.json").write_text(json.dumps(detail), encoding="utf-8")
    source = f"[scan: {scan}|htr_test.json] [OCR: htr]"
    sc = Scans()

    got = sc.line_texts(source, "p1.png")
    check("lines listed", got is not None and len(got["lines"]) == len(LINES))
    check("nothing verified at first", got["verified"] == 0 and got["edited"] == 0)
    check("model text shown", got["lines"][0]["t"].startswith("Fides quaerens inteliectum"))

    out = sc.line_save(source, "p1.png", 0, text="Fides quaerens intellectum", ok=True)
    check("edit saved and counted", out["edited"] == 1 and out["verified"] == 1)
    out = sc.line_save(source, "p1.png", 1, ok=True)
    check("verify without edit", out["lines"][1]["ok"] and not out["lines"][1]["edited"])
    out = sc.line_save(source, "p1.png", 4, kind="gloss", ok=True)
    check("gloss kind kept", out["lines"][4]["kind"] == "gloss")
    out = sc.line_save(source, "p1.png", 3, kind="noise")
    check("noise kind kept", out["lines"][3]["kind"] == "noise")
    out = sc.line_save(source, "p1.png", 3, kind="text")
    check("back to untouched drops the record",
          "3" not in json.loads((scan / "line_corrections.json").read_text(encoding="utf-8"))["p1.png"])
    try:
        sc.line_save(source, "p1.png", 2, kind="bogus")
        check("bad kind rejected", False)
    except ValueError:
        check("bad kind rejected", True)
    check("index out of range -> None", sc.line_save(source, "p1.png", 99, ok=True) is None)

    png = sc.line_image(source, "p1.png", 0)
    check("line image is a PNG", png is not None and png[:8] == b"\x89PNG\r\n\x1a\n")

    # page 2: verify everything (the "clean page" case), margin note on the last line
    for i in range(len(LINES)):
        sc.line_save(source, "p2.png", i, ok=True, kind="margin" if i == 4 else None)
    # page 3 has one noise line and one fix
    sc.line_save(source, "p3.png", 4, kind="noise", ok=True)
    sc.line_save(source, "p3.png", 0, text="Fides quaerens intellectum", ok=True)

    # stale: the stored box no longer matches the model's line -> correction ignored
    stale = json.loads((scan / "htr_test.lines.json").read_text(encoding="utf-8"))
    stale["p1.png"][0]["b"] = [1, 1, 2, 2]
    (scan / "htr_test.lines.json").write_text(json.dumps(stale), encoding="utf-8")
    got = sc.line_texts(source, "p1.png")
    check("stale correction ignored", got["lines"][0]["t"].startswith("Fides quaerens inteliectum"))
    stale["p1.png"][0]["b"] = json.loads((scan / "line_corrections.json").read_text(encoding="utf-8"))["p1.png"]["0"]["b"]
    (scan / "htr_test.lines.json").write_text(json.dumps(stale), encoding="utf-8")

    page = sc.lines_to_page(source, "p3.png")
    text = sc.page(source, "p3.png")["text"]
    check("page text rebuilt, noise dropped", page and "tubas" not in text and "intellectum" in text.lower(), text)

    # ---- export ----
    res = htr_export.export("selftest", folders=[scan], language="la", out_root=tmp / "out", log=lambda m: None)
    out = Path(res["dir"])
    imgs = sorted((out / "lines").glob("*.png"))
    gts = sorted((out / "lines").glob("*.gt.txt"))
    # p1: lines 0,1,4(gloss) verified; p2: 5 lines; p3: line 0 (+ noise line 4 excluded)
    check("png/gt pairs match", len(imgs) == len(gts) == res["lines"], f"{len(imgs)} {len(gts)} {res['lines']}")
    check("expected line count", res["lines"] == 3 + 5 + 1, str(res["lines"]))
    check("corrected text exported", any("intellectum" in g.read_text(encoding="utf-8") and "inteliectum" not in
                                        g.read_text(encoding="utf-8") for g in gts))
    check("noise left out of training", not any(g.read_text(encoding="utf-8").strip() == "tubas" and "p3" in g.name
                                                for g in gts))
    labels = json.loads((out / "layout_labels.json").read_text(encoding="utf-8"))
    check("layout labels recorded", {x["kind"] for x in labels} >= {"noise", "gloss", "margin"})
    train = (out / "train.txt").read_text(encoding="utf-8").split()
    val = (out / "val.txt").read_text(encoding="utf-8").split()
    check("split by page", len(train) + len(val) == res["lines"]
          and not ({Path(x).name.split("__")[1] for x in train} & {Path(x).name.split("__")[1] for x in val}))
    check("validation page exists", len(val) > 0)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    check("manifest has ketos command", "ketos" in manifest["train_command"] and "-f path" in manifest["train_command"])
    check("stats agree", htr_export.stats([scan])["lines"] == res["lines"])
    print(f"\nscratch: {tmp}")
    print("ALL PASSED" if not FAILED else f"FAILED: {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
