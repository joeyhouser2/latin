"""Transcribe page images with Kraken. Runs inside the isolated ``.venv-htr``.

Kraken pins its own torch, so it lives in a separate virtualenv and the main
environment talks to it through this process (see ``ingest/htr.py``). One
process loads the segmentation and recognition models once and handles many
pages, emitting one JSON object per page on stdout as soon as it is done, so
the caller can cache incrementally and a crash loses at most one page.

    .venv-htr/Scripts/python scripts/htr_worker.py --model M.mlmodel img1.jpg img2.jpg

Output lines:  {"key": "<file name>", "text": "<lines joined by \\n>", "lines": N}
Errors:        {"key": "<file name>", "error": "<message>"}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def line_detail(r) -> dict:
    """Per-line and per-word confidence and boxes (image pixels) from one recognised line.

    Line: {"t": text, "c": mean char confidence, "b": [x0,y0,x1,y1], "w": [words]};
    word: {"t": word, "c": lowest char confidence in it, "b": [x0,y0,x1,y1]}.
    """
    text = r.prediction
    conf = [float(x) for x in r.confidences]
    xs = [p[0] for p in r.boundary]
    ys = [p[1] for p in r.boundary]
    box = [min(xs), min(ys), max(xs), max(ys)]
    out = {"t": text, "c": round(sum(conf) / len(conf), 3) if conf else 0.0, "b": box, "w": []}
    cuts = list(r.cuts)
    if len(cuts) != len(text) or len(conf) != len(text):
        return out                                   # no per-character geometry: line level only
    i = 0
    for chunk in text.split(" "):
        n = len(chunk)
        if n:
            cx = [p[0] for cut in cuts[i:i + n] for p in cut]
            letters = [conf[i + k] for k, ch in enumerate(chunk) if ch.isalpha()] or conf[i:i + n]
            out["w"].append({"t": chunk, "c": round(min(letters), 3),       # punctuation never counts against a word
                             "b": [min(cx), box[1], max(cx), box[3]]})
        i += n + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="recognition .mlmodel")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cuda uses the GPU(s) left visible by CUDA_VISIBLE_DEVICES")
    ap.add_argument("--batch", type=int, default=16, help="recognition batch size")
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    from PIL import Image
    from kraken import configs, tasks

    Image.MAX_IMAGE_PIXELS = None
    seg_model = tasks.SegmentationTaskModel.load_model()
    rec_model = tasks.RecognitionTaskModel.load_model(args.model)
    seg_cfg = configs.SegmentationInferenceConfig(accelerator=args.device)
    rec_cfg = configs.RecognitionInferenceConfig(accelerator=args.device,
                                                 batch_size=args.batch)

    for path in args.images:
        key = os.path.basename(path)
        t0 = time.time()
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                seg = seg_model.predict(im, seg_cfg)
                recs = list(rec_model.predict(im, seg, rec_cfg))
            lines = [r.prediction for r in recs]
            out = {"key": key, "text": "\n".join(l for l in lines if l.strip()),
                   "lines": len(lines), "seconds": round(time.time() - t0, 1),
                   "detail": [line_detail(r) for r in recs if r.prediction.strip()]}
        except Exception as e:                           # noqa: BLE001
            out = {"key": key, "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
