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
                lines = [r.prediction for r in rec_model.predict(im, seg, rec_cfg)]
            out = {"key": key, "text": "\n".join(l for l in lines if l.strip()),
                   "lines": len(lines), "seconds": round(time.time() - t0, 1)}
        except Exception as e:                           # noqa: BLE001
            out = {"key": key, "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
