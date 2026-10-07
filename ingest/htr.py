"""Handwritten-text recognition for manuscripts, via Kraken in an isolated venv.

Tesseract reads print. Medieval manuscripts need a model trained on handwriting,
so this module drives Kraken with the CATMuS Medieval model (CC-BY, trained on
Old/Middle French, Latin, Spanish and Italian manuscripts). Kraken pins its own
torch, so it runs in ``.venv-htr`` and this module only launches
``scripts/htr_worker.py`` there and caches what comes back.

Setup (once):
    python -m venv .venv-htr
    .venv-htr/Scripts/python -m pip install kraken
    # model: https://zenodo.org/records/12743230  (16 MB)
    curl -L -o models/htr/catmus-medieval.mlmodel \\
        https://zenodo.org/api/records/12743230/files/catmus-medieval.mlmodel/content

What comes out is *graphematic*: the model transcribes what is on the page and
does **not** expand abbreviations ("dñs", "ꝑ", macrons). Run the result through
``ingest.abbrev.expand_text`` before segmentation or translation, or the
translator is handed shorthand it has never seen. The IIIF connector does this.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
MODEL = _REPO / "models" / "htr" / "catmus-medieval.mlmodel"
WORKER = _REPO / "scripts" / "htr_worker.py"

SETUP_HINT = (
    "This looks like a manuscript (handwriting), which Tesseract cannot read, and "
    "no HTR engine is set up. Install Kraken in an isolated venv and fetch the "
    "CATMuS Medieval model -- see the setup block at the top of ingest/htr.py."
)


def _venv_python() -> Optional[Path]:
    for rel in ("Scripts/python.exe", "bin/python"):
        p = _REPO / ".venv-htr" / rel
        if p.exists():
            return p
    return None


def pick_gpu(min_free_mb: int = 2500) -> Optional[int]:
    """Index of the GPU with the most free memory (>= min_free_mb), else None.

    The GPUs are shared with the web app's job queue, so choose by what is free
    right now rather than assuming device 0.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
        free = [(int(i), int(m)) for i, m in
                (ln.split(",") for ln in out.strip().splitlines() if "," in ln)]
    except Exception:                                    # noqa: BLE001
        return None
    free = [(i, m) for i, m in free if m >= min_free_mb]
    return max(free, key=lambda t: t[1])[0] if free else None


def available() -> bool:
    return _venv_python() is not None and MODEL.exists()


def transcribe_images(paths: List[str], cache_path: Optional[str] = None,
                      model: Optional[str] = None, device: str = "auto",
                      log: Callable[[str], None] = lambda m: print(m, file=sys.stderr)
                      ) -> Dict[str, str]:
    """HTR the images; returns ``{file name: text}``. Resumable via the cache."""
    cache: Dict[str, str] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    todo = [p for p in paths if Path(p).name not in cache]
    if cache:
        log(f"  HTR cache: {len(paths) - len(todo)}/{len(paths)} pages already done")
    if todo:
        py = _venv_python()
        if py is None or not (model or MODEL).exists():
            raise RuntimeError(SETUP_HINT)
        env = dict(os.environ)
        if device == "auto":
            gpu = pick_gpu()
            device = f"cuda:{gpu}" if gpu is not None else "cpu"
        if device.startswith("cuda"):
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # match nvidia-smi's numbering
            env["CUDA_VISIBLE_DEVICES"] = device.partition(":")[2] or "0"
        cmd = [str(py), str(WORKER), "--model", str(model or MODEL),
               "--device", "cuda" if device.startswith("cuda") else "cpu", *todo]
        log(f"  HTR: transcribing {len(todo)} page(s) on {device}"
            + ("  (CPU is slow: minutes per page)" if device == "cpu" else ""))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace", env=env)
        err_lines: List[str] = []
        # drain stderr concurrently or a chatty worker blocks on a full pipe
        drain = threading.Thread(
            target=lambda: err_lines.extend(proc.stderr), daemon=True)  # type: ignore[arg-type]
        drain.start()
        done = 0
        for line in proc.stdout:                            # type: ignore[union-attr]
            line = line.strip()
            if not line.startswith("{"):
                continue
            rec = json.loads(line)
            if "error" in rec:
                log(f"  HTR failed on {rec['key']}: {rec['error']}")
                continue
            cache[rec["key"]] = rec["text"]
            done += 1
            log(f"  HTR {done}/{len(todo)} {rec['key']}: {rec['lines']} lines "
                f"in {rec.get('seconds', '?')}s")
            if cache_path:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                tmp = cache_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
                os.replace(tmp, cache_path)
        code = proc.wait()
        drain.join(timeout=5)
        if code != 0:
            raise RuntimeError(f"HTR worker exited {code}: {''.join(err_lines)[-600:]}")
    return {Path(p).name: cache.get(Path(p).name, "") for p in paths}
