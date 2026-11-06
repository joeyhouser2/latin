"""Rebuild whole text lines on pages where the segmenter breaks them into fragments.

On faded papyrus (St Gall Cod. Sang. 226) Kraken's segmenter returns several short boxes per
physical line and misses the faint lines altogether, so a "line" on a worksheet can be three
letters wide. The page is also slightly tilted, so fragments of one line sit at different
heights. This module works from what the segmenter did find:

  1. fit the page's skew (the slope that makes the boxes pile up into the sharpest rows);
  2. group the boxes into rows along that slope and join each row's fragments, left to right;
  3. estimate the line pitch, fill rows the segmenter skipped between detected rows, and keep
     going above and below them while the page still shows ink in the expected band.

Rows are ``{"t", "c", "b", "p", "sk", "w": [], "grid": True}``: ``p`` is the row's outline (a
parallelogram along the skew), ``sk`` the slope, so crops can be straightened. Rows nobody read
have ``t == ""``. Everything else (viewer, worksheets, training export) treats them like any
other line.
"""
from __future__ import annotations

from statistics import median
from typing import Any, Dict, List, Optional, Tuple


def _geometry(boxes: List[Dict[str, Any]]):
    import numpy as np
    xc = np.array([(b["b"][0] + b["b"][2]) / 2 for b in boxes], dtype="float64")
    yc = np.array([(b["b"][1] + b["b"][3]) / 2 for b in boxes], dtype="float64")
    w = np.array([b["b"][2] - b["b"][0] for b in boxes], dtype="float64")
    big = [b["b"][3] - b["b"][1] for b in boxes if (b["b"][2] - b["b"][0]) >= float(np.median(w))]
    h = float(np.median(big or [b["b"][3] - b["b"][1] for b in boxes]))
    return xc, yc, w, h


def _clusters(boxes: List[Dict[str, Any]]) -> Tuple[float, float, List[List[int]], float, float, float]:
    """(slope, x-reference, rows of box indices, line pitch, box height, y' of grid line 0)."""
    import numpy as np
    xc, yc, w, h = _geometry(boxes)
    xm = float(np.average(xc, weights=w))
    best_s, best_score = 0.0, -1.0
    for s in np.linspace(-0.06, 0.06, 49):
        yp = yc - s * (xc - xm)
        bins = np.arange(yp.min() - h, yp.max() + 2 * h, 0.25 * h)
        hist, _ = np.histogram(yp, bins=bins, weights=w)
        score = float((hist ** 2).sum())
        if score > best_score:
            best_s, best_score = float(s), score
    yp = yc - best_s * (xc - xm)
    # the page's lines are evenly spaced: fit y_k = a + pitch*k to the clusters of the wide boxes,
    # then snap every box to its nearest grid line (fragments of faint lines are scattered)
    wide = np.where(w >= np.median(w))[0]
    order = sorted(wide, key=lambda i: yp[i])
    base: List[List[int]] = [[order[0]]]
    for i in order[1:]:
        if yp[i] - yp[base[-1][-1]] > 0.5 * h:
            base.append([i])
        else:
            base[-1].append(i)
    cen = [float(np.average(yp[g], weights=w[g])) for g in base]
    diffs = [b2 - a2 for a2, b2 in zip(cen, cen[1:])]
    if diffs:
        med = float(np.median(diffs))
        single = [d for d in diffs if d < 1.5 * med]
        pitch = float(np.median(single)) if single else med
    else:
        pitch = 1.4 * h
    ks = [int(round((c - cen[0]) / pitch)) for c in cen]
    if len(set(ks)) >= 3:
        pitch, a0 = np.polyfit(ks, cen, 1)
        pitch, a0 = float(pitch), float(a0)
    else:
        a0 = cen[0]
    kk = np.rint((yp - a0) / pitch).astype(int)
    groups = [[int(i) for i in np.where(kk == k)[0]] for k in sorted(set(kk.tolist()))]
    return best_s, xm, groups, pitch, h, a0


def has_gutter(boxes: List[Dict[str, Any]]) -> bool:
    """True when no box crosses some vertical line in the middle of the text block: two columns."""
    x0 = min(b["b"][0] for b in boxes)
    x1 = max(b["b"][2] for b in boxes)
    span = x1 - x0
    for k in range(30, 71, 2):
        x = x0 + span * k / 100
        if not any(b["b"][0] <= x <= b["b"][2] for b in boxes):
            return True
    return False


def is_fragmented(lines: List[Dict[str, Any]]) -> bool:
    """True when grouping along the page's skew joins many boxes into fewer lines.

    Two-column pages are never touched: their left and right columns share rows but are separate
    lines, and merging them would make page-wide "lines".
    """
    boxes = [x for x in lines if x.get("b") and not x.get("grid")]
    if len(boxes) < 6 or has_gutter(boxes):
        return False
    _s, _xm, groups, _p, _h, _a = _clusters(boxes)
    return len(boxes) > 1.4 * len(groups)


def _band(gray, x_from: float, x_to: float, yc: float, slope: float, xm: float, half: float, s: float):
    """Pixels of the skewed band centred on (y' = yc) between x_from and x_to (image coordinates,
    ``s`` = downscale factor of ``gray``). Returns None if it falls off the page."""
    import numpy as np
    Hh, Ww = gray.shape
    xs = np.arange(int(max(0, x_from * s)), int(min(Ww - 1, x_to * s)), 2)
    if xs.size == 0:
        return None
    offs = np.arange(-half * s, half * s, 2.0)
    ys = (yc + slope * (xs / s - xm)) * s
    grid = ys[None, :] + offs[:, None]
    if grid.min() < 0 or grid.max() >= Hh:
        return None
    return gray[grid.astype(int), xs[None, :].astype(int)]


def regularize(im, lines: List[Dict[str, Any]], scale_to: int = 1300) -> List[Dict[str, Any]]:
    """Whole-line rows for a page (see module docstring). ``im`` is the page image."""
    import numpy as np
    boxes = [x for x in lines if x.get("b")]
    slope, xm, groups, pitch, _h, a0 = _clusters(boxes)
    xc, yc, w, _ = _geometry(boxes)
    yp = yc - slope * (xc - xm)
    rows = sorted(({"yc": a0 + pitch * int(round((float(np.average(yp[g], weights=w[g])) - a0) / pitch)),
                    "frags": [boxes[i] for i in g], "width": float(w[g].sum())} for g in groups),
                  key=lambda r: r["yc"])
    main = [r for r in rows if r["width"] >= 0.4 * max(r["width"] for r in rows)]
    x0 = min(min(f["b"][0] for f in r["frags"]) for r in (main or rows))
    x1 = max(max(f["b"][2] for f in r["frags"]) for r in (main or rows))

    def make(y: float, frags: List[Dict[str, Any]]) -> Dict[str, Any]:
        frags = sorted(frags, key=lambda f: f["b"][0])
        lo, hi = y - 0.55 * pitch, y + 0.55 * pitch
        poly = [[x0, lo + slope * (x0 - xm)], [x1, lo + slope * (x1 - xm)],
                [x1, hi + slope * (x1 - xm)], [x0, hi + slope * (x0 - xm)]]
        ys = [p[1] for p in poly]
        return {"t": " ".join(f["t"].strip() for f in frags if f["t"].strip()),
                "c": round(sum(f.get("c", 0) for f in frags) / len(frags), 3) if frags else 0.0,
                "b": [round(x0), round(min(ys)), round(x1), round(max(ys))],
                "p": [[round(a), round(b)] for a, b in poly], "sk": round(slope, 5),
                "w": [], "grid": True}

    seq: List[Dict[str, Any]] = []
    ys_seq: List[float] = []
    for k, r in enumerate(rows):
        if k:
            for j in range(1, int(round((r["yc"] - rows[k - 1]["yc"]) / pitch))):
                seq.append(make(rows[k - 1]["yc"] + j * pitch, []))
                ys_seq.append(rows[k - 1]["yc"] + j * pitch)
        seq.append(make(r["yc"], r["frags"]))
        ys_seq.append(r["yc"])

    # extend above and below while the page shows ink in the expected band
    g = im.convert("L")
    W, H = g.size
    s = scale_to / max(W, 1)
    gray = np.asarray(g.resize((scale_to, max(1, int(H * s)))), dtype="float32")
    block = _band(gray, x0, x1, (ys_seq[0] + ys_seq[-1]) / 2, slope, xm, (ys_seq[-1] - ys_seq[0]) / 2 + pitch, s)
    bg = float(np.percentile(block, 70)) if block is not None else float(np.median(gray))
    ink = []
    for r, y in zip(seq, ys_seq):
        if r["t"]:
            bnd = _band(gray, x0, x1, y, slope, xm, 0.4 * pitch, s)
            if bnd is not None:
                ink.append(bg - float(np.percentile(bnd, 8)))
    floor = 0.4 * float(np.median(ink)) if ink else 0.0

    def extend(direction: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        pending: List[Dict[str, Any]] = []
        misses, k = 0, 1
        y0 = ys_seq[0] if direction < 0 else ys_seq[-1]
        while misses < 2 and k < 80:
            y = y0 + direction * k * pitch
            bnd = _band(gray, x0, x1, y, slope, xm, 0.4 * pitch, s)
            if bnd is None:
                break
            pending.append(make(y, []))
            contrast, black = bg - float(np.percentile(bnd, 8)), float((bnd < 0.2 * bg).mean())
            if contrast >= floor and black < 0.02:
                out.extend(pending)
                pending, misses = [], 0
            else:
                misses += 1
            k += 1
        return out

    return list(reversed(extend(-1))) + seq + extend(1)
