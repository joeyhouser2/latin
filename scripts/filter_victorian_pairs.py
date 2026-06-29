"""Clean + contrast-filter a (modern, victorian) corpus so the stylizer commits harder.

The fine-tuned stylizer was too MILD: it often passed plain English through almost
unchanged. Two corpus defects feed that behaviour:

  1. Contamination — the LLM modernizer occasionally emitted other-language text
     (CJK) or mojibake on the `src` side. Garbage pairs the model must ignore.
  2. Near-copies — pairs where `src` ≈ `tgt` (the modernizer barely changed the
     ornate prose, or the sentence was already plain). These literally teach the
     model that "copy the input" is a valid Victorianization. Dropping them removes
     the copy signal and biases the model toward a real register shift.

We keep pairs in a chrF band: drop the near-copies (chrF too HIGH) and the
nonsense outliers (chrF absurdly LOW, usually contamination or misalignment).

    python scripts/filter_victorian_pairs.py \
        --in data/parallel/victorian_pairs.jsonl \
        --out data/parallel/victorian_pairs.filtered.jsonl
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import sacrebleu

_CJK = re.compile(r"[　-鿿가-퟿＀-￯]")


def _nonascii_frac(s: str) -> float:
    return sum(ord(c) > 127 for c in s) / max(1, len(s))


def _contaminated(p) -> bool:
    """Other-language text or heavy mojibake on either side."""
    src, tgt = p["src"], p["tgt"]
    if _CJK.search(src) or _CJK.search(tgt):
        return True
    # The Victorian gold can carry the odd accented name; the modern src should be
    # near-pure ASCII. A high non-ASCII fraction on src signals contamination.
    return _nonascii_frac(src) > 0.12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/parallel/victorian_pairs.jsonl")
    ap.add_argument("--out", default="data/parallel/victorian_pairs.filtered.jsonl")
    ap.add_argument("--chrf-max", type=float, default=62.0,
                    help="drop near-copy pairs with chrF(src,tgt) above this")
    ap.add_argument("--chrf-min", type=float, default=12.0,
                    help="drop nonsense/misaligned pairs below this")
    ap.add_argument("--min-chars", type=int, default=30)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    drop = {"contaminated": 0, "near_copy": 0, "too_low": 0, "short": 0, "identical": 0}
    kept = []
    for p in pairs:
        src, tgt = (p.get("src") or "").strip(), (p.get("tgt") or "").strip()
        if not src or not tgt:
            continue
        if len(src) < args.min_chars or len(tgt) < args.min_chars:
            drop["short"] += 1
            continue
        if src.lower() == tgt.lower():
            drop["identical"] += 1
            continue
        if _contaminated(p):
            drop["contaminated"] += 1
            continue
        c = sacrebleu.sentence_chrf(src, [tgt]).score
        if c > args.chrf_max:
            drop["near_copy"] += 1
            continue
        if c < args.chrf_min:
            drop["too_low"] += 1
            continue
        kept.append(p)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for p in kept:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Read {len(pairs)} pairs.")
    for k, v in drop.items():
        print(f"  dropped {k:12}: {v}")
    print(f"Kept {len(kept)} -> {args.out}")


if __name__ == "__main__":
    main()
