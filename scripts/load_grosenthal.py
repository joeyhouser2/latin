"""Load the grosenthal/latin_english_parallel corpus into our JSONL format.

~99k sentence-aligned Latin->English pairs across many works. The dataset's
`file` field names the source work, which we use to tag era (e.g. the Vulgate is
late-antique) and keep provenance.

Run:  python scripts/load_grosenthal.py --out data/parallel/grosenthal.jsonl
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def era_for(file_field: str) -> str:
    f = (file_field or "").lower()
    if "vulgate" in f or "biblia" in f or "jerome" in f:
        return "late_antique"
    return "classical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/parallel/grosenthal.jsonl")
    ap.add_argument("--splits", nargs="+", default=["train"],
                    help="dataset splits to include (default: train only)")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("grosenthal/latin_english_parallel")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for split in args.splits:
            for row in ds[split]:
                src, tgt = (row.get("la") or "").strip(), (row.get("en") or "").strip()
                if not src or not tgt:
                    continue
                work = re.sub(r"\.json$", "", (row.get("file") or "").split("\\")[-1].split("/")[-1])
                fh.write(json.dumps({
                    "src": src, "tgt": tgt,
                    "citation": str(row.get("id", "")),
                    "src_lang": "la",
                    "era": era_for(row.get("file", "")),
                    "source": f"grosenthal:{work}",
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} pairs from splits {args.splits} -> {args.out}")


if __name__ == "__main__":
    main()
