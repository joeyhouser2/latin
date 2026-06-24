"""Mine an aligned parallel corpus from Perseus dual editions -> JSONL.

Each output line is {src, tgt, citation, src_lang, era, source} — ready to
fine-tune an era-specific source->English translator.

Examples:
    # One or more Latin works (CTS urn or group.work)
    python scripts/mine_parallel.py phi0448.phi001 --era classical --out data/parallel/caesar.jsonl

    # All works in a textgroup that have an English edition
    python scripts/mine_parallel.py --discover phi0474 --era classical --out data/parallel/cicero.jsonl

    # Greek (uses the canonical-greekLit repo)
    python scripts/mine_parallel.py tlg0012.tlg001 --greek --era ancient --out data/parallel/homer.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.parallel import PerseusParallelMiner, pair_to_dict
from ingest.perseus import PerseusConnector, GreekPerseusConnector, First1KGreekConnector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("works", nargs="*", help="work ids (CTS urn or group.work)")
    ap.add_argument("--discover", help="textgroup id: mine all of its works")
    ap.add_argument("--all", action="store_true",
                    help="mine every work in the repo that has an English edition")
    ap.add_argument("--greek", action="store_true", help="use the classical Greek corpus (greekLit)")
    ap.add_argument("--first1k", action="store_true",
                    help="use First1KGreek (post-classical / patristic Greek)")
    ap.add_argument("--era", default="classical",
                    help="language_stage tag for these pairs")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--limit", type=int, default=200, help="max works when discovering")
    args = ap.parse_args()

    if args.first1k:
        miner = PerseusParallelMiner(repo="OpenGreekAndLatin/First1KGreek", src_tag="grc")
        connector = First1KGreekConnector()
    elif args.greek:
        miner = PerseusParallelMiner(repo="PerseusDL/canonical-greekLit", src_tag="grc")
        connector = GreekPerseusConnector()
    else:
        miner = PerseusParallelMiner(repo="PerseusDL/canonical-latinLit", src_tag="lat")
        connector = PerseusConnector()

    # work id -> (src_file, eng_file) or None (filenames resolved per-work)
    targets: dict = {}
    if args.all:
        targets = {wid: files for wid, files in miner.alignable_works().items()}
        print(f"{len(targets)} works in the repo have an English edition.")
    for wid in args.works:
        targets.setdefault(wid, None)
    if args.discover:
        for wid in connector.discover(args.discover, limit=args.limit):
            targets.setdefault(wid, None)
    if not targets:
        ap.error("provide work ids, --discover <textgroup>, or --all")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total = 0
    works_with_data = 0
    items = list(targets.items())
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, (wid, files) in enumerate(items, 1):
            src_file, eng_file = files if files else (None, None)
            try:
                pairs = miner.mine(wid, era=args.era, src_file=src_file, eng_file=eng_file)
            except Exception as exc:
                print(f"  ! [{i}/{len(items)}] {wid}: {exc}")
                continue
            if pairs:
                works_with_data += 1
                for p in pairs:
                    fh.write(json.dumps(pair_to_dict(p), ensure_ascii=False) + "\n")
                total += len(pairs)
            if pairs or not args.all:  # keep --all output readable
                print(f"  [{i}/{len(items)}] {wid}: {len(pairs)} pairs")

    print(f"\nWrote {total} pairs from {works_with_data}/{len(items)} works -> {args.out}")


if __name__ == "__main__":
    main()
