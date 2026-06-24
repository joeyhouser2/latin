"""Score translation models on a held-out slice of the parallel corpus.

Reports chrF and BLEU (via sacrebleu) for each model, overall and per era, so you
can compare a fine-tune against stock NLLB (or against each other) with numbers
rather than eyeballing a few sentences.

The corpus is shuffled with the SAME fixed seed training uses (42), so a slice
taken from the end — or from --holdout-from onward — is guaranteed unseen by any
model that trained on a `--max-pairs` prefix.

Each --model is "LABEL=PATH" with an optional ":norm" suffix to strip Greek
diacritics for that model (use it for models trained with --normalize-grc).
PATH may be "stock" for the base facebook/nllb-200-distilled-600M.

Examples:
  python scripts/eval_translation.py --data data/parallel/perseus_latin.jsonl data/parallel/grosenthal.jsonl \
      --lang la --by-era --model "stock=stock" --model "tuned=models/nllb-latin"

  python scripts/eval_translation.py --data data/parallel/perseus_greek.jsonl --lang grc --holdout-from 50000 \
      --model "stock=stock" --model "v1=models/nllb-greek" --model "v2=models/nllb-greek-v2:norm"
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.translator import NLLBTranslator
from core.normalize import strip_greek_diacritics

SRC_LANG = {"la": "lat_Latn", "grc": "ell_Grek"}
STOCK = "facebook/nllb-200-distilled-600M"


def load_pairs(patterns):
    pairs = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8") as fh:
                pairs += [json.loads(line) for line in fh if line.strip()]
    return pairs


def parse_model(spec):
    label, _, path = spec.partition("=")
    norm = path.endswith(":norm")
    if norm:
        path = path[: -len(":norm")]
    if path == "stock":
        path = STOCK
    return label, path, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--lang", choices=["la", "grc"], default="la")
    ap.add_argument("--model", action="append", required=True,
                    help='"LABEL=PATH" (PATH may be "stock"); add ":norm" to strip Greek diacritics')
    ap.add_argument("--test-size", type=int, default=500)
    ap.add_argument("--holdout-from", type=int, default=0,
                    help="take the test slice starting at this shuffled index "
                         "(set >= the --max-pairs used in training; 0 = take from the end)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--by-era", action="store_true")
    args = ap.parse_args()

    import sacrebleu

    pairs = load_pairs(args.data)
    random.Random(42).shuffle(pairs)
    if args.holdout_from:
        test = pairs[args.holdout_from: args.holdout_from + args.test_size]
    else:
        test = pairs[-args.test_size:]
    srcs = [p["src"] for p in test]
    refs = [p["tgt"] for p in test]
    print(f"Held-out test pairs: {len(test)} ({args.lang}); "
          f"slice {'from %d' % args.holdout_from if args.holdout_from else 'tail'}\n")

    rows = []
    for spec in args.model:
        label, path, norm = parse_model(spec)
        translator = NLLBTranslator(
            model_name=path, src_lang=SRC_LANG[args.lang],
            preprocess=strip_greek_diacritics if norm else None,
        )
        hyps = translator.translate_batch(srcs, batch_size=args.batch_size)
        chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        rows.append((label, chrf, bleu, hyps))
        print(f"  {label:<16} chrF {chrf:5.1f}   BLEU {bleu:5.1f}"
              f"{'   (norm)' if norm else ''}")

    if args.by_era:
        eras = sorted({p.get("era", "?") for p in test})
        print("\nPer era (chrF):")
        header = "  era".ljust(18) + "".join(f"{lbl:>10}" for lbl, *_ in rows)
        print(header)
        for era in eras:
            idx = [i for i, p in enumerate(test) if p.get("era") == era]
            line = f"  {era} (n={len(idx)})".ljust(18)
            for _, _, _, hyps in rows:
                e = sacrebleu.corpus_chrf([hyps[i] for i in idx],
                                          [[refs[i] for i in idx]]).score
                line += f"{e:>10.1f}"
            print(line)

    print("\nchrF ranges 0-100 (higher = closer to the reference). "
          "A few points is a real difference on this many sentences.")


if __name__ == "__main__":
    main()
