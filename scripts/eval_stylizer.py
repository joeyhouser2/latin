"""Objectively measure how hard a Victorian stylizer commits — and whether it stays faithful.

Two failure modes we trade between:
  * TOO MILD: output ~= input (passes plain English through). Detected by high
    chrF(pred, modern_src): the model barely changed the crib.
  * DEGENERATE / UNFAITHFUL: output drifts from meaning, repeats, truncates.
    Proxied by length ratio and (on held-out pairs) chrF(pred, victorian_gold).

Metrics, on a held-out slice of the (modern, victorian) corpus:
  * register_gap = chrF(pred, gold_victorian) - chrF(src_modern, gold_victorian)
        > 0  means the model moved the plain input *toward* the Victorian gold.
  * commit = 100 - chrF(pred, src_modern)
        higher = changed the input more (committed harder). Mild models score low.
  * len_ratio = mean(len(pred)/len(src))  — sanity check against truncation/bloat.

Also runs a fixed set of plain probe sentences (no gold) and prints before/after
so register shift can be eyeballed. Compare two models with --model A --model B.

    python scripts/eval_stylizer.py --model models/stylizer-victorian
    python scripts/eval_stylizer.py --model models/stylizer-victorian --model models/stylizer-victorian-v2
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import sacrebleu

from core.stylizer import Seq2SeqStylizer, StyleUnit

# Fixed plain probes (the kind of literal MT crib the reader feeds the stylizer).
PROBES = [
    "The soldiers crossed the river at night and attacked the camp.",
    "She told her friend that she would never return to the city.",
    "The king was angry and ordered his men to burn the village.",
    "I do not know why he came here, but he asked for food and water.",
    "When the war was over, the people went back to their homes and fields.",
    "He was a brave man, but he made many mistakes in his life.",
    "The ship was lost in the storm and all the sailors drowned.",
    "Tell me, who are you and where do you come from?",
]


def chrf(a, b):
    return sacrebleu.sentence_chrf(a, [b]).score


def _stylize(model, prefix, texts, num_beams=4, max_length=256):
    s = Seq2SeqStylizer(model_name=model, prefix=prefix,
                        num_beams=num_beams, max_length=max_length)
    return s.stylize_units([StyleUnit(literal=t) for t in texts])


def evaluate(model, prefix, eval_pairs, num_beams=4, max_length=256):
    srcs = [p["src"] for p in eval_pairs]
    golds = [p["tgt"] for p in eval_pairs]
    preds = _stylize(model, prefix, srcs, num_beams, max_length)

    commit, gap, lens = [], [], []
    for src, gold, pred in zip(srcs, golds, preds):
        commit.append(100 - chrf(pred, src))
        gap.append(chrf(pred, gold) - chrf(src, gold))
        lens.append(len(pred) / max(1, len(src)))
    n = len(preds)
    # corpus chrF of preds vs gold (overall register match)
    reg = sacrebleu.corpus_chrf(preds, [golds]).score
    return {
        "n": n,
        "commit": round(sum(commit) / n, 1),
        "register_gap": round(sum(gap) / n, 1),
        "chrf_vs_gold": round(reg, 1),
        "len_ratio": round(sum(lens) / n, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="model dir(s); repeat to compare")
    ap.add_argument("--prefix", default="victorianize: ")
    ap.add_argument("--data", default="data/parallel/victorian_pairs.jsonl")
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--max-eval", type=int, default=200)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    import random
    random.Random(42).shuffle(pairs)          # SAME seed/split as finetune_stylizer
    n_eval = max(1, int(len(pairs) * args.eval_frac))
    eval_pairs = pairs[:n_eval][: args.max_eval]
    print(f"Eval on {len(eval_pairs)} held-out pairs.\n")

    for model in args.model:
        print(f"### {model}")
        m = evaluate(model, args.prefix, eval_pairs, args.num_beams, args.max_length)
        print(f"  commit (↑=harder shift) : {m['commit']}")
        print(f"  register_gap (↑=toward gold): {m['register_gap']}")
        print(f"  chrf_vs_gold            : {m['chrf_vs_gold']}")
        print(f"  len_ratio (~1.0 healthy): {m['len_ratio']}")
        print("  -- probes --")
        for src, pred in zip(PROBES, _stylize(model, args.prefix, PROBES,
                                              args.num_beams, args.max_length)):
            print(f"    IN : {src}")
            print(f"    OUT: {pred}")
        print()


if __name__ == "__main__":
    main()
