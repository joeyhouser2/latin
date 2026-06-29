"""Qualitative 3-way: old mild T5 vs new T5 vs the rich prompted 7B (victorian_prose).

The 7B `LocalLLMStylizer` victorian_prose preset is the quality ceiling (rich but
slow); the fine-tuned `Seq2SeqStylizer` is the fast/offline backend. This prints
their outputs side by side on plain probe sentences so the register gap to the
ceiling can be judged by eye.

    CUDA_VISIBLE_DEVICES=1 python scripts/compare_stylizers.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from core.stylizer import LocalLLMStylizer, Seq2SeqStylizer, StyleUnit

PROBES = [
    "The soldiers crossed the river at night and attacked the camp.",
    "She told her friend that she would never return to the city.",
    "The king was angry and ordered his men to burn the village.",
    "When the war was over, the people went back to their homes and fields.",
    "The ship was lost in the storm and all the sailors drowned.",
    "Tell me, who are you and where do you come from?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="models/stylizer-victorian-mild-bak")
    ap.add_argument("--new", default="models/stylizer-victorian")
    ap.add_argument("--skip-llm", action="store_true", help="omit the 7B reference")
    args = ap.parse_args()

    def run_t5(path):
        s = Seq2SeqStylizer(model_name=path, prefix="victorianize: ")
        return s.stylize_units([StyleUnit(literal=p) for p in PROBES])

    old = run_t5(args.old)
    new = run_t5(args.new)
    # Print T5 results immediately (fast); the 7B is slow, so stream per-probe.
    s = None
    if not args.skip_llm:
        s = LocalLLMStylizer(max_new_tokens=120, temperature=0.0)  # greedy = faster + deterministic
    for i, p in enumerate(PROBES):
        print(f"\nIN      : {p}", flush=True)
        print(f"OLD T5  : {old[i]}", flush=True)
        print(f"NEW T5  : {new[i]}", flush=True)
        if s is not None:
            print(f"7B prose: {s.stylize(p, preset='victorian_prose')}", flush=True)


if __name__ == "__main__":
    main()
