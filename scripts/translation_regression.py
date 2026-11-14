"""Regression check: does a translator still get the lines it must get right?

    python scripts/translation_regression.py --engine nllb
    python scripts/translation_regression.py --engine llm --llm-model qwen3:14b
    python scripts/translation_regression.py --engine nllb --lang la --stage late_antique

Each case in data/qc/golden_la.jsonl is a Latin line with `must` substrings the English has
to contain and `forbid` substrings it must not (case-insensitive), and the quality flags of
core/qc.py. These are constraint checks on known failure points, NOT reference translations:
passing means "didn't make the known mistakes", not "is a good translation". Add a case
whenever a bad translation is found. Exit status 1 if any case fails, so it can gate a
routing or model change.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from core import qc


def check(case: dict, english: str) -> list:
    low = english.lower()
    problems = [f"missing '{m}'" for m in case.get("must", []) if m.lower() not in low]
    problems += [f"contains '{f}'" for f in case.get("forbid", []) if f.lower() in low]
    problems += [f"flag:{f}" for f in qc.flag_segment(case["latin"], english) if f in ("empty", "copied", "repeated")]
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="data/qc/golden_la.jsonl")
    ap.add_argument("--engine", choices=["nllb", "llm"], default="nllb")
    ap.add_argument("--llm-model", default=os.environ.get("LATIN_TRANSLATE_MODEL", "qwen3:14b"))
    ap.add_argument("--lang", default="la")
    ap.add_argument("--stage", default="early_modern", help="language stage, selects the NLLB route / LLM briefing")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip()]
    server = None
    if args.engine == "llm":
        from core.local_llm import PrivateOllama
        from core.llm_translator import LLMTranslator
        server = PrivateOllama(log_path="data/joblogs/ollama-regression.log").__enter__()
        tr = LLMTranslator(server.client, args.llm_model, args.lang, args.stage)
        label = f"llm:{args.llm_model}"
    else:
        from pipeline import Library
        lib = Library()
        tr = lib.translator_for(args.lang, args.stage)
        label = f"nllb:{getattr(tr, 'model_name', '?')}"
    try:
        out = tr.translate_batch([c["latin"] for c in cases], batch_size=8)
    finally:
        if server:
            server.__exit__(None, None, None)

    failed = 0
    print(f"=== {label}  ({args.lang}/{args.stage}) ===")
    for c, en in zip(cases, out):
        probs = check(c, en)
        failed += bool(probs)
        print(f"{'FAIL' if probs else 'ok  '} {c['id']}: {en}" + (f"   <- {'; '.join(probs)}" if probs else ""))
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
