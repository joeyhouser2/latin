"""Self-test: which model dir each (language, stage) resolves to.

    python scripts/translator_routing_selftest.py

Only checks routing (no model is loaded). Models that are not present on this machine
resolve to stock NLLB, so those checks are skipped rather than failed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline import Library  # noqa: E402

CASES = [
    ("la", "late_antique", "models/nllb-latin-patristic"),
    ("la", "classical", "models/nllb-latin"),
    ("la", "medieval", "models/nllb-latin"),
    ("la", "unknown", "models/nllb-latin"),
    ("grc", "late_antique", "models/nllb-greek-v3"),
]


def main() -> int:
    failed = 0
    for lang, stage, want in CASES:
        if not os.path.isfile(os.path.join(want, "model.safetensors")):
            print(f"SKIP {lang}/{stage}: {want} not on this machine")
            continue
        got = Library._resolve_translator_key(lang, stage)
        ok = got == want
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {lang}/{stage} -> {got}" + ("" if ok else f" (wanted {want})"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
