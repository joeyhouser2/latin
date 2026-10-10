"""Measure how well ``ingest/linecheck.py`` flags real transcription errors.

    python scripts/linecheck_eval.py

Uses ``scripts/data/linecheck_truth.json`` (wrong tokens per page, from a reading of the
images) and the recogniser's per-word confidences in ``<cache>.lines.json``. Reports, for
the shipped thresholds and a sweep, precision (flagged words that are wrong), recall
(wrong words that are flagged) and how much of the page a reader must check.

Small and from one reader: it shows whether the flags are informative, not an accuracy
figure to quote. Add pages to the truth file as more get corrected.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8")

from ingest import abbrev, htr, linecheck                          # noqa: E402


def load_sets():
    truth = json.loads((REPO / "scripts" / "data" / "linecheck_truth.json").read_text(encoding="utf-8"))
    for name, spec in truth.items():
        if name.startswith("_"):
            continue
        cache = REPO / "data" / "raw" / spec["dir"] / spec["cache"]
        lines_file = Path(htr.lines_path(str(cache)))
        if not lines_file.exists():
            print(f"skip {name}: {lines_file.name} missing (analyse the pages in the viewer first)")
            continue
        detail = json.loads(lines_file.read_text(encoding="utf-8"))
        for page, errors in spec["pages"].items():
            if page in detail:
                yield name, page, detail[page], {(li, tok) for li, tok in errors}


def score(sets, vocab, min_flag):
    tp = fp = fn = words = 0
    for _, _, detail, errs in sets:
        res = linecheck.analyze(detail, vocab)
        for li, line in enumerate(res["lines"]):
            for w in line["w"]:
                wrong = (li, w["t"]) in errs
                flagged = w["f"] >= min_flag
                words += 1
                tp += wrong and flagged
                fp += (not wrong) and flagged
                fn += wrong and not flagged
    flagged = tp + fp
    prec = tp / flagged if flagged else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(words=words, wrong=tp + fn, flagged=flagged, tp=tp, precision=prec, recall=rec)


def main() -> None:
    sets = list(load_sets())
    if not sets:
        print("no evaluable pages")
        return
    vocab = abbrev.build_vocab()
    base = score(sets, vocab, 1)
    print(f"{base['words']} words, {base['wrong']} wrong ({base['wrong'] / base['words']:.0%} of the page)\n")
    print("shipped thresholds (BAD<%.2f, WEAK<%.2f, run-together>=%d):" %
          (linecheck.BAD_CONF, linecheck.WEAK_CONF, linecheck.RUN_TOGETHER_LEN))
    for label, mf in (("suspect only (red)", 2), ("suspect + weak (red+amber)", 1)):
        r = score(sets, vocab, mf)
        print(f"  {label:28s} flags {r['flagged']:3d} words ({r['flagged'] / r['words']:.0%}); "
              f"catches {r['tp']}/{r['wrong']} errors (recall {r['recall']:.0%}); "
              f"precision {r['precision']:.0%}")
    print("\nsweep (suspect + weak):")
    print("  BAD  WEAK  RTL   flagged  recall  precision")
    keep = (linecheck.BAD_CONF, linecheck.WEAK_CONF, linecheck.RUN_TOGETHER_LEN)
    try:
        for bad, weak, rtl in itertools.product((0.4, 0.5, 0.6), (0.7, 0.8, 0.9), (7, 9, 12)):
            linecheck.BAD_CONF, linecheck.WEAK_CONF, linecheck.RUN_TOGETHER_LEN = bad, weak, rtl
            r = score(sets, vocab, 1)
            print(f"  {bad:.1f}  {weak:.1f}   {rtl:2d}   {r['flagged'] / r['words']:6.0%}  "
                  f"{r['recall']:6.0%}  {r['precision']:8.0%}")
    finally:
        linecheck.BAD_CONF, linecheck.WEAK_CONF, linecheck.RUN_TOGETHER_LEN = keep


if __name__ == "__main__":
    main()
