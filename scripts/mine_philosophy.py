"""Mine the Aristotle commentators for lost philosophy (phase 3 of docs/quotation-mining-plan.md).

    python scripts/mine_philosophy.py evaluate                     # leave-one-collection-out, see below
    python scripts/mine_philosophy.py evaluate --hold-out rose --author ARISTOTLE
    python scripts/mine_philosophy.py run                          # discovery texts -> data/leads.db

A citation lead is a philosopher's name (ingest.philosophy.PHILOSOPHERS) beside a citing formula,
as for the historians. It is dropped when the nearby wording is in an open fragment collection
for that author (the baseline), or in that author's surviving works (the extant filter, which the
commentators need: most of what they cite is extant Aristotle, Plato and Theophrastus).

evaluate: pretend one collection was never made. Hide it from the baseline, mine the calibration
texts (Simplicius and Alexander, the most combed), and see whether the hidden collection's
fragments come back as top-ranked "new" leads. A lead is a hit when its passage is one the hidden
collection prints (its wording found there by the matcher). This measures recall of novelty and
ranking for this author, without any copyrighted answer key.

Downloads go to data/raw/quotations (gitignored): see ingest/philosophy.py.
"""
import argparse
import collections
import json
import pathlib
import signal
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.leads import LeadStore                                                    # noqa: E402
from ingest import philosophy                                                       # noqa: E402
from ingest.quotations import K, PHILOSOPHY_WITNESSES, find_citations, load_witness, score  # noqa: E402

OUT = ROOT / "data" / "quotations"
BASELINE_MAX = .1        # nearby wording in the author's fragment collection: already collected
# A quotation of a surviving work shows as one unbroken run of its words. Scattered shared phrases
# don't: a commentary on Aristotle shares them everywhere. On Simplicius and Alexander a run of 8+
# words dropped 91 of 463 Aristotle citations and kept 8 of the 9 at passages Rose prints, where
# "share of words >= 15%" kept none of the 9 (docs/quotation-mining-plan.md, phase 3).
EXTANT_RUN = 8
CALIBRATION = ("simplicius", "alexander")
DISCOVERY = tuple(w for w in PHILOSOPHY_WITNESSES if w not in CALIBRATION)
BEFORE, AFTER = 15, 45   # the window around the name, as find_citations uses


def longest_run(window, grams):
    """Length in words of the longest unbroken run of ``grams`` (3-grams) in a window."""
    best = run = 0
    for k in range(len(window) - K + 1):
        run = run + 1 if tuple(window[k:k + K]) in grams else 0
        best = max(best, run)
    return best + (K - 1 if best else 0)


def share(window, grams):
    covered = set()
    for k in range(len(window) - K + 1):
        if tuple(window[k:k + K]) in grams:
            covered.update(range(k, k + K))
    return round(len(covered) / max(1, len(window)), 2)


def citations(witness, fragments, extant):
    """Every philosopher citation in a text, with its baseline, extant and title evidence."""
    out = []
    for c in find_citations(witness, philosophy.PHILOSOPHERS, fragments, BEFORE, AFTER):
        window = witness.tok[max(0, c["pos"] - BEFORE):c["pos"] + AFTER]
        c["extant_share"] = share(window, extant.get(c["author"], set()))
        c["extant_run"] = longest_run(window, extant.get(c["author"], set()))
        c["title"] = philosophy.title_near(witness, c["pos"], c["author"])
        out.append(c)
    return out


def verdict(c):
    """Why a citation is not a lead, or None if it is one."""
    if c["author_share"] >= BASELINE_MAX:
        return "wording in the author's fragment collection"
    if c["title"] == "extant":
        return "an extant work's title beside the name"
    if c["extant_run"] >= EXTANT_RUN:
        return "a run of the author's surviving works"
    return None


def rank(c):
    """Higher first: a lost title, then a book formula, then least like the extant works."""
    return ({"lost": 2}.get(c["title"], 0) + {"book": 1, "story": .6, "verb": .5, "as": .3, "but": .2}.get(c["formula"], 0)
            - c["extant_share"] * 2 - c["author_share"])


def gold_spans(witness, fragments, min_coverage):
    """Word ranges of a text where the matcher finds a collection's fragments."""
    spans = []
    for f in fragments:
        if len(f["toks"]) < 12:
            continue
        coverage, where = score(f["toks"], witness)
        if coverage >= min_coverage:
            spans.append((where, where + int(len(f["toks"]) * 1.6) + 10))
    return spans


def evaluate(args):
    held = args.hold_out or ["wimmer"]
    # Every author the hidden collections cover (Diels: 19 Presocratics), unless one is named.
    authors = {args.author} if args.author else {a for h in held for a in philosophy.BASELINES[h][4]}
    others = [n for n in philosophy.BASELINES if n not in held]
    baseline = philosophy.baseline_fragments(others)
    seen, hidden = set(), []
    for f in philosophy.baseline_fragments(held):     # a chunk is repeated once per author: keep one
        if f["author"] in authors and f["n"] not in seen:
            seen.add(f["n"])
            hidden.append(f)
    extant = {a: philosophy.extant_grams(a) for a in authors}
    print(f"Hold out {'+'.join(held)} ({len(hidden)} chunks) for {', '.join(sorted(authors))}; "
          f"baseline: {', '.join(others)}; extant 3-grams: {sum(map(len, extant.values())):,}", flush=True)
    funnel, leads = collections.Counter(), []
    for name in args.witness or CALIBRATION:
        wit = load_witness(name)
        spans = gold_spans(wit, hidden, args.min_coverage)
        print(f"{name}: {len(wit.tok):,} words; passages of {'+'.join(held)} found in it: {len(spans)}", flush=True)
        for c in citations(wit, baseline, extant):
            if c["author"] not in authors:
                continue
            c["witness"] = name
            lo, hi = c["pos"] - BEFORE, c["pos"] + AFTER
            c["hit"] = any(a < hi and lo < b for a, b in spans)
            funnel["citations"] += 1
            funnel["citations at a hidden passage"] += c["hit"]
            why = verdict(c)
            if why:
                funnel[f"removed: {why}"] += 1
                funnel[f"  hidden passages among them: {why}"] += c["hit"]
                continue
            funnel["leads"] += 1
            funnel["leads at a hidden passage"] += c["hit"]
            leads.append(c)
    for k, v in funnel.items():
        print(f"  {k}: {v}")
    order = sorted(leads, key=rank, reverse=True)
    for k in (20, 50, 100):
        print(f"Precision@{k}: {sum(c['hit'] for c in order[:k])}/{min(k, len(order))}")
    base = sum(c["hit"] for c in leads) / max(1, len(leads))
    print(f"Base rate among leads: {base:.1%}")
    by = collections.defaultdict(collections.Counter)
    for c in leads:
        for key in (f"formula {c['formula']}", f"title {c['title']}", f"author {c['author']}"):
            by[key]["n"] += 1
            by[key]["hit"] += c["hit"]
    print("By evidence (hits/leads): " + "; ".join(f"{k} {v['hit']}/{v['n']}" for k, v in sorted(by.items())))
    for c in order[:args.top]:
        print(f"  {'HIT ' if c['hit'] else '    '}{c['witness'][:10]:10} {c['author'][:11]:11} {c['formula']:5} "
              f"{str(c['title']):6} ext {c['extant_share']:.2f} | {c['context'][:70]}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"philosophy_eval_{'_'.join(held)}.json").write_text(json.dumps({"funnel": funnel, "leads": order}, ensure_ascii=False,
                                                               indent=1), encoding="utf-8")


def run(args):
    """Mine texts for leads and store them for the Leads tab (kind "citation")."""
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    names = args.witness or list(DISCOVERY)
    baseline = philosophy.baseline_fragments()
    extant = {a: philosophy.extant_grams(a) for a in philosophy.EXTANT}
    store = LeadStore(args.leads) if args.leads else LeadStore()
    steps = len(names)
    try:
        for n, name in enumerate(names, 1):
            wit = load_witness(name)
            found = citations(wit, baseline, extant)
            leads = []
            removed = collections.Counter()
            for c in found:
                why = verdict(c)
                if why:
                    removed[why] += 1
                    continue
                c.update(kind="citation", witness=name, score=c["dfhg_share"], prior=None, credited=None)
                leads.append(c)
            kept = store.replace_witness(name, leads)
            print(f"  {name}: {len(found)} citations, {len(leads)} leads; removed {dict(removed)}; "
                  f"{kept['dropped']} old leads dropped | overall {n}/{steps}", flush=True)
    finally:
        store.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate", help="leave-one-collection-out on the calibration texts")
    e.add_argument("--hold-out", action="append", choices=sorted(philosophy.BASELINES),
                   help="repeatable; default wimmer. Hide every collection of an author to test it with no baseline")
    e.add_argument("--author", choices=sorted(philosophy.PHILOSOPHERS), help="default: the collection's author")
    e.add_argument("--witness", action="append", choices=sorted(PHILOSOPHY_WITNESSES))
    e.add_argument("--min-coverage", type=float, default=.5, help="lower for an OCR'd collection, e.g. .3")
    e.add_argument("--top", type=int, default=20)
    e.set_defaults(run=evaluate)
    r = sub.add_parser("run", help="store leads from the discovery texts")
    r.add_argument("--witness", action="append", choices=sorted(PHILOSOPHY_WITNESSES))
    r.add_argument("--leads", help="lead database (default data/leads.db)")
    r.set_defaults(run=run)
    args = ap.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
