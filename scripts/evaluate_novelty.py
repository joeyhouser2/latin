"""Evaluation 1 of docs/quotation-mining-plan.md: can the pipeline rediscover Jacoby's additions to Mueller?

    python scripts/evaluate_novelty.py                         # the scholia (phase 2)
    python scripts/evaluate_novelty.py --witness athenaeus --top 20

Pretend Mueller (DFHG) is all we know. Every citation lead that survives the filters is
then "new", and Jacoby's FGrHist says whether it really was new to Mueller:

  jacoby       Jacoby prints a fragment of this author at this passage: a true find
  not_jacoby   Jacoby covers the author but not this passage: a name collision, a
               mention, an OCR slip, or (rarely) something both editors missed
  unjudged     the author is not in our FGrHist volumes (I, II, III B), so no answer

Precision@k counts only judged leads. Three rankings are compared: a prior from Mueller
alone (how often DFHG credits this text with the author, per citation; see add_prior),
the local model's tags (scripts/triage_leads.py), and the citing formula.

It also reports what each filter removes, and recall: of Jacoby's additions in these
texts by authors the citation finder can name, how many come out as leads.

Needs data/quotations/known_loci.jsonl (scripts/build_known_loci.py) and the leads in
data/leads.db for the model ranking. Writes data/quotations/novelty_<texts>.json.
"""
import argparse
import collections
import json
import pathlib
import re
import sqlite3
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest.known_loci import TOLERANCE, name_key, near, witness_locus                 # noqa: E402
from ingest.quotations import (AMBIGUOUS_NAMES, RAW, WITNESSES, author_priors, author_stems, character_names,  # noqa: E402
                               find_citations, is_character, load_dfhg, load_witness)

TABLE = ROOT / "data" / "quotations" / "known_loci.jsonl"
CITE_MAX_SHARE = .1          # as scripts/mine_quotations.py: nearby wording DFHG already has
SCHOLIA = ("euripides_scholia", "sophocles_scholia", "homer_scholia")


def load_table():
    rows = [json.loads(line) for line in TABLE.open(encoding="utf-8")]
    for r in rows:
        r["key"] = name_key(r["author"].split(" ", 1)[1] if r["collection"] == "FGrHist" else r["author"])
    return rows


def model_tags(witness):
    db = sqlite3.connect(f"file:{ROOT / 'data' / 'leads.db'}?mode=ro", uri=True)
    rows = db.execute("SELECT key, tag_json, rank_score FROM leads WHERE witness=? AND kind='citation'", (witness,))
    return {k: (json.loads(t) if t else None, s) for k, t, s in rows}


def evaluate(witness, fragments, table, stems, share="dfhg_share", use_characters=True):
    tol = TOLERANCE[witness]
    wit = load_witness(witness)
    mine = [r for r in table if r["witness"] == witness and r["locus"]]
    dfhg = [(r["key"], r["locus"]) for r in mine if r["collection"] == "FHG"]
    jacoby = [(r["key"], r["locus"]) for r in mine if r["collection"] == "FGrHist"]
    jacoby_authors = {r["key"] for r in table if r["collection"] == "FGrHist"}
    tags = model_tags(witness)
    characters = character_names(witness)

    # Each filter in order, with how many leads it removes and how many of those were Jacoby
    # additions (a removal Jacoby says was wrong).
    filters = (
        ("wording in DFHG", lambda c: c[share] >= CITE_MAX_SHARE),
        ("passage in DFHG for this author",
         lambda c: c["here"] and any(a == c["key"] and near(witness, c["here"], k, tol) for a, k in dfhg)),
        ("character of the commented work",
         lambda c: use_characters and is_character(c["name"], stems[c["author"]], characters)),
    )
    funnel = collections.Counter()
    leads, all_cites = [], []
    for c in find_citations(wit, stems, fragments):
        funnel["citations"] += 1
        c["key"], c["here"] = name_key(c["author"]), witness_locus(wit, c["pos"])
        at = lambda index: c["here"] and any(a == c["key"] and near(witness, c["here"], k, tol)  # noqa: E731
                                             for a, k in index)
        if c["key"] not in jacoby_authors:
            c["label"] = "unjudged"
        elif at(jacoby) and not at(dfhg):          # an addition: in Jacoby here, not in Mueller
            c["label"] = "jacoby"
        else:
            c["label"] = "not_jacoby"
        all_cites.append(c)
        removed = next((name for name, test in filters if test(c)), None)
        if removed:
            funnel[f"removed: {removed}"] += 1
            funnel[f"  of which Jacoby additions: {removed}"] += c["label"] == "jacoby"
            continue
        funnel["new to DFHG"] += 1
        tag, rank = tags.get(f"citation|{witness}|{c['pos']}|{c['author']}", (None, 0))
        c["tag"], c["rank"] = tag, rank if tag else None
        leads.append(c)

    # The answer key: Jacoby's passages in this text that DFHG lacks, by authors the finder can name.
    nameable = {name_key(a) for a in stems}
    key_items = {(a, k) for a, k in jacoby if a in nameable
                 and not any(a == b and near(witness, k, l, tol) for b, l in dfhg)}
    recall = collections.Counter()
    for a, k in key_items:
        hit = lambda pool: any(c["key"] == a and c["here"] and near(witness, c["here"], k, tol) for c in pool)  # noqa: E731
        recall["answer key"] += 1
        recall["named by the finder"] += hit(all_cites)
        recall["survives as a lead"] += hit(leads)
    return funnel, leads, recall


def heuristic(c):
    return {"book": 1, "story": .8, "verb": .6, "as": .4, "but": .2}.get(c["formula"], 0) - c["dfhg_share"]


def add_prior(leads, fragments):
    """The Mueller-only prior (ingest.quotations.author_priors), text by text, as mining stores it."""
    for w in {c["witness"] for c in leads}:
        author_priors([c for c in leads if c["witness"] == w], fragments, WITNESSES[w][3])


RANKINGS = {
    "prior": lambda c: (c["prior"], heuristic(c)),
    "model": lambda c: (c["rank"] or 0, heuristic(c)),
    "formula": heuristic,
}


def precision(judged, k):
    return sum(c["label"] == "jacoby" for c in judged[:k])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--witness", action="append", help=f"repeatable; default {', '.join(SCHOLIA)}")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--wording", choices=("any", "author"), default="author",
                    help="drop a citation whose nearby wording DFHG has for the cited author (as mining does), "
                         "or for any author")
    ap.add_argument("--rank", choices=sorted(RANKINGS), default="prior",
                    help="order of the leads for precision@k (all three are reported)")
    ap.add_argument("--keep-characters", action="store_true",
                    help="keep names that belong to characters of the commented work")
    args = ap.parse_args()
    texts = args.witness or list(SCHOLIA)

    fragments = load_dfhg(RAW / "dfhg")
    stems = author_stems(fragments, AMBIGUOUS_NAMES)
    table = load_table()
    everything, total_funnel, total_recall = [], collections.Counter(), collections.Counter()
    for w in texts:
        funnel, leads, recall = evaluate(w, fragments, table, stems,
                                         "dfhg_share" if args.wording == "any" else "author_share",
                                         not args.keep_characters)
        labels = collections.Counter(c["label"] for c in leads)
        print(f"\n{w}: " + "; ".join(f"{k} {v}" for k, v in funnel.items()))
        print(f"  new leads: {dict(labels)}")
        print(f"  recall: {dict(recall)}")
        total_funnel.update(funnel)
        total_recall.update(recall)
        everything += [dict(c, witness=w) for c in leads]

    add_prior(everything, fragments)
    tagged = all(c["tag"] for c in everything)
    print("\nAll texts: " + "; ".join(f"{k} {v}" for k, v in total_funnel.items()))
    print(f"Recall: {dict(total_recall)}")
    judged_count = sum(c["label"] != "unjudged" for c in everything)
    print(f"Judged leads {judged_count} of {len(everything)}; Jacoby additions among them "
          f"{sum(c['label'] == 'jacoby' for c in everything)}.")
    for name, key in RANKINGS.items():
        if name == "model" and not tagged:
            print("  ranked by model: not all leads are tagged (scripts/triage_leads.py)")
            continue
        j = [c for c in sorted(everything, key=key, reverse=True) if c["label"] != "unjudged"]
        print(f"  ranked by {name:7}: " + "  ".join(f"P@{k} {precision(j, k)}/{min(k, len(j))}" for k in (20, 50, 100)))
    order = sorted(everything, key=RANKINGS[args.rank], reverse=True)
    judged = [c for c in order if c["label"] != "unjudged"]
    top = judged[:args.top]
    hits = precision(judged, args.top)
    print(f"Precision@{len(top)} ranked by {args.rank} (judged leads): {hits}/{len(top)} = {hits / max(1, len(top)):.0%}")
    by_formula = collections.defaultdict(collections.Counter)
    for c in judged:
        by_formula[c["formula"]]["judged"] += 1
        by_formula[c["formula"]]["jacoby"] += c["label"] == "jacoby"
    print("By formula (judged leads, Jacoby additions): " + "; ".join(
        f"{f} {n['jacoby']}/{n['judged']}" for f, n in sorted(by_formula.items(), key=lambda x: -x[1]["judged"])))
    if tagged:
        by_tag = collections.defaultdict(collections.Counter)
        for c in judged:
            by_tag[c["tag"]["same_person"]]["judged"] += 1
            by_tag[c["tag"]["same_person"]]["jacoby"] += c["label"] == "jacoby"
        print("By model's same_person (judged, Jacoby additions): " + "; ".join(
            f"{k} {n['jacoby']}/{n['judged']}" for k, n in by_tag.items()))
    for c in top:
        tag = c["tag"] or {}
        print(f"  {c['label']:10} {c['witness'][:9]:9} {c['here'] or c['loc']:12} {c['author'][:22]:22} "
              f"{c['formula']:5} prior {c['prior']:.2f} {tag.get('same_person', '-'):6} {c['context'][:50]}")
    out = ROOT / "data" / "quotations" / f"novelty_{'_'.join(texts)}.json"
    out.write_text(json.dumps({"funnel": total_funnel, "recall": total_recall, "leads": order}, ensure_ascii=False,
                              indent=1, default=str), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
