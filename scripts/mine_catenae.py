"""Check Cramer's catenae item by item against what is already known (phase 4 of docs/quotation-mining-plan.md).

    python scripts/mine_catenae.py            # report, and store leads in data/leads.db

Each catena item carries its author's label (ingest/catenae.py). An item is "found" when its
wording is in that author's open texts: surviving works and editions of his catena fragments
(First1KGreek; see FATHERS). Found items are already known. What remains:

  unfound    an author with open texts, but the item is not in them: a lost work, a fragment the
             editors missed, or a wrong label. These are the leads.
  no texts   the author has no open text here (Chrysostom's homilies, Theodoret on Paul, all of
             Severian, Gennadius, Photius, Apollinaris, Didymus, Theodore, Ammonius ...). Their items
             are printed in Cramer, so they are published, though perhaps not identified.
  anonymous  "Ἐξ ἀνεπιγράφου": matched against every author's texts, to name it.

Origen is the calibration author: his surviving commentaries on Matthew and John and the editions
of his catena fragments on Luke and John are open, so the share of his items found measures the
method; his unfound items are where to look.
"""
import collections
import glob
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.leads import LeadStore                                   # noqa: E402
from ingest.catenae import read_catena                             # noqa: E402
from ingest.quotations import RAW, Witness, read_tei, score        # noqa: E402

FATHERS_DIR = RAW / "fathers"
# Author -> First1KGreek text groups in data/raw/quotations/fathers. "GREGORY" in the catenae is
# Nazianzus or Nyssa; only Nazianzus is open.
FATHERS = {"ORIGEN": "tlg2042", "EUSEBIUS": "tlg2018", "CYRIL": "tlg4090", "THEODORET": "tlg4089",
           "ATHANASIUS": "tlg2035", "GREGORY": "tlg2022", "HIPPOLYTUS": "tlg2115", "IRENAEUS": "tlg1447"}
# Where the open texts cover an author's work on that book: Origen on Luke (Rauer's fragments, the
# Homilies) and John (Preuschen's fragments, the surviving Commentary). Only there does "not in his
# texts" mean something. Elsewhere (Origen on Matthew, Romans and 1 Corinthians, collected in
# editions not open here; Cyril and Theodoret, whose New Testament commentaries are not open) an
# item can be found but not missed.
COVERED = {"ORIGEN": {"tlg4102.tlg004", "tlg4102.tlg006", "tlg4102.tlg005", "tlg4102.tlg007"}}
FOUND = .5            # share of an item's words the matcher finds together in the author's texts
MIN_WORDS = 12
WITNESS = "cramer_catenae"


def catena_items():
    items = []
    for path in sorted(glob.glob(str(RAW / "witnesses" / "tlg4102.*.xml"))):
        items += read_catena(path, pathlib.Path(path).stem)
    return items


def father_texts():
    texts = {}
    for author, group in FATHERS.items():
        paths = sorted(FATHERS_DIR.glob(f"{group}.*.xml"))
        if paths:
            texts[author] = Witness(author, read_tei(paths))
    return texts


def main():
    items = [i for i in catena_items() if i.author and len(i.toks) >= MIN_WORDS]
    texts = father_texts()
    print(f"{len(items)} labelled items of {MIN_WORDS}+ words; open texts for {', '.join(texts)}", flush=True)
    by = collections.defaultdict(collections.Counter)
    leads = []
    for n, item in enumerate(items, 1):
        if item.author == "ANONYMOUS":
            best = max(((score(item.toks, w)[0], a) for a, w in texts.items()), default=(0, None))
            status = f"named: {best[1]}" if best[0] >= FOUND else "anonymous, unmatched"
        elif item.author in texts and score(item.toks, texts[item.author])[0] >= FOUND:
            status = "found"
        elif item.work in COVERED.get(item.author, ()):
            status = "unfound"
        else:
            status = "not checkable"
        by[item.author][status.split(":")[0]] += 1
        if status == "unfound" or status.startswith("named"):
            leads.append({"kind": "citation", "witness": WITNESS, "pos": f"{item.work}:{item.loc}:{n}",
                          "loc": f"{item.work} ch. {item.loc}", "author": item.author, "name": item.label,
                          "formula": "label", "score": 0.0, "context": item.text[:1200],
                          "dfhg_share": 0.0, "prior": None, "credited": None})
        if n % 500 == 0:
            print(f"  {n}/{len(items)} items | overall {n}/{len(items)}", flush=True)
    print(f"\n{'author':14} {'items':>6} {'found':>6} {'unfound':>8} {'not checkable':>14}  other")
    for author, c in sorted(by.items(), key=lambda x: -sum(x[1].values())):
        rest = {k: v for k, v in c.items() if k not in ("found", "unfound", "not checkable")}
        print(f"{author:14} {sum(c.values()):6} {c['found']:6} {c['unfound']:8} {c['not checkable']:14}  {rest or ''}")
    store = LeadStore()
    try:
        kept = store.replace_witness(WITNESS, leads)
    finally:
        store.close()
    print(f"\nstored {kept['stored']} leads (unfound items, and anonymous items given an author) in data/leads.db")


if __name__ == "__main__":
    main()
