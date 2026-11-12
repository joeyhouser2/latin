"""Phase 5 method test: do lemma shingles find paraphrases that surface shingles miss?

    python scripts/paraphrase_test.py                       # Eustathius against Athenaeus

Eustathius read an unabridged Athenaeus and reworded him constantly, and both texts are here
(Athenaeus clean TEI, Eustathius OCR). The test passages are the places where Eustathius names
Athenaeus ("ὁ Ἀθήναιος", "ὁ Δειπνοσοφιστής", "ἐν Δειπνοσοφισταῖς"): the words around such a name
are very likely Athenaeus reworded. For each, the share of its words found together in Athenaeus
(the matcher's coverage) is computed on surface forms and on lemmas (ingest/lemmas.py). The
control is the same words shuffled: whatever still matches after shuffling matched by chance.

The plan's gate: lemma matching finds more of the test passages than surface matching, at a
threshold where the shuffled control stays near 0.
"""
import argparse
import random
import re
import statistics
import unicodedata
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest.lemmas import lemma_witness, lemmatize, load_table      # noqa: E402
from ingest.quotations import load_witness, score                   # noqa: E402

# Athenaeus by name or by his book. The name is Ἀθήναιος, accented on the eta; "an Athenian" is
# Ἀθηναῖος. Nominative/accusative only ("Ἀθηναίου" is also "of an Athenian"). The OCR often loses
# or garbles the initial alpha ("ϑήναιος", "Ἰθήναιος").
ATHENAEUS_FORMS = {"αθηναιοσ", "αθηναιον", "θηναιοσ", "θηναιον", "ιθηναιοσ", "ιθηναιον"}


def cites_athenaeus(original, token):
    if token.startswith("δειπνοσοφιστ"):
        return True
    return token in ATHENAEUS_FORMS and "ήναι" in unicodedata.normalize("NFC", original).replace("ϑ", "θ").lower()
BEFORE, AFTER = 10, 60


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--witness", default="eustathius")
    ap.add_argument("--source", default="athenaeus")
    args = ap.parse_args()
    table = load_table()
    text, source = load_witness(args.witness), load_witness(args.source)
    source_lemmas = lemma_witness(source, table)
    known = sum(t in table for t in text.tok) / len(text.tok)
    print(f"{args.witness}: {len(text.tok):,} words ({known:.0%} in the lemma table); {args.source}: {len(source.tok):,}")
    positions = [i for i, (o, t) in enumerate(zip(text.orig, text.tok)) if cites_athenaeus(o, t)]
    # One passage per citation; nearby repeats of the name are the same passage.
    passages, last = [], -10 ** 9
    for i in positions:
        if i - last > AFTER:
            passages.append(i)
            last = i
    print(f"passages naming {args.source}: {len(passages)}")
    rng = random.Random(5)
    rows = []
    for i in passages:
        window = text.tok[max(0, i - BEFORE):i + AFTER]
        shuffled = window[:]
        rng.shuffle(shuffled)
        rows.append((score(window, source)[0], score(lemmatize(window, table), source_lemmas)[0],
                     score(shuffled, source)[0], score(lemmatize(shuffled, table), source_lemmas)[0]))
    print(f"median coverage: surface {statistics.median(r[0] for r in rows):.2f}, "
          f"lemma {statistics.median(r[1] for r in rows):.2f}")
    print(f"{'threshold':>9} {'surface':>8} {'lemma':>8} {'shuffled surface':>17} {'shuffled lemma':>15}")
    n = len(rows)
    for thr in (.1, .15, .2, .25, .3, .4, .5):
        found = [sum(r[k] >= thr for r in rows) for k in range(4)]
        print(f"{thr:9.2f} {found[0] / n:8.0%} {found[1] / n:8.0%} {found[2] / n:17.0%} {found[3] / n:15.0%}")
    gained = [i for i, r in zip(passages, rows) if r[1] >= .25 > r[0]]
    print(f"\npassages found only by lemmas at 0.25: {len(gained)}")
    for i in gained[:8]:
        print(f"  {text.loc[i]} | {text.context(i - BEFORE, i + 40)[:220]}")


if __name__ == "__main__":
    main()
