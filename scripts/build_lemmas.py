"""Build the Greek lemma table used for lemma matching (ingest/lemmas.py).

    latinvenv\\Scripts\\python.exe scripts/build_lemmas.py --witness athenaeus --witness eustathius

Runs Stanza's Ancient Greek lemmatizer (stanza.download("grc") once; about 6 MB of models in
~/stanza_resources) over every distinct word of the given texts, one word at a time, on the GPU if
there is one. The table maps the accent-free word to its accent-free lemma (the commonest one when
several accented forms fold together) and is merged into data/raw/quotations/lemmas_grc.json, so
it grows as texts are added. Needs the project venv: the system Python has no Stanza.
"""
import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest.lemmas import TABLE, load_table                       # noqa: E402
from ingest.quotations import WITNESSES, load_witness, norm         # noqa: E402

BATCH = 4000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--witness", action="append", required=True, choices=sorted(WITNESSES))
    args = ap.parse_args()
    import stanza
    nlp = stanza.Pipeline("grc", processors="tokenize,lemma", tokenize_pretokenized=True, verbose=False)
    table = load_table()
    forms = collections.Counter()
    for name in args.witness:
        w = load_witness(name)
        forms.update(o for o, t in zip(w.orig, w.tok) if t not in table)
    todo = sorted(forms)
    print(f"{len(todo):,} new word forms to lemmatize ({len(table):,} already in the table)", flush=True)
    votes = collections.defaultdict(collections.Counter)
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        doc = nlp([[w] for w in batch])
        for form, sentence in zip(batch, doc.sentences):
            lemma = sentence.words[0].lemma or form
            votes[norm(form)][norm(lemma) or norm(form)] += forms[form]
        print(f"  {min(i + BATCH, len(todo)):,}/{len(todo):,} | overall {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)
    for word, counter in votes.items():
        table[word] = counter.most_common(1)[0][0]
    TABLE.write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    print(f"table: {len(table):,} words -> {TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
