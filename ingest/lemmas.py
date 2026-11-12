"""Match on lemmas as well as surface forms (phase 5 of docs/quotation-mining-plan.md).

A writer who paraphrases changes inflections: Eustathius turns Athenaeus's "ἔπινον οἱ ἄνδρες"
into "πίνειν τοὺς ἄνδρας". The shingle matcher compares accent-free surface words, so such a
passage scores low. Matching the same 3-word shingles on lemmas catches it; the shuffled-word
control says whether it also matches by chance.

Lemmas come from a lookup table, accent-free word -> accent-free lemma, made once by
scripts/build_lemmas.py with Stanza's Ancient Greek lemmatizer (run in the project venv, which has
Stanza; this module needs only the table). Each distinct word is lemmatized once, out of context,
which is fast and good enough for matching. Unknown words stay as they are.
"""
from __future__ import annotations

import collections
import json

from ingest.quotations import K, RAW, Witness

TABLE = RAW / "lemmas_grc.json"


def load_table(path=TABLE):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def lemma_witness(witness: Witness, table) -> Witness:
    """The same text with each normalized word replaced by its lemma, and the shingle index rebuilt.
    Positions, locations and original words are unchanged, so results map back one to one."""
    out = object.__new__(Witness)
    out.name, out.orig, out.loc = witness.name, witness.orig, witness.loc
    out.tok = [table.get(t, t) for t in witness.tok]
    out.index = collections.defaultdict(list)
    for i in range(len(out.tok) - K + 1):
        out.index[tuple(out.tok[i:i + K])].append(i)
    return out


def lemmatize(tokens, table):
    return [table.get(t, t) for t in tokens]
