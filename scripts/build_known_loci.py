"""Build the table of already-collected passages, and measure it.

    python scripts/build_known_loci.py              # download (once), build, report
    python scripts/build_known_loci.py --measure    # also: coverage on known fragments, and current leads

Sources: DFHG's source fields (already downloaded by mine_quotations.py) and the
source lines of Jacoby's FGrHist I-II and III B (1923-57), from the open scans on
archive.org. See ingest/known_loci.py for what is parsed and why.

PRIVATE RESEARCH USE. The downloaded scan text and the table built from it are
for this library's own use and are not to be published or redistributed. They
stay in gitignored folders (data/raw/quotations, data/quotations).
"""
import argparse
import json
import pathlib
import re
import sqlite3
import sys

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest.known_loci import (TOLERANCE, as_rows, from_dfhg, from_jacoby, name_key, near,  # noqa: E402
                               suda_headwords, table_author, witness_locus)
from ingest.quotations import RAW, WITNESSES, load_dfhg, load_witness, score                     # noqa: E402

# FGrHist text volumes on archive.org that anyone may download (not lending-only):
# I (1923), II A (1926), II B (1929) from Vanderbilt; the 1957 reprint of I with addenda (its OCR
# reads more source lines) and III B (1950, authors on single cities, nos. 297-607) from Claremont
# School of Theology. III A and III C exist there only as lending-only copies, so they are not used.
FGRHIST = {"30081002444507": "I", "diefragmentederg0001unse_u5g3": "I (1957)",
           "30081009156641": "II A", "30081009156740": "II B", "diefragmentederg0000unse": "III B"}
DIR = RAW / "fgrhist"
OUT = ROOT / "data" / "quotations" / "known_loci.jsonl"
NOTICE = ("Private research use only. OCR text of Jacoby, Die Fragmente der griechischen Historiker I-III B\n"
          "(1923-57; the 1950s volumes are still in copyright), from archive.org (Vanderbilt University and\n"
          "Claremont School of Theology scans), kept for building a table of source references.\n"
          "Not to be published or redistributed.\n")
MAPPED = tuple(TOLERANCE)


def fetch():
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "NOTICE.txt").write_text(NOTICE, encoding="utf-8")
    for ident in FGRHIST:
        path = DIR / f"{ident}.txt"
        if path.exists():
            continue
        r = requests.get(f"https://archive.org/download/{ident}/{ident}_djvu.txt",
                         headers={"User-Agent": "latin-research"}, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
        print(f"downloaded {ident} ({len(r.content) / 1e6:.1f} MB)")


def build():
    jacoby = []
    for ident, vol in FGRHIST.items():
        rows = from_jacoby((DIR / f"{ident}.txt").read_text(encoding="utf-8", errors="replace"), vol)
        print(f"FGrHist {vol}: {len(rows):,} source lines, {len({r.author for r in rows})} authors")
        jacoby += rows
    fragments = load_dfhg(RAW / "dfhg")
    dfhg = from_dfhg(fragments)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in as_rows(jacoby + dfhg):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(jacoby) + len(dfhg):,} rows to {OUT.relative_to(ROOT)}")
    for w in MAPPED:
        j = {r.locus for r in jacoby if r.witness == w and r.locus}
        d = {r.locus for r in dfhg if r.witness == w and r.locus}
        extra = sum(not any(near(w, x, y, TOLERANCE[w]) for y in d) for x in j)
        print(f"  {w}: passages from FGrHist {len(j)}, from DFHG {len(d)}; FGrHist passages DFHG lacks: {extra}")
    return jacoby, dfhg, fragments


def measure(jacoby, dfhg, fragments):
    """Coverage: are the passages of fragments we know come from a text in the table?"""
    jacoby_authors = {name_key(r.author.split(" ", 1)[1]) for r in jacoby}
    for w in MAPPED:
        wit = load_witness(w)
        heads = suda_headwords(wit) if w == "suda" else None
        gold = [f for f in fragments if len(f["toks"]) >= 12 and re.search(WITNESSES[w][3], f["witness"])]
        found = []
        for f in gold:
            cov, pos = score(f["toks"], wit)
            if cov >= .5:
                found.append((f, witness_locus(wit, pos, heads)))
        print(f"\n{w}: {len(gold)} fragments DFHG credits to it, {len(found)} found by the matcher")
        for name, rows, subset in (("FGrHist", jacoby, lambda f: name_key(f["author"] or "") in jacoby_authors),
                                   ("DFHG", dfhg, lambda f: True), ("both", jacoby + dfhg, lambda f: True)):
            ix = [r.locus for r in rows if r.witness == w and r.locus]
            pool = [loc for f, loc in found if loc and subset(f)]
            for tol in sorted({0, TOLERANCE[w]}):
                hit = sum(any(near(w, loc, k, tol) for k in ix) for loc in pool)
                print(f"  {name:13} tolerance {tol}: {hit}/{len(pool)} = {hit / max(1, len(pool)):.0%}"
                      + ("  (only authors in our FGrHist volumes)" if name.startswith("FGrHist") else ""))

    # A passage is collected *for an author*: Athenaeus 28d printing Agatharchides says nothing
    # about a Clearchus citation on the same page. Names are compared loosely (Latin/German forms).
    print("\nCurrent leads at a passage already collected for the same author (any passage, for comparison):")
    db = sqlite3.connect(f"file:{ROOT / 'data' / 'leads.db'}?mode=ro", uri=True)
    for w in MAPPED:
        wit = load_witness(w)
        heads = suda_headwords(wit) if w == "suda" else None
        ix = [(table_author(r), r.locus) for r in jacoby + dfhg if r.witness == w and r.locus]
        for kind in ("citation", "reuse"):
            rows = db.execute("SELECT key, loc, author FROM leads WHERE witness=? AND kind=?", (w, kind)).fetchall()
            same = anywhere = 0
            for key, loc, author in rows:
                pos = int(key.split("|")[2]) if kind == "citation" else wit.loc.index(loc) if loc in wit.loc else None
                here = witness_locus(wit, pos, heads) if pos is not None else None
                hits = [a for a, k in ix if here and near(w, here, k, TOLERANCE[w])]
                anywhere += bool(hits)
                same += name_key(author or "") in hits
            print(f"  {w} {kind}: {same} of {len(rows)} ({anywhere} at a collected passage of any author)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true", help="report coverage and the effect on current leads")
    args = ap.parse_args()
    fetch()
    tables = build()
    if args.measure:
        measure(*tables)


if __name__ == "__main__":
    main()
