"""Mine later Greek texts for quotations of lost historians, using DFHG as the baseline.

    python scripts/mine_quotations.py fetch
    python scripts/mine_quotations.py recall --witness athenaeus
    python scripts/mine_quotations.py reuse --witness suda
    python scripts/mine_quotations.py cites --witness pindar_scholia
    python scripts/mine_quotations.py check "ου φυσει αλιθον" --extant arrian.xml

Downloads stay in data/raw/quotations and results go to data/quotations; nothing
here reads or writes corpus.db. Results are leads for checking by hand, not
discoveries: see docs/quotation-mining-pilot.md.
"""
import argparse
import csv
import json
import pathlib
import random
import re
import statistics
import sys
import time
import urllib.parse

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest.quotations import AMBIGUOUS_NAMES, Witness, author_stems, find_citations, load_dfhg, norm, read_pg, read_plain, read_tei, score, tokenize

RAW = ROOT / "data" / "raw" / "quotations"
OUT = ROOT / "data" / "quotations"
HEADERS = {"User-Agent": "latin-research"}
DFHG_API = "http://www.dfhg-project.org/DFHG/api.php"
DFHG_AUTHORS = "https://raw.githubusercontent.com/DFHG-project/DFHG-project.github.io/master/dfhg_lofts_cite_urns.csv"
GH = "https://raw.githubusercontent.com"
PERSEUS = f"{GH}/PerseusDL/canonical-greekLit/master/data"
FIRST1K = f"{GH}/OpenGreekAndLatin/First1KGreek/master/data"
IA = "https://archive.org/download/{0}/{0}_djvu.txt"
MIN_TOKENS = 12  # shorter fragments match by chance too easily

# name: kind, {file: url}, and a pattern for this witness in DFHG's source field.
WITNESSES = {
    "athenaeus": ("tei", {"athenaeus.xml": f"{PERSEUS}/tlg0008/tlg001/tlg0008.tlg001.perseus-grc4.xml"}, r"Athen(?:\.|ae)"),
    "suda": ("tei", {"suda.xml": f"{FIRST1K}/tlg9010/tlg001/tlg9010.tlg001.1st1K-grc1.xml"}, r"Suid|Suda"),
    "pindar_scholia": ("tei", {f"tlg5034{p}.xml": f"{FIRST1K}/tlg5034/tlg001{p}/tlg5034.tlg001{p}.perseus-grc1.xml" for p in "abcd"},
                       r"Schol[^|]*Pind|Pind[^|]*[Ss]chol"),
    "procopius_catena": ("pg", {"pg087_1.txt": f"{GH}/calfa-co/Patrologia-Graeca/main/PG087_1/PG087_1_text.txt"}, r"Procop[^|]*Gaz"),
    "stobaeus_ocr": ("plain", {f"{i}.txt": IA.format(i) for i in ("ioannisstobaeiec00stobuoft", "joannisstobaeian03stovuoft", "joannisstobaeian04stovuoft")}, r"Stob"),
}


def download(url, path):
    if path.exists():
        return True
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=120)
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            return True
        except requests.RequestException as error:
            last_error = error
            time.sleep(3 * (attempt + 1))
    print(f"  failed: {url}: {last_error}")
    return False


def fetch(_args):
    """Download witnesses and DFHG. Files already present are kept, so a rerun resumes."""
    for name, (_, files, _) in WITNESSES.items():
        print(name, flush=True)
        for filename, url in files.items():
            download(url, RAW / "witnesses" / filename)
    authors_csv = RAW / "dfhg_authors.csv"
    download(DFHG_AUTHORS, authors_csv)
    with authors_csv.open(encoding="utf-8") as f:
        authors = sorted({row["DFHG Authors"].strip() for row in csv.DictReader(f)})
    directory = RAW / "dfhg"
    directory.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, author in enumerate(authors):
        path = directory / (urllib.parse.quote(author, safe="") + ".json")
        if path.exists():
            continue
        try:
            response = requests.get(DFHG_API, params={"author": author}, headers=HEADERS, timeout=60)
            response.raise_for_status()
            path.write_text(json.dumps(response.json(), ensure_ascii=False), encoding="utf-8")
        except (requests.RequestException, ValueError):  # the API answers some names with an HTML usage page
            failed.append(author)
        time.sleep(0.5)  # a small academic server
        if i % 50 == 0:
            print(f"DFHG {i}/{len(authors)}", flush=True)
    print(f"DFHG authors {len(authors)}, failed {len(failed)}: {failed}")


def load_witness(name):
    kind, files, _ = WITNESSES[name]
    paths = [RAW / "witnesses" / f for f in files]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        sys.exit(f"Missing {missing}: run `mine_quotations.py fetch` first.")
    parts = {"tei": read_tei, "plain": read_plain}.get(kind, lambda p: read_pg(p[0]))(paths)
    witness = Witness(name, parts)
    print(f"{name}: {len(witness.tok)} Greek words", flush=True)
    return witness


def fragments():
    found = load_dfhg(RAW / "dfhg")
    if not found:
        sys.exit("No DFHG data: run `mine_quotations.py fetch` first.")
    return found


def write_results(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {path.relative_to(ROOT)}")


def recall(args):
    """Of the fragments DFHG credits to this witness, how many does the matcher find?"""
    witness = load_witness(args.witness)
    pattern = WITNESSES[args.witness][2]
    gold = [f for f in fragments() if len(f["toks"]) >= MIN_TOKENS and re.search(pattern, f["witness"])]
    if not gold:
        sys.exit("DFHG credits no usable fragments to this witness.")
    rng = random.Random(27)
    found, control = [], []
    for fragment in gold:
        found.append(score(fragment["toks"], witness)[0])
        shuffled = fragment["toks"][:]
        rng.shuffle(shuffled)
        control.append(score(shuffled, witness)[0])
    print(f"known fragments: {len(gold)}; median coverage {statistics.median(found):.3f}")
    for threshold in (.2, .3, .5, .7, .9):
        print(f"  coverage >= {threshold}: found {sum(s >= threshold for s in found) / len(gold):.3f}, "
              f"shuffled control {sum(s >= threshold for s in control) / len(gold):.3f}")


def reuse(args):
    """DFHG fragments found in a witness, split by whether DFHG names that witness."""
    witness = load_witness(args.witness)
    pattern = WITNESSES[args.witness][2]
    listed, unlisted = 0, []
    for fragment in fragments():
        if len(fragment["toks"]) < MIN_TOKENS:
            continue
        coverage, where = score(fragment["toks"], witness)
        if coverage < args.min_coverage:
            continue
        if re.search(pattern, fragment["witness"]):
            listed += 1
        else:
            unlisted.append({"loc": witness.loc[where], "coverage": round(coverage, 2), "author": fragment["author"], "fragment": fragment["n"],
                             "dfhg_source": fragment["witness"], "fragment_text": fragment["text"],
                             "context": witness.context(where - 10, where + len(fragment["toks"]) + 10)})
    print(f"source listed by DFHG: {listed}; not listed: {len(unlisted)} (an upper bound, see the pilot report)")
    write_results(f"reuse_{args.witness}.json", unlisted)


def cites(args):
    """Named citations of DFHG authors whose surrounding words match no DFHG fragment."""
    witness = load_witness(args.witness)
    all_fragments = fragments()
    stems = author_stems(all_fragments, frozenset() if args.include_ambiguous else AMBIGUOUS_NAMES)
    found = find_citations(witness, stems, all_fragments)
    leads = [c for c in found if c["dfhg_share"] < args.max_dfhg_share]
    print(f"authors with a Greek name: {len(stems)}; citations: {len(found)}; unmatched leads: {len(leads)}")
    write_results(f"cites_{args.witness}.json", leads)


def check(args):
    """Does a phrase occur anywhere in DFHG's records, or in the given extant texts?"""
    records = fragments()
    extant = ""
    for path in args.extant:
        extant += " " + " ".join(n for _, n in tokenize(re.sub(r"<[^>]+>", " ", pathlib.Path(path).read_text(encoding="utf-8", errors="replace"))))
    for phrase in args.phrases:
        key = " ".join(n for _, n in tokenize(phrase))
        hits = [(f["author"], f["n"]) for f in records if key in " ".join(n for _, n in tokenize(f["blob"]))]
        print(f"{norm(phrase)!r}: DFHG {hits[:6] or 'no'}; extant {'yes' if key in extant else 'no'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="download witnesses and DFHG").set_defaults(run=fetch)
    for name, run, helptext in (("recall", recall, "measure recall on known fragments"),
                                ("reuse", reuse, "find DFHG fragments in a witness"),
                                ("cites", cites, "find named citations DFHG lacks")):
        sub = commands.add_parser(name, help=helptext)
        sub.add_argument("--witness", required=True, choices=sorted(WITNESSES))
        sub.set_defaults(run=run)
        if name == "reuse":
            sub.add_argument("--min-coverage", type=float, default=.6)
        if name == "cites":
            sub.add_argument("--max-dfhg-share", type=float, default=.1)
            sub.add_argument("--include-ambiguous", action="store_true", help="keep authors whose names mostly mean someone else")
    sub = commands.add_parser("check", help="look for phrases in DFHG and extant texts")
    sub.add_argument("phrases", nargs="+")
    sub.add_argument("--extant", nargs="*", default=[], help="TEI or text files of the author's surviving works")
    sub.set_defaults(run=check)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
