"""Mine later Greek texts for quotations of lost historians, using DFHG as the baseline.

    python scripts/mine_quotations.py run --witness suda --witness pindar_scholia
    python scripts/mine_quotations.py fetch
    python scripts/mine_quotations.py recall --witness athenaeus
    python scripts/mine_quotations.py reuse --witness suda
    python scripts/mine_quotations.py cites --witness pindar_scholia
    python scripts/mine_quotations.py check "ου φυσει αλιθον" --extant arrian.xml

`run` is what the web app's job queue calls (kind "mine_quotes"): it downloads
whatever is missing, then stores both kinds of lead in data/leads.db for the
Leads tab. A rerun replaces a text's leads, keeping the model's tags on leads
it finds again and every lead you gave a verdict. The other commands are for measuring and poking at the matcher; they
write JSON to data/quotations. Nothing here reads or writes corpus.db.

Results are leads for checking by hand, not discoveries: see
docs/quotation-mining-pilot.md.
"""
import argparse
import csv
import json
import pathlib
import random
import re
import signal
import statistics
import sys
import time
import urllib.parse

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.leads import LeadStore                                      # noqa: E402
from ingest.quotations import (AMBIGUOUS_NAMES, COMMENTED, RAW, WITNESSES, author_priors, author_stems,  # noqa: E402
                               character_names, commented_url, find_citations, is_character, load_dfhg,
                               load_witness, norm, score, tokenize, witness_paths)

OUT = ROOT / "data" / "quotations"
HEADERS = {"User-Agent": "latin-research"}
DFHG_API = "http://www.dfhg-project.org/DFHG/api.php"
DFHG_AUTHORS = "https://raw.githubusercontent.com/DFHG-project/DFHG-project.github.io/master/dfhg_lofts_cite_urns.csv"
MIN_TOKENS = 12        # shorter fragments match by chance too easily
REUSE_MIN = .6         # coverage for a reuse lead
CITE_MAX_SHARE = .1    # a citation whose nearby words DFHG has for that author is already collected


def download(url, path):
    if path.exists():
        return True
    last_error = None
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
    print(f"  failed: {url}: {last_error}", flush=True)
    return False


def fetch_witness(name):
    ok = True
    for path, url in zip(witness_paths(name), WITNESSES[name][2].values()):
        if url is None:                        # re-OCR'd text: run scripts/reocr_scans.py to make it
            ok = path.exists() and ok
            if not path.exists():
                print(f"  missing {path.name}: run scripts/reocr_scans.py {path.stem}", flush=True)
            continue
        ok = download(url, path) and ok
    for work in COMMENTED.get(name, ()):      # the commented work, for its characters' names
        ok = download(commented_url(work), RAW / "commented" / f"{work}.xml") and ok
    return ok


def fetch_dfhg():
    """Pull DFHG author by author. Files already present are kept, so a rerun resumes."""
    authors_csv = RAW / "dfhg_authors.csv"
    if not download(DFHG_AUTHORS, authors_csv):
        return
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
        except (requests.RequestException, ValueError):   # the API answers some names with an HTML usage page
            failed.append(author)
        time.sleep(0.5)  # a small academic server
        if i % 50 == 0:
            print(f"DFHG {i}/{len(authors)}", flush=True)
    print(f"DFHG authors {len(authors)}, failed {len(failed)}: {failed}", flush=True)


def fetch(_args):
    for name in WITNESSES:
        print(name, flush=True)
        fetch_witness(name)
    fetch_dfhg()


def need_witness(name):
    if not fetch_witness(name):
        sys.exit(f"Could not download {name}: see the errors above.")
    witness = load_witness(name)
    print(f"{name}: {len(witness.tok):,} Greek words", flush=True)
    return witness


def fragments():
    found = load_dfhg(RAW / "dfhg")
    if not found:
        sys.exit("No DFHG data: run `mine_quotations.py fetch` first.")
    return found


def reuse_leads(witness, all_fragments, min_coverage=REUSE_MIN):
    """(how many DFHG already credits to this text, leads for the rest)."""
    pattern = WITNESSES[witness.name][3]
    listed, leads = 0, []
    for fragment in all_fragments:
        if len(fragment["toks"]) < MIN_TOKENS:
            continue
        coverage, where = score(fragment["toks"], witness)
        if coverage < min_coverage:
            continue
        if re.search(pattern, fragment["witness"]):
            listed += 1
        else:
            leads.append({"kind": "reuse", "witness": witness.name, "loc": witness.loc[where], "score": round(coverage, 2),
                          "author": fragment["author"], "fragment": fragment["n"], "dfhg_source": fragment["witness"],
                          "fragment_text": fragment["text"],
                          "context": witness.context(where - 10, where + len(fragment["toks"]) + 10)})
    return listed, leads


def citation_leads(witness, all_fragments, include_ambiguous=False, max_share=CITE_MAX_SHARE):
    """Citations whose nearby wording DFHG lacks *for the cited author*, minus names of characters
    in the work a commentary explains. Both choices are measured in docs/quotation-mining-plan.md
    (phase 2): checking wording against all of DFHG lost 49 of Jacoby's additions in the scholia,
    and the character filter removed 100 citations without losing one."""
    stems = author_stems(all_fragments, frozenset() if include_ambiguous else AMBIGUOUS_NAMES)
    characters = character_names(witness.name)
    found = find_citations(witness, stems, all_fragments)
    leads = [dict(c, kind="citation", score=c["dfhg_share"]) for c in found
             if c["author_share"] < max_share and not is_character(c["name"], stems[c["author"]], characters)]
    return len(stems), len(found), leads


def write_results(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {path.relative_to(ROOT)}")


def _raise_interrupt(*_):
    raise KeyboardInterrupt


def run(args):
    """Download what's missing, then store reuse and citation leads for each text in data/leads.db."""
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _raise_interrupt)    # the queue's cancel button
    names = args.witness or list(WITNESSES)
    steps = len(names) + 1
    print(f"=== Mine quotations: {len(names)} texts === {', '.join(names)}", flush=True)
    if not any((RAW / "dfhg").glob("*.json")):
        print("Downloading DFHG (once; about 10 minutes at a polite pace)", flush=True)
        fetch_dfhg()
    all_fragments = fragments()
    print(f"DFHG: {len(all_fragments):,} fragments | overall 1/{steps}", flush=True)
    store = LeadStore(args.leads) if args.leads else LeadStore()
    try:
        for n, name in enumerate(names, 2):
            witness = need_witness(name)
            listed, reuse = reuse_leads(witness, all_fragments)
            stems, cites, leads = citation_leads(witness, all_fragments, args.include_ambiguous)
            author_priors(leads, all_fragments, WITNESSES[name][3])     # the Leads tab's "most promising"
            kept = store.replace_witness(name, reuse + leads)
            print(f"  {name}: {listed} fragments DFHG credits to it, {len(reuse)} reuse leads; "
                  f"{cites} citations of {stems} authors, {len(leads)} unmatched; "
                  f"{kept['dropped']} old leads dropped | overall {n}/{steps}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted: texts finished so far are saved.", flush=True)
        raise SystemExit(130)
    finally:
        store.close()
    print("Done. Open the Leads tab, or sort the new leads with the local model.", flush=True)


def recall(args):
    """Of the fragments DFHG credits to this text, how many does the matcher find?"""
    witness = need_witness(args.witness)
    pattern = WITNESSES[args.witness][3]
    gold = [f for f in fragments() if len(f["toks"]) >= MIN_TOKENS and re.search(pattern, f["witness"])]
    if not gold:
        sys.exit("DFHG credits no usable fragments to this text.")
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
    listed, leads = reuse_leads(need_witness(args.witness), fragments(), args.min_coverage)
    print(f"source listed by DFHG: {listed}; not listed: {len(leads)} (an upper bound, see the pilot report)")
    write_results(f"reuse_{args.witness}.json", leads)


def cites(args):
    stems, found, leads = citation_leads(need_witness(args.witness), fragments(), args.include_ambiguous,
                                         args.max_dfhg_share)
    print(f"authors with a Greek name: {stems}; citations: {found}; unmatched leads: {len(leads)}")
    write_results(f"cites_{args.witness}.json", leads)


def check(args):
    """Does a phrase occur anywhere in DFHG's records, or in the given extant texts?"""
    records = fragments()
    extant = ""
    for path in args.extant:
        text = re.sub(r"<[^>]+>", " ", pathlib.Path(path).read_text(encoding="utf-8", errors="replace"))
        extant += " " + " ".join(n for _, n in tokenize(text))
    for phrase in args.phrases:
        key = " ".join(n for _, n in tokenize(phrase))
        hits = [(f["author"], f["n"]) for f in records if key in " ".join(n for _, n in tokenize(f["blob"]))]
        print(f"{norm(phrase)!r}: DFHG {hits[:6] or 'no'}; extant {'yes' if key in extant else 'no'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("run", help="find leads and store them for the Leads tab")
    sub.add_argument("--witness", action="append", choices=sorted(WITNESSES), help="repeatable; default all")
    sub.add_argument("--include-ambiguous", action="store_true", help="keep authors whose names mostly mean someone else")
    sub.add_argument("--leads", help="lead database (default data/leads.db)")
    sub.set_defaults(run=run)
    commands.add_parser("fetch", help="download every text and DFHG").set_defaults(run=fetch)
    for name, func, helptext in (("recall", recall, "measure recall on known fragments"),
                                 ("reuse", reuse, "find DFHG fragments in a text"),
                                 ("cites", cites, "find named citations DFHG lacks")):
        sub = commands.add_parser(name, help=helptext)
        sub.add_argument("--witness", required=True, choices=sorted(WITNESSES))
        sub.set_defaults(run=func)
        if name == "reuse":
            sub.add_argument("--min-coverage", type=float, default=REUSE_MIN)
        if name == "cites":
            sub.add_argument("--max-dfhg-share", type=float, default=CITE_MAX_SHARE)
            sub.add_argument("--include-ambiguous", action="store_true")
    sub = commands.add_parser("check", help="look for phrases in DFHG and extant texts")
    sub.add_argument("phrases", nargs="+")
    sub.add_argument("--extant", nargs="*", default=[], help="TEI or text files of the author's surviving works")
    sub.set_defaults(run=check)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
