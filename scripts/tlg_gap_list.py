"""List Greek works with no open TEI text, ranked by how likely they are to quote lost works.

Uses TLG's numbering without TLG: the Perseus catalog records works by TLG ID and
often links scans of an edition. Works with a Greek file in Perseus or First1KGreek
are dropped. The catalog is far smaller than TLG, and PTA, PG Corpus and corpus.db
are not checked, so this is a starting point rather than a census.

    python scripts/tlg_gap_list.py
    python scripts/tlg_gap_list.py --catalog path/to/catalog_data --no-ocr-check
"""
import argparse
import collections
import csv
import pathlib
import re
import subprocess
import time
import xml.etree.ElementTree as ET

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADERS = {"User-Agent": "latin-research"}
MODS = "{http://www.loc.gov/mods/v3}"
CATALOG_REPO = "https://github.com/PerseusDL/catalog_data.git"
TEI_REPOS = ("PerseusDL/canonical-greekLit", "OpenGreekAndLatin/First1KGreek")
# Title keywords, in rank order. Rough by nature: most works land in "other".
GENRES = (
    ("scholia", r"\bscholia|^tlg50"),
    ("catena", r"catena"),
    ("anthology", r"anthologi|florileg|sacra parallela|excerpt|eclog|chrestomath|bibliotheca|collectanea|gnomolog|sententiae|apophthegm"),
    ("lexicon", r"lexic|etymolog|onomast|glossa|ethnica|synagoge|de vocabulis"),
    ("commentary", r"comment|hypomnem|in aristotel|in plat|in hom|in hes|in arat|exegesis|expositio|paraphras"),
    ("miscellany", r"quaestion|problemata|mirabil|paradox|varia historia|de fluviis|epitome|strom|praeparatio"),
)
RANK = {genre: i for i, (genre, _) in enumerate(GENRES)}


def get_catalog(path):
    """Clone only the Greek records: other paths in this repo are invalid on Windows."""
    if (path / "mods" / "greekLit").exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--no-checkout", CATALOG_REPO, str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "sparse-checkout", "set", "mods/greekLit"], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "--quiet"], check=True)


def open_tei_works():
    works = set()
    for repo in TEI_REPOS:
        response = requests.get(f"https://api.github.com/repos/{repo}/git/trees/master?recursive=1", headers=HEADERS, timeout=120)
        response.raise_for_status()
        tree = response.json()
        if tree.get("truncated"):
            raise RuntimeError(f"{repo} file listing was truncated")
        for item in tree["tree"]:
            match = re.match(r"data/(tlg\d+)/(tlg\w+)/.*grc.*\.xml$", item["path"])
            if match:
                works.add(match.groups())
    return works


def clean(text):
    return re.sub(r"\s+", " ", text or "").strip().replace(",,", ",")


def read_edition(path):
    root = ET.parse(path).getroot()
    uniform = next((t for info in root.iter(MODS + "titleInfo") if info.get("type") == "uniform" for t in info.iter(MODS + "title")), None)
    first_title = next(root.iter(MODS + "title"), None)
    creator, editor = "", ""
    for name in root.iter(MODS + "name"):
        role = " ".join(r.text or "" for r in name.iter(MODS + "roleTerm")).lower()
        full = ", ".join(p.text or "" for p in name.iter(MODS + "namePart") if p.get("type") != "date")
        if "creator" in role and not creator:
            creator = full
        if "editor" in role and not editor:
            editor = full
    urls = [clean(u.text) for u in root.iter(MODS + "url")]
    return {
        "uniform": clean("".join(uniform.itertext())) if uniform is not None else "",
        "title": clean("".join(first_title.itertext())) if first_title is not None else "",
        "creator": clean(creator), "editor": clean(editor),
        "date": next((clean(d.text) for d in root.iter(MODS + "dateIssued") if d.text and re.search(r"\d{4}", d.text)), ""),
        "archive": next((u for u in urls if "archive.org/details" in u), ""),
        "google": next((u for u in urls if "books.google" in u), ""),
    }


def has_ocr(url):
    """Roughly 1 archive.org item in 5 is image-only, so ask rather than assume."""
    identifier = url.rstrip("/").split("/")[-1]
    for attempt in range(3):
        try:
            response = requests.get(f"https://archive.org/metadata/{identifier}/files", headers=HEADERS, timeout=40)
            response.raise_for_status()
            names = [f.get("name", "") for f in response.json().get("result", [])]
            return "yes" if any(n.endswith("_djvu.txt") for n in names) else ("no" if names else "missing")
        except (requests.RequestException, ValueError):
            time.sleep(3 * (attempt + 1))
    return "error"


def build_rows(catalog):
    editions = collections.defaultdict(list)
    unreadable = 0
    for path in (catalog / "mods" / "greekLit").glob("tlg*/tlg*/*grc*/*.xml"):
        try:
            editions[(path.parts[-4], path.parts[-3])].append(read_edition(path))
        except ET.ParseError:
            unreadable += 1
    open_works = open_tei_works()
    rows = []
    for (author_id, work_id), eds in editions.items():
        if (author_id, work_id) in open_works:
            continue
        title = next((e["uniform"] for e in eds if e["uniform"]), eds[0]["title"])
        author = next((e["creator"] for e in eds if e["creator"]), "")
        haystack = f"{author_id} {title} {author} {eds[0]['title']}".lower()
        genre = next((g for g, pattern in GENRES if re.search(pattern, haystack)), "other")
        best = min(eds, key=lambda e: (not e["archive"], not e["google"], -int((re.search(r"\d{4}", e["date"]) or [0])[0])))
        rows.append({"urn": f"{author_id}.{work_id}", "author": author, "title": title, "cls": genre, "editions": len(eds), "editor": best["editor"],
                     "date": best["date"], "archive": best["archive"], "google": best["google"], "ocr_text": ""})
    rows.sort(key=lambda r: (RANK.get(r["cls"], 99), not r["archive"], r["author"], r["urn"]))
    print(f"catalogued works {len(editions)}, with open TEI {sum(k in open_works for k in editions)}, gaps {len(rows)}, unreadable records {unreadable}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=pathlib.Path, default=ROOT / "data" / "raw" / "quotations" / "catalog_data")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "docs" / "tlg-gap-list.csv")
    parser.add_argument("--no-ocr-check", action="store_true", help="skip asking archive.org about works with a genre label")
    args = parser.parse_args()
    get_catalog(args.catalog)
    rows = build_rows(args.catalog)
    if not args.no_ocr_check:
        checked = {}
        for row in rows:
            if row["cls"] != "other" and row["archive"]:
                if row["archive"] not in checked:
                    checked[row["archive"]] = has_ocr(row["archive"])
                    time.sleep(0.3)
                row["ocr_text"] = checked[row["archive"]]
    with args.out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(collections.Counter(r["cls"] for r in rows).most_common())
    print("OCR among labelled works:", collections.Counter(r["ocr_text"] for r in rows if r["cls"] != "other"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
