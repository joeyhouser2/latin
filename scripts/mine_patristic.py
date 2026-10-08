"""Mine a patristic Latin<->English parallel corpus: the 200s-900s "Loeb gap".

The two halves are public domain and already reachable by connectors we have:

  English  CCEL's Ante-Nicene / Nicene & Post-Nicene Fathers (training/ccel.py)
  Latin    The Latin Library (ingest/latin_library.py)

They share no citation scheme, so pairing is done by the comparable-text LaBSE
aligner (training/aligner.py) -- the same path that handled Hesiod and the
Homeric Hymns, where no shared citation level existed either.

Latin Library rather than Corpus Corporum for the Latin: Corpus Corporum's
display_text.php returns only a work's default loaded section, so long works
come back truncated.

Each MANIFEST row maps one CCEL work (matched by a lowercase substring of its
title) to a Latin Library slug. Verified per work before anything is written:

  * both sides must fetch and be long enough to be a whole work;
  * the English/Latin word ratio must look like a translation (a wrong pairing
    usually shows up here first);
  * the yield -- pairs over Latin sentences -- must clear --min-yield, which
    catches a mismatch the ratio gate let through.

A work failing a gate is reported and skipped, never written. LaBSE loads once
and is reused across works (loading per work dominated the runtime otherwise).

Example:
    python scripts/mine_patristic.py --out data/parallel/patristic_latin.jsonl
    python scripts/mine_patristic.py --volume anf03 --limit 3 --dry-run
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.ccel import CCELExtractor
from training.aligner import align_texts, make_labse_embedder
from ingest.latin_library import LatinLibraryConnector
from core.segmenter import segment_text

LL = "https://www.thelatinlibrary.com/tertullian/tertullian.{slug}.shtml"

# (ccel_volume, ccel_title_substring, latin_library_slug)
# ANF vol 3 = Tertullian. Titles are matched case-insensitively as substrings of
# CCEL's own div2 titles, so they survive small punctuation differences.
MANIFEST = [
    ("anf03", "apology",                  "apol"),
    ("anf03", "on idolatry",              "idololatria"),
    ("anf03", "de spectaculis",           "spect"),
    ("anf03", "de corona",                "corona"),
    ("anf03", "to scapula",               "scapulam"),
    ("anf03", "ad nationes",               "nationes"),
    ("anf03", "answer to the jews",       "iudaeos"),
    ("anf03", "soul's testimony",         "testimonia"),
    ("anf03", "treatise on the soul",     "anima"),
    ("anf03", "prescription against heretics", "praescrip"),
    ("anf03", "against hermogenes",       "herm"),
    ("anf03", "against the valentinians", "valentinianos"),
    ("anf03", "flesh of christ",          "carne"),
    ("anf03", "resurrection of the flesh", "resurrectione"),
    ("anf03", "against praxeas",          "praxean"),
    ("anf03", "scorpiace",                "scorpiace"),
    ("anf03", "against all heresies",     "haereses"),
    ("anf03", "on repentance",            "paen"),
    ("anf03", "on baptism",               "baptismo"),
    ("anf03", "on prayer",                "oratione"),
    ("anf03", "ad martyras",              "martyres"),
    ("anf03", "on patience",              "patientia"),
]


def _words(text):
    return len(text.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/parallel/patristic_latin.jsonl")
    ap.add_argument("--volume", action="append",
                    help="restrict to these CCEL volumes (default: all in the manifest)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N works")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-chars", type=int, default=25)
    ap.add_argument("--max-merge", type=int, default=2)
    ap.add_argument("--min-words", type=int, default=400,
                    help="skip a side shorter than this (truncated fetch / stub page)")
    ap.add_argument("--ratio", default="0.9,2.6",
                    help="allowed English/Latin word ratio, as LOW,HIGH")
    ap.add_argument("--min-yield", type=float, default=0.15,
                    help="skip a work whose pairs/Latin-sentences falls below this")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and run the gates, but do not align or write")
    args = ap.parse_args()

    lo, hi = (float(x) for x in args.ratio.split(","))
    rows = [r for r in MANIFEST if not args.volume or r[0] in args.volume]
    if args.limit:
        rows = rows[: args.limit]

    ccel, ll = CCELExtractor(), LatinLibraryConnector()
    volumes = {}          # volume -> {lowercased title: joined english}
    embedder = None       # LaBSE, loaded on first real alignment

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fh = None if args.dry_run else open(args.out, "w", encoding="utf-8")

    print(f"{'work':<34}{'EN w':>8}{'LA w':>8}{'ratio':>7}{'pairs':>7}{'yield':>7}  note")
    total = 0
    kept = skipped = 0
    try:
        for volume, title_sub, slug in rows:
            label = f"{slug}"
            if volume not in volumes:
                print(f"  fetching CCEL {volume} ...")
                volumes[volume] = {
                    w.title.lower(): "\n\n".join(t for _h, t in w.chapters)
                    for w in ccel.works(volume)
                }
            en = next((text for t, text in volumes[volume].items() if title_sub in t), None)
            if en is None:
                print(f"{label:<34}{'-':>8}{'-':>8}{'-':>7}{'-':>7}{'-':>7}  SKIP no CCEL work matching {title_sub!r}")
                skipped += 1
                continue

            try:
                _meta, sections = ll.fetch(LL.format(slug=slug))
            except Exception as exc:                      # 404, encoding, timeout
                print(f"{label:<34}{_words(en):>8}{'-':>8}{'-':>7}{'-':>7}{'-':>7}  SKIP latin fetch failed: {type(exc).__name__}")
                skipped += 1
                continue
            la = "\n\n".join(t for _l, t in sections)

            ew, lw = _words(en), _words(la)
            ratio = ew / max(1, lw)
            gate = ""
            if ew < args.min_words or lw < args.min_words:
                gate = f"SKIP too short (min {args.min_words} words)"
            elif not (lo <= ratio <= hi):
                gate = f"SKIP ratio outside {lo}-{hi} (likely mispaired or truncated)"
            if gate:
                print(f"{label:<34}{ew:>8,}{lw:>8,}{min(ratio,999):>7.2f}{"-":>7}{"-":>7}  {gate}")
                skipped += 1
                continue
            if args.dry_run:
                print(f"{label:<34}{ew:>8,}{lw:>8,}{ratio:>7.2f}{'-':>7}{'-':>7}  ok (dry run)")
                continue

            if embedder is None:
                embedder = make_labse_embedder()
            pairs = align_texts(la, en, embedder, src_lang="la",
                                threshold=args.threshold, min_chars=args.min_chars,
                                max_merge=args.max_merge)
            n_sent = max(1, len([s for s in segment_text(la, lang="la")
                                 if len(s.strip()) > args.min_chars]))
            yield_ = len(pairs) / n_sent
            if yield_ < args.min_yield:
                print(f"{label:<34}{ew:>8,}{lw:>8,}{ratio:>7.2f}{len(pairs):>7,}{yield_:>7.0%}  "
                      f"SKIP yield below {args.min_yield:.0%} -- check the pairing")
                skipped += 1
                continue

            for k, p in enumerate(pairs):
                fh.write(json.dumps({
                    "src": p.src, "tgt": p.tgt,
                    "citation": f"{slug}#{k}",
                    "src_lang": "la", "era": "late_antique",
                    "source": f"CCEL {volume} / TLL tertullian.{slug}",
                    "score": p.score,
                }, ensure_ascii=False) + "\n")
            total += len(pairs)
            kept += 1
            print(f"{label:<34}{ew:>8,}{lw:>8,}{ratio:>7.2f}{len(pairs):>7,}{yield_:>7.0%}  ok")
    finally:
        if fh:
            fh.close()

    print(f"\n{kept} works kept, {skipped} skipped, {total:,} pairs"
          + ("" if args.dry_run else f" -> {args.out}"))


if __name__ == "__main__":
    main()
