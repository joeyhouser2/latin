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


def _latin_url(slug_or_url: str) -> str:
    """A manifest row gives either a Tertullian slug or a full URL (other authors)."""
    return slug_or_url if slug_or_url.startswith("http") else LL.format(slug=slug_or_url)

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
    # Augustine's chapter-structured (non-book) works, by full URL.
    ("npnf103", "catechising of the uninstructed",
     "https://www.thelatinlibrary.com/augustine/catechizandis.shtml"),
    ("npnf103", "faith and the creed",
     "https://www.thelatinlibrary.com/augustine/fide.shtml"),
]


# Multi-book works: (ccel_volume, div1 work-title substring, latin URL pattern).
# In the NPNF volumes a <div1> is the work and its <div2 type="Book"> children are
# the books, numbered in @n -- and the Latin side is one page per book, so these
# align book against book, which is both cleaner and cheaper than whole works.
BOOK_MANIFEST = [
    ("npnf101", "the confessions", "https://www.thelatinlibrary.com/augustine/conf{n}.shtml"),
    ("npnf102", "city of god",     "https://www.thelatinlibrary.com/augustine/civ{n}.shtml"),
    ("npnf103", "on the holy trinity", "https://www.thelatinlibrary.com/augustine/trin{n}.shtml"),
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
    book_rows = [r for r in BOOK_MANIFEST if not args.volume or r[0] in args.volume]

    ccel, ll = CCELExtractor(), LatinLibraryConnector()
    volumes = {}          # volume -> {lowercased div2 title: joined english}
    trees = {}            # volume -> parsed, apparatus-stripped volume root
    state = {"total": 0, "kept": 0, "skipped": 0, "embedder": None, "units": 0}

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fh = None if args.dry_run else open(args.out, "w", encoding="utf-8")

    def row(label, ew, lw, ratio, pairs, yld, note):
        f = lambda v, w=8: f"{v:>{w},}" if isinstance(v, int) else f"{v:>{w}}"
        r = f"{min(ratio, 999):>7.2f}" if isinstance(ratio, float) else f"{ratio:>7}"
        y = f"{yld:>7.0%}" if isinstance(yld, float) else f"{yld:>7}"
        print(f"{label:<22}{f(ew)}{f(lw)}{r}{f(pairs, 7)}{y}  {note}")

    def unit(label, en, latin_url, source, cit_tag):
        """Gate one English/Latin pairing, then align and write it."""
        if args.limit and state["units"] >= args.limit:
            return False
        state["units"] += 1
        try:
            _meta, sections = ll.fetch(latin_url)
        except Exception as exc:                      # 404, encoding, timeout
            row(label, _words(en), "-", "-", "-", "-",
                f"SKIP latin fetch failed: {type(exc).__name__}")
            state["skipped"] += 1
            return True
        la = "\n\n".join(t for _l, t in sections)

        ew, lw = _words(en), _words(la)
        ratio = ew / max(1, lw)
        if ew < args.min_words or lw < args.min_words:
            row(label, ew, lw, ratio, "-", "-", f"SKIP too short (min {args.min_words} words)")
            state["skipped"] += 1
            return True
        if not (lo <= ratio <= hi):
            row(label, ew, lw, ratio, "-", "-",
                f"SKIP ratio outside {lo}-{hi} (likely mispaired or truncated)")
            state["skipped"] += 1
            return True
        if args.dry_run:
            row(label, ew, lw, ratio, "-", "-", "ok (dry run)")
            return True

        if state["embedder"] is None:
            state["embedder"] = make_labse_embedder()
        pairs = align_texts(la, en, state["embedder"], src_lang="la",
                            threshold=args.threshold, min_chars=args.min_chars,
                            max_merge=args.max_merge)
        n_sent = max(1, len([s for s in segment_text(la, lang="la")
                             if len(s.strip()) > args.min_chars]))
        yield_ = len(pairs) / n_sent
        if yield_ < args.min_yield:
            row(label, ew, lw, ratio, len(pairs), yield_,
                f"SKIP yield below {args.min_yield:.0%} -- check the pairing")
            state["skipped"] += 1
            return True

        for k, p in enumerate(pairs):
            fh.write(json.dumps({
                "src": p.src, "tgt": p.tgt,
                "citation": f"{cit_tag}#{k}",
                "src_lang": "la", "era": "late_antique",
                "source": source,
                "score": p.score,
            }, ensure_ascii=False) + "\n")
        state["total"] += len(pairs)
        state["kept"] += 1
        row(label, ew, lw, ratio, len(pairs), yield_, "ok")
        return True

    print(f"{'unit':<22}{'EN w':>8}{'LA w':>8}{'ratio':>7}{'pairs':>7}{'yield':>7}  note")
    try:
        # --- whole works (one Latin page each) ------------------------------
        for volume, title_sub, slug in rows:
            if volume not in volumes:
                print(f"  fetching CCEL {volume} ...")
                volumes[volume] = {
                    w.title.lower(): "\n\n".join(t for _h, t in w.chapters)
                    for w in ccel.works(volume)
                }
            en = next((t for title, t in volumes[volume].items() if title_sub in title), None)
            label = (slug.rsplit("/", 1)[-1].replace(".shtml", "")
                     if slug.startswith("http") else slug)
            if en is None:
                row(label, "-", "-", "-", "-", "-",
                    f"SKIP no CCEL work matching {title_sub!r}")
                state["skipped"] += 1
                continue
            if not unit(label, en, _latin_url(slug),
                        f"CCEL {volume} / TLL {label}", label):
                break

        # --- multi-book works (one Latin page per book) ---------------------
        for volume, work_sub, pattern in book_rows:
            if volume not in trees:
                print(f"  fetching CCEL {volume} ...")
                trees[volume] = ccel.fetch_volume(volume)
            books = ccel.books(volume, work_sub, root=trees[volume])
            tag = re.search(r"/([a-z]+)\{n\}", pattern)
            tag = tag.group(1) if tag else work_sub.replace(" ", "")
            if not books:
                row(tag, "-", "-", "-", "-", "-", f"SKIP no books under div1 {work_sub!r}")
                state["skipped"] += 1
                continue
            print(f"  {work_sub}: {len(books)} books in {volume}")
            for n, en in books:
                if not unit(f"{tag}{n}", en, pattern.format(n=n),
                            f"CCEL {volume} / TLL {tag}{n}", f"{tag}{n}"):
                    break
    finally:
        if fh:
            fh.close()

    print(f"\n{state['kept']} units kept, {state['skipped']} skipped, {state['total']:,} pairs"
          + ("" if args.dry_run else f" -> {args.out}"))


if __name__ == "__main__":
    main()
