"""Mine verse-translation parallel pairs: Perseus source <-> Gutenberg verse English.

Mines Workstream A data (see docs/roadmap-verse-stylizer.md): fetches a source work
from Perseus (by CTS id) and a public-domain English verse translation from Project
Gutenberg, aligns them with the comparable-text aligner, and writes
{src, tgt, score, citation, src_lang, era, style, form, source} JSONL for fine-tuning
the poetic layer.

Strongly prefer --by-book: align book-by-book (source split by Perseus book
citation, English by "BOOK I/II/..." headers). This is far cleaner than whole-text
alignment because it stops cross-book drift — verse translations expand unevenly, so
the proportional-diagonal assumption of a single whole-text pass is weak. Validated
on Aeneid I (Dryden): clean couplet<->hexameter pairs.

Defaults to LaBSE cross-lingual alignment (cached). --via-mt translates the source
with the routed translator and aligns English<->English instead (emits original src).

Examples:
    # Aeneid, per book, by curated target (verify Gutenberg id first)
    python scripts/mine_verse.py --target aeneid_dryden --gutenberg 228 --by-book \
        --out data/parallel/aeneid_dryden.jsonl

    # Pope's Iliad (Greek source we already have for Workstream B)
    python scripts/mine_verse.py --src-cts tlg0012.tlg001 --greek --gutenberg 6130 \
        --by-book --era archaic --form heroic_couplet --out data/parallel/iliad_pope.jsonl
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import requests

from ingest.base import Connector
from ingest.perseus import PerseusConnector, GreekPerseusConnector
from training.parallel import PerseusParallelMiner
from training.aligner import align_texts, align_texts_via_mt, make_labse_embedder
from training.verse_targets import by_key


_PG_URLS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]
_PG_START = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", re.I | re.S)
_PG_END = re.compile(r"\*\*\*\s*END OF.*?\*\*\*", re.I | re.S)
# A book header: BOOK/GEORGIC + optional "the" + a roman numeral OR an ordinal word
# ("THE EIGHTH"). Editions vary (Aeneid/Homer use roman; Brookes More's Ovid uses
# word-ordinals in the bodies; Dryden's Georgics uses "GEORGIC I").
# Case-SENSITIVE keyword (only the m flag): real book headers are all-caps, so this
# excludes mid-prose strays like "Book of" / "Book vi" that would fragment the run.
_BOOK_HDR = re.compile(
    r"(?m)^[ \t]*(?:BOOK|GEORGIC)[ \t]+(?:THE[ \t]+)?"
    r"([IVXLC]+|[A-Za-z]+(?:-[A-Za-z]+)?)\b")
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_ORDINALS = {w: i + 1 for i, w in enumerate([
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
    "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
    "twenty-first", "twenty-second", "twenty-third", "twenty-fourth"])}


def fetch_gutenberg(book_id: int) -> str:
    """Download a Gutenberg plain-text ebook and strip its boilerplate header/footer.
    Retries each URL (Gutenberg occasionally rate-limits a single hit)."""
    sess = requests.Session()
    sess.headers.update({"User-Agent": "LatinReader-Research/1.0 (scholarly)"})
    text = None
    for tmpl in _PG_URLS:
        url = tmpl.format(id=book_id)
        for _ in range(3):
            try:
                r = sess.get(url, timeout=60)
                if r.ok and len(r.text) > 1000:
                    text = r.text
                    break
            except requests.RequestException:
                continue
        if text:
            break
    if text is None:
        raise SystemExit(f"Could not fetch Gutenberg #{book_id} (tried {len(_PG_URLS)} URLs)")
    m1 = _PG_START.search(text)
    if m1:
        text = text[m1.end():]
    m2 = _PG_END.search(text)
    if m2:
        text = text[:m2.start()]
    return text.strip()


def _roman_to_int(s: str) -> int:
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = _ROMAN.get(ch, 0)
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


def _book_number(token: str):
    """Parse a captured header token (roman numeral or ordinal word) to an int, or
    None if it's neither (e.g. a false 'BOOK' match in prose)."""
    t = token.strip().lower().rstrip(".")
    if t and all(c in "ivxlc" for c in t):
        return _roman_to_int(t)
    return _ORDINALS.get(t.replace(" ", "-"))


def _source_miner(greek: bool) -> PerseusParallelMiner:
    repo = "PerseusDL/canonical-greekLit" if greek else "PerseusDL/canonical-latinLit"
    return PerseusParallelMiner(repo=repo, src_tag=("grc" if greek else "lat"))


def fetch_source(cts: str, greek: bool) -> str:
    """Fetch a Perseus work by CTS id and return its plain source text (whole work)."""
    connector = GreekPerseusConnector() if greek else PerseusConnector()
    raw_meta, parts = connector.fetch(cts)
    doc = Connector.build_document(raw_meta, parts)
    return " ".join(seg.latin_text for seg in doc.iter_segments())


def fetch_source_books(cts: str, greek: bool) -> dict:
    """Return {book_number: text} from the Perseus citation structure.

    Groups leaves by their 'book' citation level (Aeneid/Homer are book-level leaves;
    deeper book.chapter works get their chapters concatenated within each book)."""
    m = _source_miner(greek)
    group, work = m._parse(cts)
    files = [f["name"] for f in m._list_dir(f"data/{group}/{work}") if f.get("type") == "file"]
    secs = m._sections(group, work, m._pick(files, "grc" if greek else "lat"))
    books: dict = {}
    for chain, text in secs:
        bn = None
        for sub, n in chain:
            if sub == "book":
                bn = n          # value of the (innermost) 'book' level
        if bn is None and chain:
            bn = chain[-1][1]
        books.setdefault(str(bn), []).append(text)
    return {k: " ".join(v) for k, v in books.items()}


def split_english_books(text: str) -> dict:
    """Split a Gutenberg verse text into {book_number: text} on 'BOOK <roman>' headers.

    A work's headers often appear more than once (a contents/argument list, then the
    bodies) and editions mix roman and word-ordinal styles, with stray "Book ..."
    matches in prose/endnotes. We split the header hits into maximal ascending-by-value
    runs and keep the LONGEST (latest on a tie) — that's the real sequence of book
    bodies, robust to a TOC duplicate and to junk matches."""
    marks = [(mm.start(), n) for mm in _BOOK_HDR.finditer(text)
             if (n := _book_number(mm.group(1))) and 1 <= n <= 50]
    if not marks:
        return {}
    runs, cur = [], [marks[0]]
    for pos, n in marks[1:]:
        if n > cur[-1][1]:
            cur.append((pos, n))
        else:
            runs.append(cur)
            cur = [(pos, n)]
    runs.append(cur)
    # Pick the run spanning the most text: book bodies are spread across the work,
    # while a contents/argument list clusters its (same-count) headers in a few KB.
    best = max(runs, key=lambda r: (r[-1][0] - r[0][0], len(r)))
    out: dict = {}
    for i, (pos, num) in enumerate(best):
        end = best[i + 1][0] if i + 1 < len(best) else len(text)
        out[str(num)] = text[pos:end]
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", help="curated key from training/verse_targets.py")
    ap.add_argument("--src-cts", help="Perseus CTS id (group.work) for the source")
    ap.add_argument("--src-file", help="local source text file (whole-text mode only)")
    ap.add_argument("--greek", action="store_true", help="source is Greek (greekLit repo)")
    ap.add_argument("--gutenberg", type=int, help="Gutenberg ebook id for the English verse")
    ap.add_argument("--eng-file", help="local English text file (overrides --gutenberg)")
    ap.add_argument("--by-book", action="store_true",
                    help="align book-by-book (recommended; needs a CTS source)")
    ap.add_argument("--era", default="classical", help="source language stage tag")
    ap.add_argument("--form", default="heroic_couplet", help="English verse form tag")
    ap.add_argument("--style", default="verse", help="style tag written to each pair")
    ap.add_argument("--labse", action="store_true",
                    help="LaBSE cross-lingual align (default); --via-mt for translate-then-align")
    ap.add_argument("--via-mt", action="store_true",
                    help="translate source with the routed model, then align English<->English")
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="min bead similarity to keep (raise for cleaner/fewer pairs)")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    args = ap.parse_args()

    cts, greek, era, form, src_lang = args.src_cts, args.greek, args.era, args.form, None
    if args.target:
        t = by_key(args.target)
        if not t:
            ap.error(f"unknown target {args.target!r}")
        cts = cts or t.source_cts
        greek = greek or t.greek_repo
        era, form, src_lang = t.era, t.form, t.src_lang
        if args.gutenberg is None and args.eng_file is None and t.gutenberg_id:
            args.gutenberg = t.gutenberg_id
            print(f"Using best-guess Gutenberg id {t.gutenberg_id} — VERIFY this edition.")
    src_lang = src_lang or ("grc" if greek else "la")

    # English text.
    if args.eng_file:
        eng_text = open(args.eng_file, encoding="utf-8").read()
    elif args.gutenberg:
        print(f"Fetching Gutenberg #{args.gutenberg}...")
        eng_text = fetch_gutenberg(args.gutenberg)
    else:
        ap.error("need --gutenberg or --eng-file")

    # Build the list of (citation, src_text, eng_text) segments to align.
    segments = []
    if args.by_book:
        if not cts:
            ap.error("--by-book needs --src-cts or --target (book structure from Perseus)")
        print(f"Fetching source {cts} by book ({'greek' if greek else 'latin'})...")
        src_books = fetch_source_books(cts, greek)
        eng_books = split_english_books(eng_text)
        if not eng_books:
            ap.error("no 'BOOK <roman>' headers found in the English text; omit --by-book")
        common = sorted((b for b in src_books if b in eng_books), key=int)
        print(f"source books={len(src_books)} english books={len(eng_books)} common={len(common)}")
        segments = [(b, src_books[b], eng_books[b]) for b in common]
    else:
        src_text = (open(args.src_file, encoding="utf-8").read() if args.src_file
                    else fetch_source(cts, greek) if cts else None)
        if src_text is None:
            ap.error("need --src-cts, --src-file, or --target")
        segments = [("all", src_text, eng_text)]

    # Build the aligner backend once.
    via_mt = args.via_mt and not args.labse
    if via_mt:
        from core.embedder import Embedder
        from pipeline import Library
        embedder, translator = Embedder(), Library._build_translator(src_lang, era)
    else:
        embedder, translator = make_labse_embedder(), None

    all_pairs = []
    for cit, s, t in segments:
        if via_mt:
            ps = align_texts_via_mt(s, t, translator, embedder, src_lang=src_lang,
                                    threshold=args.threshold)
        else:
            ps = align_texts(s, t, embedder, src_lang=src_lang, threshold=args.threshold)
        for p in ps:
            all_pairs.append((cit, p))
        print(f"  book {cit}: {len(ps)} pairs")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    source_label = args.target or cts or "verse"
    with open(args.out, "w", encoding="utf-8") as fh:
        for cit, p in all_pairs:
            fh.write(json.dumps({
                "src": p.src, "tgt": p.tgt, "score": p.score, "citation": cit,
                "src_lang": src_lang, "era": era, "style": args.style,
                "form": form, "source": f"verse:{source_label}",
            }, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_pairs)} verse pairs -> {args.out}")


if __name__ == "__main__":
    main()
