"""Stylize / scan an existing document from the terminal.

Runs the post-translation Stylizer layer (Victorian prose, verse) and CLTK
scansion over a document already in the library, mirroring scripts/ingest.py.

Examples:
    # List documents with their ids
    python scripts/stylize.py list

    # Victorian-prose stylize a document (writes english_styled in the store)
    python scripts/stylize.py 12 --preset victorian_prose

    # Verse: scan the metre first, then render as blank verse
    python scripts/stylize.py 12 --preset verse_blank --scan --meter hexameter

    # Preview only (print, don't write) the first 5 segments
    python scripts/stylize.py 12 --preset verse_couplet --preview 5

    # Translate untranslated segments first, then stylize; lighter local model
    python scripts/stylize.py 12 --translate --preset victorian_prose \
        --model Qwen/Qwen2.5-3B-Instruct

Presets: victorian_prose | verse_blank | verse_couplet
Metres (for --scan): hexameter | pentameter | elegiac
The first run downloads the local instruct model (~several GB) from Hugging Face.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Latin/Greek titles are Unicode; the default Windows console is cp1252 and would
# raise UnicodeEncodeError on print. Force UTF-8 output where supported.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover - older/odd streams
    pass

from pipeline import Library
from core.stylizer import LocalLLMStylizer, StyleUnit, PRESETS


def _print_documents(lib: Library) -> None:
    docs = lib.list_documents()
    if not docs:
        print("No documents in the library yet. Ingest some with scripts/ingest.py.")
        return
    print(f"{len(docs)} document(s):")
    for d in docs:
        stage = (d.language_stage or "unknown").replace("_", " ")
        print(f"  [{d.id:>4}] {d.author or 'Anon.':<22} {d.title}  "
              f"({d.language_name}, {stage}{', verse' if d.genre == 'poetry' else ''})")


def _preview(lib: Library, doc, preset: str, n: int, model: str) -> None:
    """Stylize the first N translated segments and print — no store writes."""
    segs = [s for s in doc.iter_segments() if s.is_translated][:n]
    if not segs:
        print("Nothing to preview: no translated segments. Use --translate first.")
        return
    stylizer = LocalLLMStylizer(model_name=model) if model else LocalLLMStylizer()
    context = {
        "source_lang": doc.language_name,
        "author": doc.author,
        "era": (doc.language_stage or "").replace("_", " ") or None,
    }
    units = [StyleUnit(literal=s.english_text, source=s.latin_text, scansion=s.scansion)
             for s in segs]
    styled = stylizer.stylize_units(units, preset=preset, context=context)
    for s, out in zip(segs, styled):
        print("\n  source :", s.latin_text)
        if s.scansion:
            print("  metre  :", s.scansion)
        print("  literal:", s.english_text)
        print("  styled :", out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc", help="document id to stylize, or 'list'")
    ap.add_argument("--preset", default="victorian_prose", choices=sorted(PRESETS),
                    help="style/verse register to render (default: victorian_prose)")
    ap.add_argument("--scan", action="store_true",
                    help="scan verse metre into each segment before stylizing")
    ap.add_argument("--meter", default="hexameter",
                    choices=["hexameter", "pentameter", "elegiac"],
                    help="metre for --scan (default: hexameter)")
    ap.add_argument("--scan-only", action="store_true",
                    help="only scan the metre; do not stylize")
    ap.add_argument("--translate", action="store_true",
                    help="translate untranslated segments first (uses routed translator)")
    ap.add_argument("--preview", type=int, metavar="N", default=0,
                    help="stylize the first N segments and print without writing")
    ap.add_argument("--model", default=None,
                    help="override the local instruct model (HF id or local path)")
    args = ap.parse_args()

    lib = Library()
    try:
        if args.doc == "list":
            _print_documents(lib)
            return

        try:
            doc_id = int(args.doc)
        except ValueError:
            ap.error("doc must be a numeric document id or 'list'")

        doc = lib.get_document(doc_id)
        if doc is None:
            print(f"No document with id {doc_id}. Run 'list' to see ids.")
            return
        print(f"[{doc.id}] {doc.author or 'Anon.'} — {doc.title} "
              f"({doc.language_name}, {len(list(doc.iter_segments()))} segments)")

        if args.translate:
            n = lib.translate_document(doc_id)
            print(f"Translated {n} segment(s).")

        if args.scan or args.scan_only:
            n = lib.scan_document(doc_id, meter=args.meter)
            print(f"Scanned {n} line(s) as {args.meter}.")
            if args.scan_only:
                return
            doc = lib.get_document(doc_id)   # reload so scansion feeds the stylizer

        if args.preview:
            _preview(lib, doc, args.preset, args.preview, args.model)
            return

        if args.model:
            lib.stylizer = LocalLLMStylizer(model_name=args.model)
        n = lib.stylize_document(doc_id, preset=args.preset)
        print(f"Stylized {n} segment(s) as '{args.preset}' "
              f"(stored in english_styled; literal translation untouched).")
    finally:
        lib.close()
    print("Done.")


if __name__ == "__main__":
    main()
