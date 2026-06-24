"""Mine sentence-level parallel pairs with the LaBSE + monotonic-DP aligner.

Two modes, both emitting the JSONL schema `training/finetune.py` consumes
({src, tgt, citation, src_lang, era, source}):

  --refine IN.jsonl
      Take the coarse chapter-/section-level pairs from scripts/mine_parallel.py
      and split each into clean sentence pairs. Metadata is preserved; the
      citation gets a #k suffix per sentence pair. This is the cheap win: it
      upgrades data you already have.

  --src SRC.txt --tgt TGT.txt
      Align two comparable raw-text files that are NOT pre-aligned (e.g. a
      Patrologia Latina work vs its CCEL English translation). Provide
      --src-lang / --era / --source for the output metadata.

Examples:
    python scripts/align_corpus.py --refine data/parallel/cicero.jsonl \
        --out data/parallel/cicero.sent.jsonl

    python scripts/align_corpus.py --src tert.la.txt --tgt tert.en.txt \
        --src-lang la --era late_antique --source "Tertullian, Apologeticum" \
        --out data/parallel/tertullian.jsonl

By default it loads LaBSE (a ~1.8GB download on first run). Use --mini to fall
back to the lighter search embedder for a quick smoke test (lower quality).

--via-mt translates the source with the routed (src-lang, era) model first, then
aligns English<->English. This needs no LaBSE download (defaults to the MiniLM
embedder) and sidesteps weak Latin/Greek embeddings; the model is used only as a
matching heuristic, so kept pairs are still original-src <-> gold-English:

    python scripts/align_corpus.py --refine data/parallel/cicero.jsonl --via-mt \
        --out data/parallel/cicero.sent.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.aligner import align_texts, align_texts_via_mt, make_labse_embedder


def _build_embedder(args):
    from core.embedder import Embedder
    if args.model:
        return Embedder(args.model)
    # English<->English alignment (via-mt) needs no heavy cross-lingual encoder, so
    # default to the light MiniLM there; otherwise LaBSE does the cross-lingual work.
    if args.mini or args.via_mt:
        return Embedder()  # project default MiniLM
    return make_labse_embedder()


def _build_translator(args):
    """Routed fine-tuned translator for (src_lang, era) — same picker the reader uses
    (best model dir for the pair, with Greek diacritic-stripping where needed)."""
    from pipeline import Library
    return Library._build_translator(args.src_lang, args.era)


def _align(src_text, tgt_text, embedder, translator, src_lang, args):
    """Dispatch to cross-lingual or translate-then-align, with shared filter params."""
    common = dict(src_lang=src_lang, threshold=args.threshold,
                  min_chars=args.min_chars, max_merge=args.max_merge)
    if translator is not None:
        return align_texts_via_mt(src_text, tgt_text, translator, embedder, **common)
    return align_texts(src_text, tgt_text, embedder, **common)


def _refine(args, embedder, translator, fh):
    total = 0
    with open(args.refine, encoding="utf-8") as src_fh:
        for n, line in enumerate(src_fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pairs = _align(
                rec["src"], rec["tgt"], embedder, translator,
                rec.get("src_lang", args.src_lang), args,
            )
            base_cit = rec.get("citation", str(n))
            for k, p in enumerate(pairs):
                fh.write(json.dumps({
                    "src": p.src, "tgt": p.tgt,
                    "citation": f"{base_cit}#{k}",
                    "src_lang": rec.get("src_lang", args.src_lang),
                    "era": rec.get("era", args.era),
                    "source": rec.get("source", args.source or "refined"),
                    "score": p.score,
                }, ensure_ascii=False) + "\n")
            total += len(pairs)
            if pairs or n % 50 == 0:
                print(f"  [{n}] {base_cit}: {len(pairs)} sentence pairs")
    return total


def _align_files(args, embedder, translator, fh):
    with open(args.src, encoding="utf-8") as s:
        src_text = s.read()
    with open(args.tgt, encoding="utf-8") as t:
        tgt_text = t.read()
    pairs = _align(src_text, tgt_text, embedder, translator, args.src_lang, args)
    for k, p in enumerate(pairs):
        fh.write(json.dumps({
            "src": p.src, "tgt": p.tgt, "citation": str(k),
            "src_lang": args.src_lang, "era": args.era,
            "source": args.source or f"{args.src} <-> {args.tgt}",
            "score": p.score,
        }, ensure_ascii=False) + "\n")
    return len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine", help="JSONL of coarse pairs to split into sentences")
    ap.add_argument("--src", help="source raw-text file (comparable-texts mode)")
    ap.add_argument("--tgt", help="target (English) raw-text file")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--src-lang", default="la", help="la | grc")
    ap.add_argument("--era", default="unknown", help="language_stage tag")
    ap.add_argument("--source", help="provenance string for output rows")
    ap.add_argument("--threshold", type=float, default=0.45, help="min cosine to keep a pair")
    ap.add_argument("--min-chars", type=int, default=25, help="min source length to keep")
    ap.add_argument("--max-merge", type=int, default=2, help="max sentences per side in a bead")
    ap.add_argument("--via-mt", action="store_true",
                    help="translate the source with the routed (src-lang, era) model, then "
                         "align English<->English (robust, no LaBSE download; assumes one "
                         "language/era per input)")
    ap.add_argument("--mini", action="store_true", help="use the light MiniLM embedder (smoke test)")
    ap.add_argument("--model", help="override the embedder model name")
    args = ap.parse_args()

    if not args.refine and not (args.src and args.tgt):
        ap.error("provide --refine IN.jsonl, or both --src and --tgt")

    embedder = _build_embedder(args)
    translator = _build_translator(args) if args.via_mt else None
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        total = (_refine(args, embedder, translator, fh) if args.refine
                 else _align_files(args, embedder, translator, fh))

    print(f"\nWrote {total} sentence pairs -> {args.out}")


if __name__ == "__main__":
    main()
