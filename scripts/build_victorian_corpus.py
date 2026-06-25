"""Build a (modern, victorian) parallel corpus to train the standalone stylizer (C).

The insight: we already own a large, authentic 19th-century English corpus — the
gold `tgt` sides of our existing parallel data are Loeb/Murray-era translations
(genuinely Victorian prose). So instead of mining fresh Victorian text, we gather
those English sentences and LLM-*modernize* each (the 'modernize' preset), yielding
clean, consistent (modern, victorian) pairs. Training direction is modern -> victorian.

Why modernize rather than victorianize-from-scratch: turning ornate prose into plain
English is a low-hallucination task, and the Victorian side stays authentic (real
translators' prose), not the LLM's guess at "Victorian". This also avoids the verse
fine-tune's failure mode (paraphrastic, inconsistent targets).

Output JSONL: {src: modern, tgt: victorian, src_lang: "en", source}. Train a small
seq2seq (T5/BART) modern->victorian, then load via Seq2SeqStylizer.

Examples:
    # Dry run: just gather/inspect the Victorian sentences (no model load)
    python scripts/build_victorian_corpus.py --dry-run --limit 2000

    # Full run (downloads the local instruct model on first use; use a free GPU)
    CUDA_VISIBLE_DEVICES=1 python scripts/build_victorian_corpus.py \
        --limit 4000 --out data/parallel/victorian_pairs.jsonl
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

# Existing corpora whose English sides are authentic 19th-c. translation prose.
DEFAULT_SOURCES = [
    "data/parallel/perseus_latin.jsonl",
    "data/parallel/perseus_greek.jsonl",
    "data/parallel/first1k_greek.jsonl",
    "data/parallel/homer.sent.jsonl",
    "data/parallel/greek_archaic.sent.jsonl",
]


# Reject non-prose / noisy targets before distillation.
_GREEK = re.compile(r"[Α-Ωα-ωἀ-ῼ]")                       # residual Greek script
_STAGE = re.compile(r"\(\s*(?:Enter|Exit|Exeunt|Aside|to\s+\w+self|sings|aside)\b", re.I)
_SPEAKER = re.compile(r"\b[A-Z]{4,}\b")                    # ALL-CAPS speaker labels (EUCLIO)
_LATINISH = re.compile(r"\b(?:mentul|cinaed|pathic|catamit)\w*", re.I)  # left-in crude transliterations


def _is_clean_prose(t: str) -> bool:
    """True for clean Victorian prose: drop drama (stage directions / speaker
    labels), Greek-script residue, and a few crude left-in transliterations."""
    if _GREEK.search(t) or _STAGE.search(t) or _SPEAKER.search(t) or _LATINISH.search(t):
        return False
    if t.count("(") + t.count(")") > 1:        # heavy parentheticals -> usually apparatus/stage
        return False
    letters = sum(c.isalpha() for c in t)
    return letters >= 0.6 * len(t)             # mostly letters (not tables/numerals/markup)


def gather_victorian(sources, limit=4000, min_chars=40, max_chars=300):
    """Collect deduped, clean English `tgt` sentences in a length band, spread-sampled
    to `limit` (deterministic — every Nth, for source/length diversity)."""
    seen, pool = set(), []
    for path in sources:
        if not os.path.isfile(path):
            print(f"  (skip missing source: {path})")
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = (rec.get("tgt") or "").strip()
            if (min_chars <= len(t) <= max_chars and t not in seen
                    and _is_clean_prose(t)):
                seen.add(t)
                pool.append(t)
    if limit and len(pool) > limit:
        step = len(pool) / limit
        pool = [pool[int(i * step)] for i in range(limit)]
    return pool


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", help="JSONL files with a `tgt` field "
                    "(default: the Victorian-prose translation corpora)")
    ap.add_argument("--out", default="data/parallel/victorian_pairs.jsonl")
    ap.add_argument("--limit", type=int, default=4000, help="max sentences to process")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=300)
    ap.add_argument("--model", default=None, help="local instruct model (HF id or path)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only gather/preview the Victorian sentences; do not run the LLM")
    args = ap.parse_args()

    sources = args.sources or DEFAULT_SOURCES
    victorian = gather_victorian(sources, args.limit, args.min_chars, args.max_chars)
    print(f"Gathered {len(victorian)} Victorian sentences from {len(sources)} source(s).")
    if not victorian:
        raise SystemExit("No sentences gathered — check --sources paths.")

    if args.dry_run:
        print("\nSamples (dry run, no modernization):")
        for v in victorian[:5]:
            print("  -", v[:120])
        return

    from core.stylizer import LocalLLMStylizer
    stylizer = LocalLLMStylizer(model_name=args.model) if args.model else LocalLLMStylizer()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, vic in enumerate(victorian, 1):
            modern = stylizer.stylize(vic, preset="modernize")
            # Keep only pairs where modernization actually changed the text.
            if modern and modern.strip() and modern.strip().lower() != vic.strip().lower():
                fh.write(json.dumps({"src": modern, "tgt": vic, "src_lang": "en",
                                     "source": "victorian_distill"}, ensure_ascii=False) + "\n")
                kept += 1
            if i % 50 == 0:
                fh.flush()
                print(f"  {i}/{len(victorian)}  (kept {kept})")
    print(f"\nWrote {kept} (modern, victorian) pairs -> {args.out}")
    print("Next: train a seq2seq modern->victorian, then Seq2SeqStylizer(model_name=...).")


if __name__ == "__main__":
    main()
