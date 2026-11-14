"""Translate untranslated segments for library documents matching a filter,
using the pipeline's per-(language,stage) translator routing.

Resumable: only untranslated segments are touched, and translations are written
in small chunks, so an interrupted run (even mid-document) loses at most the
current chunk — just re-run to continue.

Usage:
    python scripts/translate_pending.py --source-prefix "ALIM ("
    python scripts/translate_pending.py --language la --chunk 200 --batch-size 16
    python scripts/translate_pending.py --doc-id 42
    python scripts/translate_pending.py --doc-id 42 --section-range 2 5
    CUDA_VISIBLE_DEVICES=0 python scripts/translate_pending.py --source-prefix "ALIM ("

German editorial apparatus inside Latin documents (e.g. Dreves/Blume's notes
in the Analecta Hymnica) is detected and translated as German rather than
Latin -- see ingest/german_detect.py. To fix segments translated before that
routing existed, re-translate *only* the German ones (Latin untouched):
    python scripts/translate_pending.py --doc-id 423 --retranslate-german --dry-run
    python scripts/translate_pending.py --source-prefix "Internet Archive (analectahymnica" --retranslate-german
A re-translation first backs corpus.db up (sqlite backup API, safe alongside
other writers) to data/corpus.db.bak-preGermanRetranslate-<timestamp>, or
reuses one taken in the last few hours (the UI queues a job per document).
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time

import sentence_transformers  # noqa: F401  (import order: see harvest script)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import Library
from core import qc
from core.local_llm import DEFAULT_CHAT_MODEL
from ingest.garble_detect import is_garbled, UNTRANSLATABLE_PLACEHOLDER
from ingest.mixed_lang_translate import translate_mixed_batch, _GREEK_CHAR, is_mostly_greek
from ingest.german_detect import is_german


def _is_german_segment(text: str) -> bool:
    """German routing applies to clean, Greek-free segments of Latin docs
    (garbled and Greek-bearing segments keep their own dedicated paths)."""
    return not is_garbled(text) and not _GREEK_CHAR.search(text) and is_german(text)


BACKUP_REUSE_HOURS = 6


def _backup_before_retranslate(db_path: str) -> None:
    """Keep a restorable copy before overwriting translations. Uses the sqlite
    backup API (safe with WAL and concurrent writers), never a file copy.

    The web UI queues one job per selected document, so a recent backup is
    reused rather than writing a ~200 MB copy per document: re-translation
    only rewrites German segments, deterministically, so the backup taken
    before the first job of a batch is the one worth restoring from."""
    folder = os.path.dirname(os.path.abspath(db_path))
    prefix = os.path.basename(db_path) + ".bak-preGermanRetranslate-"
    recent = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(prefix)]
    recent = [f for f in recent if time.time() - os.path.getmtime(f) < BACKUP_REUSE_HOURS * 3600]
    if recent:
        print(f"Using recent backup {max(recent, key=os.path.getmtime)}", flush=True)
        return
    backup = f"{db_path}{prefix[len(os.path.basename(db_path)):]}{time.strftime('%Y%m%d%H%M%S')}"
    src, dest = sqlite3.connect(db_path), sqlite3.connect(backup)
    src.backup(dest)
    dest.close()
    src.close()
    print(f"Backed up corpus to {backup}", flush=True)


def retranslate_german(lib, docs, args):
    """Re-translate already-translated German segments with the German
    translator. Only segments that is_german() flags are read or written;
    every other segment's translation is left exactly as it is."""
    plan = []
    for d in docs:
        if d.language != "la":
            continue
        full = lib.store.get_document(d.id)
        segs = [s
                for section in _section_slice(full, args.section_range)
                for s in sorted(section.segments, key=lambda x: x.order)
                if s.is_translated and _is_german_segment(s.latin_text)]
        if segs:
            plan.append((d, segs))
    total = sum(len(p) for _, p in plan)
    print(f"=== Re-translate German: {len(plan)} docs, {total:,} segments ===")
    for d, segs in plan:
        print(f"  [{d.id}] {(d.source or '')[:50]:50} {len(segs):>6,}")
    if args.dry_run or not total:
        return

    _backup_before_retranslate(lib.db_path)

    tr = lib.german_translator()
    if hasattr(tr, "max_length"):
        tr.max_length = args.max_length
    grand = 0
    t_start = time.time()
    for d, segs in plan:
        for i in range(0, len(segs), args.chunk):
            batch = segs[i:i + args.chunk]
            eng = tr.translate_batch([s.latin_text for s in batch],
                                     batch_size=args.batch_size)
            # An OOM-skipped segment comes back "" -- keep its old
            # translation rather than blanking it.
            pairs = [(s.id, e) for s, e in zip(batch, eng) if e and e.strip()]
            lib.store.set_translations(pairs, reset_styled=True)
            grand += len(batch)
            rate = grand / max(time.time() - t_start, 1e-6)
            print(f"  [{d.id}] {min(i + args.chunk, len(segs)):>6,}/{len(segs):<6,} "
                  f"| overall {grand:,}/{total:,} {rate:.1f} seg/s", flush=True)
    print(f"\nDone. Re-translated {grand:,} German segments in "
          f"{(time.time() - t_start) / 60:.1f} min.")


def _print_sample(doc, pending, tr, model_label, args, allowed_names) -> None:
    """The spot-check gate: translate a few runs of consecutive segments spread across
    the work, show them with their quality flags, and write nothing.

    Runs of consecutive lines (not single scattered ones) because the LLM engine reads
    neighbouring lines as context, so a lone line would not show how it really performs."""
    n = args.sample
    run = min(8, n)
    starts = sorted({int(i * max(len(pending) - run, 0) / max(n // run - 1, 1))
                     for i in range(max(n // run, 1))})
    picked = []
    for st in starts:
        picked.extend(pending[st:st + run])
    picked = picked[:n]
    eng = tr.translate_batch([s.latin_text for s in picked], batch_size=args.batch_size)
    flags = qc.flag_batch([s.latin_text for s in picked], eng, allowed_names)
    print(f"\n=== Sample of {len(picked)} / {len(pending):,} segments of [{doc.id}] "
          f"{doc.title} -- {model_label} ===")
    for s, e, f in zip(picked, eng, flags):
        print(f"  {s.latin_text}\n    -> {e}" + (f"    [FLAGS: {f}]" if f else ""))
    c = qc.summarize(flags)
    print(f"\n  flagged {c['flagged']}/{c['checked']}: "
          + (", ".join(f"{k} {v}" for k, v in c.items() if k not in ('checked', 'flagged')) or "none"))


def _is_papyrus(doc) -> bool:
    """A Latin papyrus, tablet or ostracon from ingest/papyri.py (genre "... papyrus ...")."""
    return doc.language == "la" and "papyrus" in (doc.genre or "")


def _section_slice(doc, section_range):
    """The document's sections narrowed to a 1-based inclusive range.

    Matches the range picker in book_creator's corpus tab, which numbers
    sections from 1 in `ord` order -- so "sections 2-5" means the same thing
    in both projects, and preparing what you are about to print does not mean
    translating the whole work.
    """
    sections = sorted(doc.sections, key=lambda s: s.order)
    if not section_range:
        return sections
    first, last = section_range
    return sections[max(1, first) - 1:min(len(sections), last)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-prefix", default="",
                    help="only docs whose source starts with this (e.g. 'ALIM (')")
    ap.add_argument("--doc-id", type=int, default=None, help="only this document")
    ap.add_argument("--section-range", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    default=None, help="only sections FIRST..LAST of the document, 1-based inclusive, in `ord` order (needs --doc-id). Lets a short printed range be prepared without running the whole work.")
    ap.add_argument("--language", default="", help="only this language (la/grc)")
    ap.add_argument("--retranslate-german", action="store_true",
                    help="instead of translating pending segments, re-translate "
                         "already-translated German segments of Latin docs as "
                         "German (Latin segments are not touched)")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --retranslate-german: only report what would change")
    ap.add_argument("--skip-translated", action="store_true",
                    help="skip docs already known to have a published English "
                         "translation (translation_status == 'translated')")
    ap.add_argument("--skip-done-elsewhere", action="store_true",
                    help="skip documents the shared ledger (data/ledger) records as "
                         "fully translated on another computer")
    ap.add_argument("--engine", choices=["nllb", "llm"], default="nllb",
                    help="llm: stage-aware local LLM (Ollama, pinned to this process's GPU) "
                         "for old vernacular documents NLLB cannot read -- only de/fr/it/nl/"
                         "pl/hu/ru documents are translated; see core/llm_translator.py")
    ap.add_argument("--llm-model", default=os.environ.get("LATIN_TRANSLATE_MODEL", DEFAULT_CHAT_MODEL),
                    help="Ollama model for --engine llm")
    ap.add_argument("--redo", action="store_true",
                    help="also re-translate segments that already have a translation "
                         "(clears their styled version); for replacing NLLB output with --engine llm")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="spot-check gate: translate N segments spread evenly across each "
                         "document, print Latin / English / quality flags side by side, and "
                         "write NOTHING. Look at it before queueing the full run.")
    ap.add_argument("--glossary", default="",
                    help="text file of names, places and terms for the document (with --engine "
                         "llm): added to the translator's briefing, and its capitalised words "
                         "are exempt from the new_name quality flag")
    ap.add_argument("--batch-size", type=int, default=16, help="model batch size")
    ap.add_argument("--chunk", type=int, default=200,
                    help="segments per DB commit (resume granularity)")
    ap.add_argument("--max-length", type=int, default=256,
                    help="cap on tokenization/generation length (lower = less VRAM)")
    args = ap.parse_args()

    # A range means nothing across a whole queue of documents -- each has its
    # own section numbering, so the same 2..5 would mean a different span in
    # every one of them.
    if args.section_range and args.doc_id is None:
        ap.error("--section-range needs --doc-id: section numbers are per-document")

    lib = Library()
    docs = lib.store.list_documents()
    if args.doc_id is not None:
        docs = [d for d in docs if d.id == args.doc_id]
    if args.source_prefix:
        docs = [d for d in docs if d.source and d.source.startswith(args.source_prefix)]
    if args.language:
        docs = [d for d in docs if d.language == args.language]
    if args.skip_translated:
        docs = [d for d in docs if d.translation_status != "translated"]
    if args.engine == "llm":
        # Vernaculars, plus Latin papyri: NLLB loops on their Leiden brackets and
        # gaps, where an LLM briefed on the conventions reads them (core/llm_translator.py).
        # An explicitly named --doc-id is the exception: asking for one Latin work by
        # id is the way to put a single poem through the LLM instead of NLLB.
        keep = [d for d in docs if d.language not in ("la", "grc") or _is_papyrus(d)
                or args.doc_id is not None]
        if len(keep) != len(docs):
            print(f"--engine llm handles vernacular documents, Latin papyri and any Latin/Greek "
                  f"work named with --doc-id; skipping {len(docs) - len(keep)} other document(s)")
        docs = keep
    if args.skip_done_elsewhere:
        # Fully translated on another machine, per the committed ledger
        # (scripts/ledger.py). Anything already finished here is a no-op anyway.
        from core.ledger import done_elsewhere_sources
        done = done_elsewhere_sources()
        before = len(docs)
        docs = [d for d in docs if d.source not in done]
        print(f"Ledger: skipping {before - len(docs)} documents already translated elsewhere")

    if args.retranslate_german:
        retranslate_german(lib, docs, args)
        lib.close()
        return

    # Count work up front.
    plan = []   # (doc, pending_segments)
    for d in docs:
        full = lib.store.get_document(d.id)
        pending = [s
                   for section in _section_slice(full, args.section_range)
                   for s in sorted(section.segments, key=lambda x: x.order)
                   if args.redo or not s.is_translated]
        if pending:
            plan.append((d, pending))
    total = sum(len(p) for _, p in plan)
    print(f"=== Translate pending: {len(plan)} docs, {total:,} segments ===\n")

    grand = 0
    flag_totals = {"checked": 0, "flagged": 0}
    t_start = time.time()
    server = None
    if args.engine == "llm":
        from core.local_llm import PrivateOllama
        from core.llm_translator import LLMTranslator
        os.makedirs("data/joblogs", exist_ok=True)
        server = PrivateOllama(log_path="data/joblogs/ollama-translate.log").__enter__()
        if not server.client.has_model(args.llm_model):
            server.__exit__(None, None, None)
            raise SystemExit(f"model {args.llm_model!r} is not pulled in Ollama")
        print(f"LLM engine: {args.llm_model}")
    glossary_text = ""
    if args.glossary:
        with open(args.glossary, encoding="utf-8") as fh:
            glossary_text = fh.read().strip()
    allowed_names = qc.load_glossary_names(glossary_text)
    for d, pending in plan:
        if server:
            note = "\n".join(x for x in (d.translation_evidence, glossary_text) if x) or None
            tr = LLMTranslator(server.client, args.llm_model, d.language,
                               "papyrus" if _is_papyrus(d) else d.language_stage,
                               title=d.title, author=d.author, century=d.century,
                               genre=d.genre, note=note)
            model_label = f"llm:{args.llm_model}"
            # Greek subscriptions, letters and glosses inside a Latin papyrus
            # get the Greek briefing (core/llm_translator.py, ("grc", "papyrus")).
            # One line per request, without CONTEXT lines: these lines are long and
            # run on into each other, so in a batch the model moves words between
            # numbered lines, and given context lines it translates those as well.
            greek_tr = (LLMTranslator(server.client, args.llm_model, "grc", "papyrus",
                                      title=d.title, author=d.author, century=d.century,
                                      genre=d.genre, note=note, batch=1, context=0)
                        if _is_papyrus(d) else None)
        else:
            tr = lib.translator_for(d.language, d.language_stage)
            model_label = f"nllb:{getattr(tr, 'model_name', 'unknown')}"
            greek_tr = None
        if hasattr(tr, "max_length"):
            tr.max_length = args.max_length
        if args.sample:
            _print_sample(d, pending, tr, model_label, args, allowed_names)
            continue
        n = len(pending)
        t0 = time.time()
        done = 0
        for i in range(0, n, args.chunk):
            batch = pending[i:i + args.chunk]
            # Segments whose source is too corrupted to be real Latin (e.g.
            # embedded quotations in a script the original OCR never
            # recognized, garbled into Latin-alphabet-lookalike noise) get an
            # honest placeholder instead of being sent through the
            # translator -- an NMT model doesn't fail loudly on garbage input,
            # it produces fluent, confident, entirely fabricated English,
            # which reads as real content and is worse than admitting the
            # source is unusable. See ingest/garble_detect.py.
            clean = [s for s in batch if not is_garbled(s.latin_text)]
            garbled = [s for s in batch if is_garbled(s.latin_text)]

            # Some clean-enough segments still carry genuine embedded Greek
            # (recovered by ingest/page_splice.py's re-OCR pipeline -- see
            # ingest/mixed_lang_translate.py for why those go through a
            # separate, safer path rather than the plain translator).
            # In a papyrus, a mostly-Greek segment is translated whole as Greek:
            # the mixed path would strip it down to a name or two.
            greek = [s for s in clean if greek_tr and is_mostly_greek(s.latin_text)]
            mixed = [s for s in clean if d.language == "la" and s not in greek
                     and _GREEK_CHAR.search(s.latin_text)]
            # German editorial apparatus (see ingest/german_detect.py) goes
            # to stock NLLB as German; the Latin model mangles it.
            german = [s for s in clean if d.language == "la" and s not in mixed
                      and s not in greek and is_german(s.latin_text)]
            pure = [s for s in clean if s not in mixed and s not in german and s not in greek]

            # (segment, English, which translator wrote it): the third item is what
            # lets a later pass retranslate only what one model produced.
            results = []
            if pure:
                eng = tr.translate_batch([s.latin_text for s in pure],
                                         batch_size=args.batch_size)
                results.extend((s, e, model_label) for s, e in zip(pure, eng))
            if greek:
                eng = greek_tr.translate_batch([s.latin_text for s in greek],
                                               batch_size=args.batch_size)
                results.extend((s, e, f"{model_label}+grc") for s, e in zip(greek, eng))
            if mixed:
                eng = translate_mixed_batch([s.latin_text for s in mixed], tr,
                                            batch_size=args.batch_size)
                results.extend((s, e, f"{model_label}+mixed") for s, e in zip(mixed, eng))
            if german:
                gtr = lib.german_translator()
                if hasattr(gtr, "max_length"):
                    gtr.max_length = args.max_length
                eng = gtr.translate_batch([s.latin_text for s in german],
                                          batch_size=args.batch_size)
                results.extend((s, e, "nllb:stock:deu_Latn") for s, e in zip(german, eng))
            results.extend((s, UNTRANSLATABLE_PLACEHOLDER, "placeholder") for s in garbled)
            by_model = {}
            for s, e, m in results:
                by_model.setdefault(m, []).append((s, e))
            for m, items in by_model.items():
                flags = ({s.id: f for (s, _), f in zip(items, qc.flag_batch(
                    [s.latin_text for s, _ in items], [e for _, e in items], allowed_names))}
                         if m != "placeholder" else {s.id: "" for s, _ in items})
                lib.store.set_translations([(s.id, e) for s, e in items],
                                           reset_styled=args.redo, model=m, flags=flags)
                for f in flags.values():
                    flag_totals["checked"] += 1
                    if f:
                        flag_totals["flagged"] += 1
                        for one in f.split(","):
                            flag_totals[one] = flag_totals.get(one, 0) + 1
            done += len(batch)
            grand += len(batch)
            rate = grand / max(time.time() - t_start, 1e-6)
            eta = (total - grand) / rate / 3600
            print(f"  [{d.id}] {(d.source or '')[:22]:22} {done:>6,}/{n:<6,} "
                  f"| overall {grand:,}/{total:,} {rate:.1f} seg/s ETA {eta:.1f}h "
                  f"| flagged {flag_totals['flagged']}/{flag_totals['checked']}",
                  flush=True)
        print(f"  [{d.id}] DONE {n} segs in {time.time()-t0:.0f}s", flush=True)

    if args.sample:
        print("\nSample only: nothing was written. Re-run without --sample to translate.")
    else:
        detail = ", ".join(f"{k} {v}" for k, v in sorted(flag_totals.items())
                           if k not in ("checked", "flagged"))
        print(f"\nDone. Translated {grand:,} segments across {len(plan)} docs "
              f"in {(time.time()-t_start)/3600:.1f}h. Quality flags: "
              f"{flag_totals['flagged']}/{flag_totals['checked']} segments flagged"
              + (f" ({detail})" if detail else "") + ".")
    if server:
        server.__exit__(None, None, None)
    lib.close()


if __name__ == "__main__":
    main()
