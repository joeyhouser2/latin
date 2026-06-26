"""Harvest a curated batch of *untranslated* late-antique / early-medieval texts
(roughly 200s-1000 CE) in Latin and Greek, ingest them, and machine-translate
every untranslated segment with the trained era-specific models.

Sources:
  - DigilibLT     -> late-antique Latin (TEI). All ~untranslated by nature.
  - First1KGreek  -> patristic / late-antique Greek. Only works WITHOUT an
                     ``-eng`` edition are picked here (genuinely untranslated).

Each entry is (source, identifier, century, author, short_title). Author/title
are hints for the run log; the connector's own metadata wins on ingest, except
that we stamp ``century`` so the reader can place the work in time.

Usage:
    python scripts/harvest_late_antique.py                 # ingest + translate all
    python scripts/harvest_late_antique.py --no-translate  # ingest only
    python scripts/harvest_late_antique.py --max-chars 600000
    python scripts/harvest_late_antique.py --cap-segments 1500   # per-doc safety cap

Set CUDA_VISIBLE_DEVICES=0 to use the 4070 SUPER (the faster card) for translation.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

# On Windows, faiss and torch each bundle an OpenMP runtime; if faiss's loads
# first (it's imported when pipeline builds the FAISS index) it clashes with
# torch's and segfaults at the first sentence-transformers import. Warming
# sentence_transformers here — before pipeline pulls in faiss — fixes the order.
import sentence_transformers  # noqa: F401  (import for side effect: load order)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import Library
from ingest.registry import get_connector
from ingest.base import Connector


# (source, identifier, century_CE, author, short_title)
CURATED = [
    # --- Latin: DigilibLT, late-antique (stage late_antique) -----------------
    ("digiliblt", "DLT000592", 3, "Anon.",            "Passio Perpetuae et Felicitatis"),
    ("digiliblt", "DLT000180", 4, "Anon.",            "Expositio totius mundi et gentium"),
    ("digiliblt", "DLT000124", 4, "Anon.",            "De rebus bellicis"),
    ("digiliblt", "DLT000123", 4, "Anon.",            "De Physiognomonia liber"),
    ("digiliblt", "DLT000144", 4, "Dictys Cretensis", "Ephemeris belli Troiani"),
    ("digiliblt", "DLT000030", 4, "Anon.",            "Epitome de Caesaribus"),
    ("digiliblt", "DLT000253", 5, "Anon.",            "Historia Apollonii regis Tyri (rec. A)"),
    ("digiliblt", "DLT000115", 5, "Dares Phrygius",   "De excidio Troiae historia"),
    ("digiliblt", "DLT000522", 6, "Iordanes",         "De origine actibusque Getarum (Getica)"),
    ("digiliblt", "DLT000010", 6, "Anon. Valesianus", "Excerpta Valesiana pars posterior"),
    ("digiliblt", "DLT000571", 6, "Antoninus Plac.",  "Itinerarium (rec. prior)"),
    # --- Greek: First1KGreek, patristic, untranslated (stage late_antique) ----
    ("first1k_greek", "tlg2018.tlg003", 4, "Eusebius", "Contra Hieroclem"),
    ("first1k_greek", "tlg2018.tlg011", 4, "Eusebius", "Onomasticon"),
    ("first1k_greek", "tlg2018.tlg020", 4, "Eusebius", "Vita Constantini"),
    ("first1k_greek", "tlg2018.tlg021", 4, "Eusebius", "Constantini oratio ad coetum sanctorum"),
    ("first1k_greek", "tlg2018.tlg022", 4, "Eusebius", "De laudibus Constantini"),
    ("first1k_greek", "tlg2042.tlg007", 3, "Origenes", "Exhortatio ad martyrium"),
    ("first1k_greek", "tlg2042.tlg008", 3, "Origenes", "Origen (tlg008)"),
    ("first1k_greek", "tlg2042.tlg045", 3, "Origenes", "Epistula ad Africanum"),
    # --- Latin: openMGH, late-antique / early-medieval, under-translated -------
    # MGH is a critical-edition series, so the connector tags translation status
    # per work (defaults to "unknown"); these are picked for the 400s-800s window
    # and for being little-/un-translated. Several are large — raise --max-chars
    # (e.g. 1_500_000) to keep the bigger volumes instead of skipping them.
    ("mgh", "bsb00000826", 5, "Dracontius et al.", "Carmina (Merobaudes, Dracontius, Eugenius Tolet.)"),
    ("mgh", "bsb00000795", 6, "Avitus Viennensis", "Opera quae supersunt"),
    ("mgh", "bsb00000796", 6, "Ennodius",          "Opera"),
    ("mgh", "bsb00000824", 6, "Cassiodorus",       "Variae"),
    ("mgh", "bsb00000827", 7, "Aldhelmus",         "Opera"),
    ("mgh", "bsb00000749", 7, "Anon. / Fredegar",  "Fredegarii Chronica; Vitae sanctorum"),
    ("mgh", "bsb00000750", 7, "Anon.",             "Passiones vitaeque sanctorum aevi Merovingici (I)"),
    # --- Greek: Patristic Text Archive (pta), late-antique (stage late_antique)
    # CapiTainS TEI; most works have a German but NO English translation -> the
    # English-gap material this library targets (translation_status "unknown").
    ("pta", "pta0013.pta003", 4, "Amphilochius of Iconium", "Epistula synodalis"),
    ("pta", "pta0001.pta001", 5, "Severian of Gabala",      "De fide et lege naturae"),
    ("pta", "pta0001.pta002", 5, "Severian of Gabala",      "De paenitentia et compunctione"),
    ("pta", "pta0003.pta007", 4, "Eusebius of Caesarea",    "Contra Marcellum"),
    ("pta", "pta0003.pta009", 4, "Eusebius of Caesarea",    "De ecclesiastica theologia"),
    ("pta", "pta0004.pta003", 5, "Theodoret of Cyrrhus",    "Historia ecclesiastica"),
    ("pta", "pta0006.pta001", 5, "Hesychius of Jerusalem",  "Commentarius magnus in Psalmos"),
    # --- Greek: Patrologia Graeca Corpus (pg_corpus), Byzantine OCR ------------
    # Whole PG volumes — LARGE; raise --max-chars (e.g. 3_000_000) to keep them.
    ("pg_corpus", "PG003", 6, "Ps.-Dionysius Areopagita", "Corpus Areopagiticum (PG vol. 3)"),
    ("pg_corpus", "PG101", 9, "Photius",                   "Opera (PG vol. 101)"),
]


def expected_source(src: str, ident: str) -> str | None:
    """Reconstruct the ``source`` string a connector will stamp, so we can skip
    works already in the library without re-fetching them."""
    if src == "digiliblt":
        return f"DigilibLT ({ident})"
    if src in ("first1k_greek", "perseus", "perseus_greek", "pta"):
        label = "PTA" if src == "pta" else "Perseus"
        gw = ident.split(":")[-1].replace("/", ".").split(".")
        return f"{label} ({gw[0]}.{gw[1]})"
    if src == "mgh":
        m = re.search(r"(bsb\d+)", ident, re.IGNORECASE)
        return f"MGH ({m.group(1).lower()})" if m else None
    if src == "pg_corpus":
        m = re.search(r"PG(\d+(?:_\d+)?)", ident, re.IGNORECASE)
        if not m:
            return None
        head, _, tail = m.group(1).partition("_")
        vol = f"PG{int(head):03d}" + (f"_{tail}" if tail else "")
        return f"PG Corpus ({vol})"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-translate", action="store_true",
                    help="ingest only; skip the machine-translation pass")
    ap.add_argument("--max-chars", type=int, default=700_000,
                    help="skip a work whose raw text exceeds this (avoids legal codes etc.)")
    ap.add_argument("--cap-segments", type=int, default=0,
                    help="per-doc safety cap on segments translated (0 = no cap)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--source", default="",
                    help="comma-separated connector names to limit the run "
                         "(e.g. 'pta,pg_corpus' to pull only the Byzantine Greek)")
    args = ap.parse_args()

    src_filter = {s.strip() for s in args.source.split(",") if s.strip()}
    curated = [c for c in CURATED if not src_filter or c[0] in src_filter]

    lib = Library()
    connectors: dict = {}
    ingested = []   # (doc_id, label, n_seg)
    skipped = []
    existing = {d.source for d in lib.store.list_documents() if d.source}

    scope = f" [{', '.join(sorted(src_filter))}]" if src_filter else ""
    print(f"=== Harvest: {len(curated)} curated works{scope} ===\n")
    for src, ident, century, author, short in curated:
        label = f"{author} — {short}"
        if expected_source(src, ident) in existing:
            print(f"HAVE  {label}  (already ingested)")
            skipped.append((label, "already ingested"))
            continue
        conn = connectors.get(src) or connectors.setdefault(src, get_connector(src))
        try:
            raw_meta, parts = conn.fetch(ident, century=century)
            nchars = sum(len(t) for _, t in parts)
            if nchars > args.max_chars:
                print(f"SKIP  {label}  ({nchars:,} chars > max)")
                skipped.append((label, f"{nchars:,} chars"))
                continue
            doc = Connector.build_document(raw_meta, parts)
            lib.ingest(doc)
            n_seg = len(list(doc.iter_segments()))
            ingested.append((doc.id, doc.language, doc.language_stage, label, n_seg))
            print(f"INGEST [{doc.id}] {doc.language}/{doc.language_stage}  {label}  "
                  f"({n_seg} segs, {nchars:,} chars)")
        except Exception as exc:
            print(f"FAIL  {label}: {exc}")
            skipped.append((label, str(exc)[:80]))

    print(f"\nIngested {len(ingested)} works, skipped {len(skipped)}.")

    if args.no_translate:
        lib.close()
        return

    # Translate every curated work present in the library (whether ingested this
    # run or already there), not just the ones ingested just now.
    by_source = {d.source: d for d in lib.store.list_documents()}
    to_translate = []   # (doc_id, lang, stage, label, n_seg)
    seen_ids = set()
    for src, ident, century, author, short in curated:
        d = by_source.get(expected_source(src, ident))
        if d is None or d.id in seen_ids:
            continue
        seen_ids.add(d.id)
        full = lib.store.get_document(d.id)
        n_seg = len(list(full.iter_segments()))
        to_translate.append((d.id, d.language, d.language_stage,
                             f"{author} — {short}", n_seg))

    print(f"\n=== Translating {len(to_translate)} works ===")
    grand = 0
    for doc_id, lang, stage, label, n_seg in to_translate:
        cap = args.cap_segments
        t0 = time.time()
        if cap and n_seg > cap:
            # translate only the first `cap` segments (safety bound for huge docs)
            doc = lib.store.get_document(doc_id)
            pending = [s for s in doc.iter_segments() if not s.is_translated][:cap]
            tr = lib.translator_for(lang, stage)
            eng = tr.translate_batch([s.latin_text for s in pending], batch_size=args.batch_size)
            lib.store.set_translations([(s.id, e) for s, e in zip(pending, eng)])
            n = len(pending)
            note = f" (capped at {cap}/{n_seg})"
        else:
            n = lib.translate_document(doc_id, batch_size=args.batch_size)
            note = ""
        grand += n
        dt = time.time() - t0
        print(f"  [{doc_id}] {label}: {n} segments in {dt:.0f}s{note}")

    print(f"\nDone. Translated {grand} segments across {len(to_translate)} works.")
    lib.close()


if __name__ == "__main__":
    main()
