"""Summarize translated documents with a local LLM, so they can be searched.

    python scripts/summarize.py --doc-id 386
    python scripts/summarize.py --doc-id 385 --doc-id 387 --model qwen3:14b
    python scripts/summarize.py --all --min-translated 0.9 --limit 50

Normally run by the web app's job queue (kind "summarize"), which pins it to
whichever GPU the translation job is not using. It starts a private Ollama
server on that card (see core/local_llm.py), writes to data/summaries.db, and
opens corpus.db **read-only** -- so it can run beside a translation without
becoming a second writer on the corpus.

How a document is summarized
----------------------------
The text is cut into chunks of ~``--chunk-chars`` characters, each chunk
summarized on its own ("part" summaries, stored with the segment offsets they
cover so search can open the reader at the right place), and then the part
summaries are summarized into one paragraph for the whole work -- in rounds, if
there are too many parts to fit one prompt. A document that fits in one chunk
gets its document summary directly.

By default the model sees **both** the original and the machine translation,
line by line (``--source both``). The MT is weak on this material, and a 12B
model reads Latin well enough to catch where "The day of Melody" was really
German apparatus, or where a line was left in Latin. ``--source english`` halves
the input for speed, at the price of summarizing the MT's mistakes.

``--source original`` is the *preview* mode: it needs no translation. The model
reads only the Latin/Greek (a sample of ``--sample-parts`` evenly spaced chunks
for long works), is told it is looking at an untranslated OCR text, and writes
what the work is *or may be*, flagging uncertainty. Use it to decide whether a
work is worth translating. The preview is replaced automatically by a real
summary once the document has translations.

Resumable: parts are committed as they finish and reused on a re-run (matched
by exact span, model and translation state), and documents whose summary
already reflects their current translations are skipped unless ``--force``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.local_llm import OllamaClient, OllamaError, PrivateOllama   # noqa: E402
from core.summaries import SummaryStore                            # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACEHOLDER_PREFIX = "[untranslatable"
LANG_NAMES = {"la": "Latin", "grc": "Greek"}

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "topics"],
}

SYSTEM = (
    "You are a scholar of Latin and Greek writing entries for a searchable "
    "catalogue of mostly untranslated texts. You are precise and concrete: you "
    "name the people, places, feasts, arguments and subjects actually present, "
    "and you never invent anything the text does not contain. When the text is "
    "too corrupt to tell what it says, you say so plainly."
)

_HAS_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


@dataclass
class Chunk:
    text: str
    offset_first: int
    offset_last: int
    section_first: int
    section_last: int


def _raise_interrupt(*_):
    raise KeyboardInterrupt


def load_document(conn: sqlite3.Connection, doc_id: int) -> Optional[Dict[str, Any]]:
    doc = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if doc is None:
        return None
    rows = conn.execute(
        """SELECT sec.ord AS sord, seg.latin_text AS la, seg.english_text AS en
           FROM segments seg JOIN sections sec ON seg.section_id = sec.id
           WHERE sec.doc_id = ? ORDER BY sec.ord, seg.ord""", (doc_id,)).fetchall()
    section_no: Dict[int, int] = {}
    for r in rows:
        section_no.setdefault(r["sord"], len(section_no) + 1)
    translated = sum(1 for r in rows if r["en"] and r["en"].strip())
    return {"doc": doc, "rows": rows, "section_no": section_no,
            "translated": translated, "total": len(rows)}


def make_chunks(loaded: Dict[str, Any], mode: str, budget: int) -> List[Chunk]:
    """Cut a document into prompt-sized chunks of line pairs, in reading order.

    Lines with no word in them (verse and page numbers, sigla, stray
    punctuation) are dropped: in the Analecta volumes they are about a fifth of
    all segments, and would otherwise spend a fifth of every prompt on noise.
    """
    chunks: List[Chunk] = []
    buf: List[str] = []
    size = 0
    first = sec_first = None
    last = sec_last = 0
    for offset, r in enumerate(loaded["rows"]):
        la = (r["la"] or "").strip()
        en = (r["en"] or "").strip()
        if not _HAS_WORD.search(la):
            continue
        has_en = bool(en) and en != la and not en.startswith(PLACEHOLDER_PREFIX)
        if mode == "original":
            line = la
        elif mode == "english":
            if not has_en:
                continue
            line = en
        else:
            line = f"LA: {la}" + (f"\nEN: {en}" if has_en else "")
        line = line[:budget]
        if buf and size + len(line) > budget:
            chunks.append(Chunk("\n".join(buf), first, last, sec_first, sec_last))
            buf, size, first = [], 0, None
        if first is None:
            first, sec_first = offset, loaded["section_no"][r["sord"]]
        buf.append(line)
        size += len(line) + 1
        last, sec_last = offset, loaded["section_no"][r["sord"]]
    if buf:
        chunks.append(Chunk("\n".join(buf), first, last, sec_first, sec_last))
    return chunks


def describe_work(doc: sqlite3.Row) -> str:
    bits = [doc["title"]]
    if doc["author"]:
        bits.append(f"by {doc['author']}")
    lang = LANG_NAMES.get(doc["language"], doc["language"])
    stage = (doc["language_stage"] or "").replace("_", " ")
    extra = ", ".join(b for b in (lang, stage if stage != "unknown" else "",
                                  f"{doc['century']}th c." if doc["century"] else "") if b)
    return f"{' '.join(bits)} ({extra})"


def part_prompt(doc: sqlite3.Row, chunk: Chunk, i: int, n: int, mode: str) -> str:
    lang = LANG_NAMES.get(doc["language"], "Latin")
    if mode == "original":
        how = f"The text is the original {lang}, not yet translated; read it directly."
    elif mode == "both":
        how = (f"Each LA line is the original {lang}; the EN line under it, when present, is "
               f"a rough machine translation that is often wrong -- trust the {lang} where they "
               f"disagree.")
    else:
        how = "The text is a rough machine translation into English and is often wrong."
    return (
        f"Work: {describe_work(doc)}\n"
        f"Part {i} of {n}, sections {chunk.section_first}-{chunk.section_last}.\n\n"
        f"{how} The source is OCR of a printed edition and may include page numbers, "
        f"running heads, manuscript sigla and editors' notes in German or French: "
        f"describe the actual text, and mention editorial apparatus only if it is most "
        f"of the passage.\n\n"
        f"----- TEXT -----\n{chunk.text}\n----- END -----\n\n"
        f"Return JSON with:\n"
        f'- "summary": 2-4 sentences on what this part actually contains: subjects, '
        f"occasions, arguments, named people and places. Specific, no filler such as "
        f'"This passage discusses".\n'
        f'- "topics": 3-8 short English subject tags, including proper names.'
    )


def document_prompt(doc: sqlite3.Row, body: str, from_parts: bool,
                    mode: str = "both", sampled: bool = False) -> str:
    if mode == "original":
        scope = ("Below are summaries of sampled passages from across the work, in order."
                 if sampled and from_parts else
                 "Below are summaries of its consecutive parts, in order."
                 if from_parts else "Below is the complete original text.")
        sample_note = "; only a sample of the work was read" if sampled else ""
        return (
            f"Work: {describe_work(doc)}\n{scope}\n\n"
            f"This is a PREVIEW written before any translation exists, to help decide "
            f"whether the work is worth translating. The text is OCR and may be noisy"
            f"{sample_note}. Use the title and author as clues, but base claims on the "
            f"text. Say what the work is or most likely is, and say plainly where you "
            f"are guessing or where the OCR is too poor to tell.\n\n"
            f"----- INPUT -----\n{body}\n----- END -----\n\n"
            f"Return JSON with:\n"
            f'- "summary": 4-7 sentences: genre, subject, structure, period/context, and '
            f"notable people and places; hedge ('appears to', 'possibly') where unsure.\n"
            f'- "topics": 5-12 English subject tags, including proper names.'
        )
    intro = ("Below are summaries of its consecutive parts, in order."
             if from_parts else
             "Below is the complete text: original lines (LA) with a rough, often wrong "
             "machine translation (EN) where one exists.")
    return (
        f"Work: {describe_work(doc)}\n{intro}\n\n"
        f"----- INPUT -----\n{body}\n----- END -----\n\n"
        f"Return JSON with:\n"
        f'- "summary": one catalogue paragraph of 4-7 sentences: what kind of work this '
        f"is, what it contains and how it is organised, and its notable people, places "
        f"and themes. Specific; do not pad.\n"
        f'- "topics": 5-12 English subject tags, including proper names.'
    )


def clean(out: Dict[str, Any], max_topics: int) -> Dict[str, Any]:
    summary = re.sub(r"\s+", " ", str(out.get("summary", ""))).strip()
    topics, seen = [], set()
    for t in out.get("topics") or []:
        # Models like snake_case tags ("Vesper_service"); tags are for people.
        t = re.sub(r"[\s_]+", " ", str(t)).strip(" .;,")
        if t and t.lower() not in seen:
            seen.add(t.lower())
            topics.append(t)
    return {"summary": summary, "topics": topics[:max_topics]}


class Progress:
    def __init__(self, total: int):
        self.total, self.done, self.worked, self.t0 = total, 0, 0, time.time()

    def tick(self, doc_id: int, title: str, what: str, reused: bool = False) -> None:
        self.done += 1
        # Rate counts only chunks the model actually wrote: parts reused from an
        # interrupted run cost nothing, and counting them made a resumed job
        # report 100,000 chunks/min and an ETA of zero.
        if not reused:
            self.worked += 1
        mins = max(time.time() - self.t0, 1e-6) / 60
        rate = self.worked / mins
        eta = (self.total - self.done) / rate / 60 if rate else 0
        title = " ".join(title.split())      # some catalogue titles carry line breaks
        print(f"  [{doc_id}] {title[:28]:28} {what:14} | overall {self.done:,}/{self.total:,} "
              f"{rate:.1f} chunks/min ETA {eta:.1f}h", flush=True)


def summarize_document(client: OllamaClient, store: SummaryStore, loaded: Dict[str, Any],
                       args, progress: Progress, embed_ok: bool) -> None:
    doc = loaded["doc"]
    doc_id, title = doc["id"], doc["title"]
    chunks = make_chunks(loaded, args.source, args.chunk_chars)
    sampled = False
    if args.source == "original" and len(chunks) > args.sample_parts:
        # Evenly spaced chunks, always including the first (title/preface) and last.
        n = args.sample_parts
        idx = sorted({round(k * (len(chunks) - 1) / (n - 1)) for k in range(n)}) if n > 1 else [0]
        chunks = [chunks[k] for k in idx]
        sampled = True
    common = dict(model=args.model, source_mode=args.source,
                  translated_count=loaded["translated"], title=title, author=doc["author"])
    llm = dict(num_ctx=args.num_ctx)

    if len(chunks) == 1:
        out = clean(client.chat_json(args.model, SYSTEM,
                                     document_prompt(doc, chunks[0].text, from_parts=False, mode=args.source),
                                     SCHEMA, **llm), 12)
        progress.tick(doc_id, title, "document")
    else:
        parts: List[Dict[str, Any]] = []
        for i, ch in enumerate(chunks):
            hit = store.find_part(doc_id, ch.offset_first, ch.offset_last, args.model,
                                  args.source, loaded["translated"])
            if hit is not None:
                parts.append({"summary": hit["summary"], "topics": json.loads(hit["topics"]),
                              "chunk": ch})
                progress.tick(doc_id, title, f"part {i + 1}/{len(chunks)} (kept)", reused=True)
                continue
            out = clean(client.chat_json(args.model, SYSTEM,
                                         part_prompt(doc, ch, i + 1, len(chunks), args.source),
                                         SCHEMA, **llm), 8)
            store.add(doc_id=doc_id, level="part", summary=out["summary"],
                      topics=out["topics"], part_index=i, part_count=len(chunks),
                      seg_offset_first=ch.offset_first, seg_offset_last=ch.offset_last,
                      section_first=ch.section_first, section_last=ch.section_last, **common)
            parts.append({**out, "chunk": ch})
            progress.tick(doc_id, title, f"part {i + 1}/{len(chunks)}")

        # Reduce, in rounds if the part summaries do not fit one prompt.
        items = [f"[Part {k + 1}, sections {p['chunk'].section_first}-{p['chunk'].section_last}] "
                 f"{p['summary']} (topics: {', '.join(p['topics'])})"
                 for k, p in enumerate(parts)]
        while True:
            batches, cur, size = [], [], 0
            for it in items:
                if cur and size + len(it) > args.chunk_chars:
                    batches.append(cur)
                    cur, size = [], 0
                cur.append(it)
                size += len(it) + 1
            batches.append(cur)
            if len(batches) == 1:
                break
            items = []
            for b in batches:
                o = clean(client.chat_json(args.model, SYSTEM,
                                           document_prompt(doc, "\n".join(b), from_parts=True, mode=args.source, sampled=sampled),
                                           SCHEMA, **llm), 12)
                items.append(f"{o['summary']} (topics: {', '.join(o['topics'])})")
        out = clean(client.chat_json(args.model, SYSTEM,
                                     document_prompt(doc, "\n".join(items), from_parts=True, mode=args.source, sampled=sampled),
                                     SCHEMA, **llm), 12)
        progress.tick(doc_id, title, "document")

    store.add(doc_id=doc_id, level="document", summary=out["summary"], topics=out["topics"],
              part_count=len(chunks), seg_offset_first=0,
              seg_offset_last=max(0, loaded["total"] - 1), **common)
    store.prune_stale(doc_id, loaded["translated"], args.model, args.source)

    if embed_ok:
        rows = store.conn.execute(
            "SELECT id, summary, topics FROM summaries WHERE doc_id=? AND embedding IS NULL",
            (doc_id,)).fetchall()
        texts = [f"search_document: {title}. {r['summary']} Topics: "
                 f"{', '.join(json.loads(r['topics']))}" for r in rows]
        for i in range(0, len(texts), 64):
            vecs = client.embed(args.embed_model, texts[i:i + 64])
            store.set_embeddings(args.embed_model,
                                 [(r["id"], v) for r, v in zip(rows[i:i + 64], vecs)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-id", type=int, action="append", default=[],
                    help="document to summarize (repeatable)")
    ap.add_argument("--all", action="store_true",
                    help="every document translated past --min-translated")
    ap.add_argument("--min-translated", type=float, default=0.5,
                    help="with --all: minimum fraction of segments translated (default 0.5)")
    ap.add_argument("--min-segments", type=int, default=3,
                    help="with --all: skip documents shorter than this (single epigrams "
                         "gain nothing from a summary; default 3)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N documents")
    ap.add_argument("--model", default=os.environ.get("LATIN_SUMMARY_MODEL", "gemma4:12b"))
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--source", choices=["both", "english", "original"], default="both")
    ap.add_argument("--sample-parts", type=int, default=6,
                    help="with --source original: read at most this many evenly spaced "
                         "chunks of a long work (default 6)")
    ap.add_argument("--chunk-chars", type=int, default=11000,
                    help="characters of text per part prompt (~3.5k tokens)")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--force", action="store_true",
                    help="re-summarize even documents whose summary is current")
    ap.add_argument("--ollama-url", default="",
                    help="use this running Ollama instead of starting a private one "
                         "(then GPU pinning is up to that server)")
    ap.add_argument("--corpus", default=os.path.join(REPO_ROOT, "data", "corpus.db"))
    ap.add_argument("--summaries", default=os.path.join(REPO_ROOT, "data", "summaries.db"))
    args = ap.parse_args()

    if not args.doc_id and not args.all:
        ap.error("give --doc-id (repeatable) or --all")
    if os.name == "nt":
        # The queue cancels with CTRL_BREAK. Python's default for SIGBREAK is to
        # die on the spot without running `finally` -- turn it into a normal
        # interrupt so the private Ollama is shut down properly on cancel.
        signal.signal(signal.SIGBREAK, _raise_interrupt)

    # Read-only by construction: this job must never write corpus.db, because
    # that is what allows it to run alongside a translation.
    corpus = sqlite3.connect(f"file:{args.corpus}?mode=ro", uri=True, timeout=30)
    corpus.row_factory = sqlite3.Row
    store = SummaryStore(args.summaries)
    current = store.summarized_doc_ids()

    if args.doc_id:
        candidates = list(dict.fromkeys(args.doc_id))
    else:
        candidates = [r[0] for r in corpus.execute(
            """SELECT sec.doc_id FROM sections sec JOIN segments seg ON seg.section_id = sec.id
               GROUP BY sec.doc_id
               HAVING COUNT(*) >= ? AND
                      1.0 * SUM(seg.english_text IS NOT NULL AND seg.english_text <> '')
                      / COUNT(*) >= ?
               ORDER BY sec.doc_id""", (args.min_segments, args.min_translated))]

    plan = []
    for doc_id in candidates:
        loaded = load_document(corpus, doc_id)
        if loaded is None:
            print(f"  [{doc_id}] no such document -- skipped")
            continue
        if loaded["translated"] == 0 and args.source != "original":
            print(f"  [{doc_id}] nothing translated yet -- skipped (translate it first)")
            continue
        if not args.force and current.get(doc_id) == loaded["translated"]:
            continue    # summary already reflects the current translation
        n = len(make_chunks(loaded, args.source, args.chunk_chars))
        if args.source == "original":
            n = min(n, args.sample_parts)
        if n == 0:
            print(f"  [{doc_id}] no summarizable text -- skipped")
            continue
        plan.append((doc_id, n))
        if args.limit and len(plan) >= args.limit:
            break
    total = sum(n + (1 if n > 1 else 0) for _, n in plan)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "(not pinned)")
    print(f"=== Summarize ({args.model}, source={args.source}): {len(plan)} docs, "
          f"{total:,} chunks === on CUDA device {gpu}\n", flush=True)
    if not plan:
        return

    t0 = time.time()
    server = None
    try:
        if args.ollama_url:
            client = OllamaClient(args.ollama_url)
        else:
            log = os.path.join(REPO_ROOT, "data", "joblogs",
                               f"ollama-summarize-{os.getpid()}.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            server = PrivateOllama(log_path=log).__enter__()
            client = server.client
        if not client.has_model(args.model):
            names = ", ".join(m["name"] for m in client.models())
            raise SystemExit(f"model {args.model!r} is not pulled. Available: {names}")
        embed_ok = client.has_model(args.embed_model)
        if not embed_ok:
            print(f"  (embedding model {args.embed_model!r} not found -- summaries will "
                  f"be keyword-searchable only)", flush=True)

        progress = Progress(total)
        first = True
        for doc_id, _ in plan:
            loaded = load_document(corpus, doc_id)
            try:
                summarize_document(client, store, loaded, args, progress, embed_ok)
            except OllamaError as exc:
                print(f"  [{doc_id}] FAILED: {exc}", flush=True)
            if first and server is not None:
                # Proof of placement, into the job log: which card Ollama loaded onto.
                for line in server.gpu_lines():
                    if "inference compute" in line and "CUDA" in line:
                        print("  ollama:", line[line.find("description"):][:70], flush=True)
                first = False
    except KeyboardInterrupt:
        print("\nInterrupted -- finished parts are saved and will be reused.", flush=True)
        raise SystemExit(130)
    finally:
        if server is not None:
            server.__exit__(None, None, None)
        corpus.close()
        store.close()

    print(f"\nDone. Summarized {len(plan)} docs in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
