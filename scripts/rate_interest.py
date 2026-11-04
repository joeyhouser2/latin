"""Rate summarized works for how interesting they look, so the Summaries tab can sort by it.

    python scripts/rate_interest.py                  # every work not yet rated
    python scripts/rate_interest.py --doc-id 67 --force
    python scripts/rate_interest.py --model qwen3:14b --force   # re-rate everything

Normally run by the web app's job queue (kind "rate"). Reads only the
catalogue summaries in data/summaries.db and writes ratings back there; corpus.db
is opened read-only for each work's language and century. One short prompt per
work (see core/interest.py for the rubric), so a full library takes minutes,
not hours. New summaries are rated by summarize.py as they are written; this is
for the backlog, and for re-rating with another model.

Resumable: each rating is committed as it is made, and works whose rating is of
their current summary are skipped unless --force.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import interest                                           # noqa: E402
from core.local_llm import OllamaClient, OllamaError, PrivateOllama   # noqa: E402
from core.summaries import SummaryStore                            # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _raise_interrupt(*_):
    raise KeyboardInterrupt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-id", type=int, action="append", default=[],
                    help="only this work (repeatable); default every unrated work")
    ap.add_argument("--model", default=os.environ.get("LATIN_SUMMARY_MODEL", "gemma4:12b"))
    ap.add_argument("--force", action="store_true", help="re-rate works already rated")
    ap.add_argument("--limit", type=int, default=None, help="stop after N works")
    ap.add_argument("--ollama-url", default="",
                    help="use this running Ollama instead of starting a private one")
    ap.add_argument("--corpus", default=os.path.join(REPO_ROOT, "data", "corpus.db"))
    ap.add_argument("--summaries", default=os.path.join(REPO_ROOT, "data", "summaries.db"))
    args = ap.parse_args()
    if os.name == "nt":
        signal.signal(signal.SIGBREAK, _raise_interrupt)   # queue cancel; see summarize.py

    store = SummaryStore(args.summaries)
    todo = store.documents_to_rate(force=args.force, doc_ids=args.doc_id)[:args.limit]
    corpus = sqlite3.connect(f"file:{args.corpus}?mode=ro", uri=True, timeout=30)
    meta = {r[0]: tuple(r[1:]) for r in corpus.execute(
        "SELECT id, language, century, translation_status FROM documents")}
    corpus.close()

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "(not pinned)")
    print(f"=== Rate interest ({args.model}): {len(todo):,} works === on CUDA device {gpu}\n",
          flush=True)
    if not todo:
        return

    t0, done, server = time.time(), 0, None
    try:
        if args.ollama_url:
            client = OllamaClient(args.ollama_url)
        else:
            log = os.path.join(REPO_ROOT, "data", "joblogs", f"ollama-rate-{os.getpid()}.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            server = PrivateOllama(log_path=log).__enter__()
            client = server.client
        if not client.has_model(args.model):
            names = ", ".join(m["name"] for m in client.models())
            raise SystemExit(f"model {args.model!r} is not pulled. Available: {names}")

        for row in todo:
            work = interest.describe(row["title"], row["author"],
                                     *meta.get(row["doc_id"], (None, None, None)))
            try:
                r = interest.rate(client, args.model, work, row["summary"],
                                  json.loads(row["topics"]))
            except OllamaError as exc:
                print(f"  [{row['doc_id']}] FAILED: {exc}", flush=True)
                continue
            store.set_interest(row["doc_id"], row["id"], r, args.model)
            done += 1
            per_min = done / max(time.time() - t0, 1e-6) * 60
            eta = (len(todo) - done) / per_min / 60
            title = " ".join((row["title"] or "").split())
            print(f"  [{row['doc_id']}] {title[:28]:28} {r['score']:4.1f} "
                  f"({r['subject']}{r['story']}{r['history']}{r['neglect']}) "
                  f"| overall {done:,}/{len(todo):,} {per_min:.0f} works/min ETA {eta:.1f}h",
                  flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted -- ratings so far are saved.", flush=True)
        raise SystemExit(130)
    finally:
        if server is not None:
            server.__exit__(None, None, None)
        store.close()

    print(f"\nDone. Rated {done:,} works in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
