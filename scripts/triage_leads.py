"""Have a local model read each quotation lead and tag it, so the Leads tab can rank them.

    python scripts/triage_leads.py                    # every lead not yet tagged
    python scripts/triage_leads.py --witness suda --force --model qwen3:14b

Normally run by the web app's job queue (kind "triage_leads") on a free GPU, with
a private Ollama like the summarize job. Reads and writes only data/leads.db
(DFHG files are read for each author's profile). See core/lead_triage.py for
the questions asked. Resumable: each tag is committed as it is made.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import lead_triage                                          # noqa: E402
from core.leads import LeadStore                                      # noqa: E402
from core.local_llm import OllamaClient, OllamaError, PrivateOllama   # noqa: E402
from ingest.quotations import RAW, WITNESSES, load_dfhg              # noqa: E402


# Fits one card with room to spare; tried in order when no model is named.
PREFERRED = ("gemma4:12b", "qwen3:14b", "gemma3:12b")


def _raise_interrupt(*_):
    raise KeyboardInterrupt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("LATIN_SUMMARY_MODEL", ""),
                    help=f"default: the first installed of {', '.join(PREFERRED)}")
    ap.add_argument("--witness", choices=sorted(WITNESSES), help="only leads from this text")
    ap.add_argument("--kind", choices=("citation", "reuse"), help="only this kind of lead")
    ap.add_argument("--force", action="store_true", help="re-tag leads already tagged")
    ap.add_argument("--limit", type=int, default=None, help="stop after N leads")
    ap.add_argument("--leads", default=None, help="lead database (default data/leads.db)")
    ap.add_argument("--ollama-url", default="", help="use this running Ollama instead of starting a private one")
    args = ap.parse_args()
    if os.name == "nt":
        signal.signal(signal.SIGBREAK, _raise_interrupt)   # queue cancel; see summarize.py

    store = LeadStore(args.leads) if args.leads else LeadStore()
    todo = store.to_tag(force=args.force, witness=args.witness, kind=args.kind, limit=args.limit)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "(not pinned)")
    print(f"=== Sort leads ({args.model or 'default model'}): {len(todo):,} leads === on CUDA device {gpu}", flush=True)
    if not todo:
        print("Nothing to sort: every lead is tagged (use --force to redo).")
        return
    fragments = load_dfhg(RAW / "dfhg")
    profiles = {}

    t0, done, failed, server = time.time(), 0, 0, None
    try:
        if args.ollama_url:
            client = OllamaClient(args.ollama_url)
        else:
            log = os.path.join(ROOT, "data", "joblogs", f"ollama-leads-{os.getpid()}.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            server = PrivateOllama(log_path=log).__enter__()
            client = server.client
        if not args.model:
            args.model = next((m for m in PREFERRED if client.has_model(m)), "")
            if not args.model:
                raise SystemExit(f"none of {', '.join(PREFERRED)} is pulled: name a model with --model")
            print(f"Using {args.model}", flush=True)
        if not client.has_model(args.model):
            names = ", ".join(m["name"] for m in client.models())
            raise SystemExit(f"model {args.model!r} is not pulled. Available: {names}")

        for row in todo:
            lead = dict(row, witness_label=WITNESSES.get(row["witness"], (row["witness"],))[0])
            if lead["kind"] == "citation" and lead["author"] not in profiles:
                profiles[lead["author"]] = lead_triage.author_profile(lead["author"], fragments)
            try:
                tag = lead_triage.tag_lead(client, args.model, lead, profiles.get(lead["author"], ""))
            except OllamaError as exc:
                failed += 1
                print(f"  [{row['id']}] FAILED: {exc}", flush=True)
                continue
            store.set_tag(row["id"], tag, args.model, lead_triage.rank(lead["kind"], tag))
            done += 1
            per_min = done / max(time.time() - t0, 1e-6) * 60
            what = (f"{tag['same_person']:6} {tag['reference']:10}" if lead["kind"] == "citation"
                    else f"{tag['relation']:12}      ")
            print(f"  [{row['id']}] {row['witness'][:12]:12} {(row['author'] or '')[:22]:22} {what} c{tag['confidence']} "
                  f"| overall {done + failed:,}/{len(todo):,} {per_min:.0f} leads/min", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted: tags so far are saved.", flush=True)
        raise SystemExit(130)
    finally:
        if server is not None:
            server.__exit__(None, None, None)
        store.close()
    print(f"\nDone. Tagged {done:,} leads ({failed} failed) in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
