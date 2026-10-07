"""FastAPI backend for the library: browse, read, queue work.

Run it with ``python scripts/serve.py`` (or the desktop launcher) and open
http://127.0.0.1:8000.

Every handler here is a plain ``def``, not ``async def``, on purpose: they do
blocking SQLite reads and, in the case of ``/api/discover``, blocking HTTP calls
to a remote catalogue. FastAPI runs sync handlers in a threadpool, so one slow
Gallica search does not stall the event loop and freeze the rest of the page.

The heavy objects -- FAISS index, embedder, translator -- are never loaded at
import time. The index alone is ~600MB and the models are GPU-resident; a page
that only lists documents should start in under a second. Semantic search
therefore loads its machinery lazily, in the background, on the first query.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.models import LANGUAGE_STAGES, LANGUAGES          # noqa: E402
from core.stylizer import PRESETS                           # noqa: E402
from ingest.registry import available_sources, get_connector  # noqa: E402
from web import files as files_mod                          # noqa: E402
from web.jobs import JOB_KINDS, JobQueue                     # noqa: E402
from web.library_view import LibraryView                     # noqa: E402
from core.summaries import HL_CLOSE, HL_OPEN, SummaryStore   # noqa: E402
from core.local_llm import OllamaClient                      # noqa: E402


view = LibraryView()
queue = JobQueue()
summaries = SummaryStore()
# The shared Ollama service, used only to embed search queries (tiny, fast).
# Summaries themselves are written by jobs on a private, GPU-pinned server.
llm = OllamaClient(timeout=30)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    queue.start_worker()
    yield
    # Deliberately does not kill a job in flight: it is a separate process that
    # may be hours into a run. It will not survive the console window closing
    # (it shares that console), but it does survive this app's own shutdown, and
    # anything that does die is resumable -- its row comes back as "interrupted"
    # with a requeue button.
    queue.shutdown(kill_running=False)
    view.close()
    summaries.close()


app = FastAPI(title="Latin Library", docs_url="/api/docs", redoc_url=None,
              lifespan=lifespan)

# Lazily-built semantic search. _search_state is one of: "cold", "loading",
# "ready", "failed".
_search_lib = None
_search_state = "cold"
_search_error = ""
_search_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Library: documents and reading
# ---------------------------------------------------------------------------

@app.get("/api/facets")
def api_facets() -> Dict[str, Any]:
    data = view.facets()
    data["stages"] = [s for s in data["stages"]]
    data["all_stages"] = list(LANGUAGE_STAGES)
    data["language_names"] = LANGUAGES
    data["sources_available"] = available_sources()
    data["presets"] = sorted(PRESETS)
    data["job_kinds"] = {k: v.description for k, v in JOB_KINDS.items()}
    data["summaries"] = summaries.stats()
    return data


@app.get("/api/documents")
def api_documents(q: str = "", language: str = "", stage: str = "",
                  source_prefix: str = "", status: str = "",
                  translation_status: str = "", sort: str = "author",
                  offset: int = 0, limit: int = Query(50, le=500)) -> Dict[str, Any]:
    return view.documents(q=q, language=language, stage=stage,
                          source_prefix=source_prefix, status=status,
                          translation_status=translation_status, sort=sort,
                          offset=offset, limit=limit)


@app.get("/api/documents/{doc_id}")
def api_document(doc_id: int) -> Dict[str, Any]:
    doc = view.document(doc_id)
    if doc is None:
        raise HTTPException(404, "no such document")
    return doc


@app.get("/api/documents/{doc_id}/segments")
def api_segments(doc_id: int, offset: int = 0, limit: int = Query(300, le=2000),
                 section_id: Optional[int] = None) -> Dict[str, Any]:
    return view.segments(doc_id, offset=offset, limit=limit, section_id=section_id)


@app.get("/api/documents/{doc_id}/export")
def api_export(doc_id: int, column: str = "both") -> Any:
    """The whole document as plain text, for reading outside the browser.

    ``column`` is latin | english | styled | both.
    """
    doc = view.document(doc_id)
    if doc is None:
        raise HTTPException(404, "no such document")
    out: List[str] = [f"{doc['title']}", f"{doc['author'] or 'Anon.'}", ""]
    offset, page = 0, 1000
    while True:
        chunk = view.segments(doc_id, offset=offset, limit=page)
        for s in chunk["items"]:
            en = s["styled"] if (column == "styled" and s["styled"]) else s["english"]
            if column == "latin":
                out.append(s["latin"])
            elif column in ("english", "styled"):
                out.append(en or "")
            else:
                out.append(s["latin"])
                out.append(f"    {en or '— not yet translated —'}")
                out.append("")
        offset += page
        if offset >= chunk["total"]:
            break
    body = "\n".join(out)
    # PlainTextResponse: a JSONResponse JSON-encodes the body even when told the
    # media type is text/plain, which shipped a quoted string full of "\n".
    return PlainTextResponse(body, headers={
        "Content-Disposition": f'attachment; filename="doc{doc_id}-{column}.txt"'})


# ---------------------------------------------------------------------------
# Files on disk
# ---------------------------------------------------------------------------

@app.get("/api/files")
def api_files(root: str = "data", path: str = "", q: str = "") -> Dict[str, Any]:
    try:
        return files_mod.listing(root, path, q)
    except files_mod.OutsideRoot as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/files/preview")
def api_file_preview(root: str = "data", path: str = "") -> Dict[str, Any]:
    try:
        return files_mod.preview(root, path)
    except files_mod.OutsideRoot as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/files/download")
def api_file_download(root: str = "data", path: str = "") -> FileResponse:
    try:
        target = files_mod.download_path(root, path)
    except files_mod.OutsideRoot as exc:
        raise HTTPException(400, str(exc))
    if target is None:
        raise HTTPException(404, "not available for download")
    return FileResponse(target, filename=target.name)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class JobRequest(BaseModel):
    kind: str
    params: Dict[str, Any] = {}
    label: str = ""
    not_before: Optional[str] = None      # ISO-8601 UTC, e.g. "2026-09-21T02:00:00+00:00"


@app.get("/api/jobs")
def api_jobs(limit: int = 100, status: str = "") -> Dict[str, Any]:
    jobs = [j.to_dict() for j in queue.list(limit=limit, status=status)]
    # The label says "doc 460"; the person watching wants the title.
    titles = view.titles(j["params"]["doc_id"] for j in jobs if j["params"].get("doc_id"))
    for j in jobs:
        j["title"] = titles.get(j["params"].get("doc_id"))
    return {"items": jobs,
            "running": [j for j in jobs if j["status"] == "running"],
            # False when another instance of the app owns the queue: this page
            # can still enqueue, but jobs run in that other process.
            "worker_active": queue.worker_active,
            "gpus": queue.gpu_status()}


@app.post("/api/jobs")
def api_create_job(req: JobRequest) -> Dict[str, Any]:
    try:
        job = queue.enqueue(req.kind, req.params, label=req.label,
                            not_before=req.not_before)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc))
    return job.to_dict()


class BulkJobRequest(BaseModel):
    """Queue the same kind of job for many documents at once.

    The reason this exists rather than looping in the browser: queueing 400
    documents should be one atomic click, and the UI should not be able to half
    it by navigating away.
    """
    kind: str
    doc_ids: List[int]
    params: Dict[str, Any] = {}
    not_before: Optional[str] = None


@app.post("/api/jobs/bulk")
def api_bulk_jobs(req: BulkJobRequest) -> Dict[str, Any]:
    created = []
    # gpu="alternate": deal the documents out across the cards in turn, so two
    # translations can run side by side (they touch different documents).
    cards = [g.index for g in queue.gpus]
    for n, doc_id in enumerate(req.doc_ids):
        params = dict(req.params, doc_id=doc_id)
        if params.get("gpu") == "alternate":
            if cards:
                params["gpu"] = cards[n % len(cards)]
            else:
                params.pop("gpu")
        try:
            created.append(queue.enqueue(req.kind, params,
                                         not_before=req.not_before).to_dict())
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc))
    return {"created": len(created), "items": created}


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel_job(job_id: int) -> Dict[str, Any]:
    return {"cancelled": queue.cancel(job_id)}


@app.post("/api/jobs/{job_id}/requeue")
def api_requeue_job(job_id: int) -> Dict[str, Any]:
    job = queue.requeue(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/log")
def api_job_log(job_id: int, lines: int = 300) -> Dict[str, Any]:
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return {"job": job.to_dict(), "log": queue.log_tail(job_id, lines=lines)}


@app.post("/api/jobs/clear")
def api_clear_jobs() -> Dict[str, Any]:
    return {"removed": queue.clear_finished()}


# ---------------------------------------------------------------------------
# Source catalogues (browse before you ingest)
# ---------------------------------------------------------------------------

# How each connector's text is obtained, for the "needs OCR" flag in the catalogue.
#   text    -- the source serves finished text; nothing to OCR
#   library -- a scan whose library publishes its own OCR (still a scan)
#   scan    -- page images only: Tesseract (print) or HTR (handwriting) must read them
_OCR_BY_SOURCE = {"mdz": "library", "iiif": "scan", "ocrimages": "scan"}


def _with_ocr_hint(source: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    """Add ``ocr`` (text|library|scan), and for the scan catalogues (vd, europeana)
    the ``identifier`` / ``fetchable`` fields the Find-texts table expects."""
    rec = dict(rec)
    if source in ("vd", "europeana"):
        from ingest.copies import route
        links = rec.get("free_links") or rec.get("shown_at", []) + rec.get("shown_by", [])
        targets = [route(u) for u in links]
        targets = [t for t in targets if t]
        rec.setdefault("identifier", f"{rec['db']}:{rec['ppn']}" if source == "vd" else rec.get("id"))
        rec["fetchable"] = bool(targets)
        rec.setdefault("url", links[0] if links else None)
        rec.setdefault("catalogue", rec.get("db", source).upper() if source == "vd" else rec.get("provider"))
        kinds = {"mdz": "library", "iiif": "scan", "treatises": "text"}
        rec["ocr"] = kinds.get(targets[0][0], "scan") if targets else None
        rec.setdefault("note", None if targets else "no readable digital copy")
    elif source in _OCR_BY_SOURCE:
        rec.setdefault("fetchable", True)
        rec["ocr"] = _OCR_BY_SOURCE[source]
    else:
        rec.setdefault("ocr", "text" if rec.get("fetchable") else None)
    return rec


class DiscoverRequest(BaseModel):
    source: str
    query: str
    limit: int = 25


@app.post("/api/discover")
def api_discover(req: DiscoverRequest) -> Dict[str, Any]:
    """Ask a connector what it can find, without ingesting anything.

    Connectors that implement ``catalog()`` return rich records (title, date,
    whether the item actually has machine-readable text); the rest fall back to
    bare identifiers from ``discover()``.
    """
    try:
        connector = get_connector(req.source)
    except KeyError as exc:
        raise HTTPException(400, str(exc))
    try:
        if hasattr(connector, "catalog"):
            items = connector.catalog(req.query, limit=req.limit)
        else:
            items = [{"identifier": i} for i in
                     connector.discover(req.query, limit=req.limit)]
    except NotImplementedError:
        raise HTTPException(400, f"{req.source} cannot enumerate a catalogue; "
                                 f"ingest a single identifier instead")
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(502, f"{type(exc).__name__}: {exc}")
    items = [_with_ocr_hint(req.source, r) for r in items]
    return {"source": req.source, "query": req.query, "items": items}


# ---------------------------------------------------------------------------
# Summaries (written by "summarize" jobs; searched here)
# ---------------------------------------------------------------------------

@app.get("/api/summaries/search")
def api_summary_search(q: str = "", level: str = "", limit: int = Query(20, le=100)) -> Dict[str, Any]:
    """Hybrid keyword + meaning search over document and part summaries.

    Snippet highlights come back wrapped in control-character sentinels, which
    the client HTML-escapes around before turning into <mark>; see
    core.summaries.HL_OPEN.
    """
    if level not in ("", "document", "part"):
        raise HTTPException(400, "level must be document, part or empty")
    if not q.strip():
        return {"mode": "recent", "items": summaries.recent(limit),
                "stats": summaries.stats(), "hl": [HL_OPEN, HL_CLOSE]}
    stats = summaries.stats()
    vec, note = None, ""
    if stats["embed_model"]:
        try:
            vec = llm.embed(stats["embed_model"], [f"search_query: {q}"], timeout=20)[0]
        except Exception as exc:                               # noqa: BLE001
            note = f"meaning search unavailable ({type(exc).__name__}); keyword only"
    out = summaries.search(q, level=level, limit=limit, query_vec=vec,
                           embed_model=stats["embed_model"])
    out.update({"stats": stats, "note": note, "hl": [HL_OPEN, HL_CLOSE]})
    return out


@app.get("/api/documents/{doc_id}/summaries")
def api_doc_summaries(doc_id: int) -> Dict[str, Any]:
    return summaries.for_document(doc_id)


@app.get("/api/llm/models")
def api_llm_models() -> Dict[str, Any]:
    """Chat models available in the local Ollama, for the summarizer's model picker."""
    try:
        models = llm.models(timeout=5)
    except Exception as exc:                                   # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "models": []}
    chat = [{"name": m["name"], "size_gb": round(m.get("size", 0) / 1e9, 1),
             "params": m.get("details", {}).get("parameter_size")}
            for m in models
            if "embed" not in m["name"] and "completion" in (m.get("capabilities") or ["completion"])]
    return {"available": True, "models": sorted(chat, key=lambda m: m["name"])}


# ---------------------------------------------------------------------------
# Semantic search (lazy)
# ---------------------------------------------------------------------------

def _load_search() -> None:
    global _search_lib, _search_state, _search_error
    try:
        from pipeline import Library
        lib = Library()
        with _search_lock:
            _search_lib, _search_state = lib, "ready"
    except Exception as exc:                                   # noqa: BLE001
        with _search_lock:
            _search_state, _search_error = "failed", f"{type(exc).__name__}: {exc}"


@app.get("/api/search")
def api_search(q: str, k: int = 10, only_untranslated: bool = False,
               stage: str = "") -> Dict[str, Any]:
    global _search_state
    with _search_lock:
        state = _search_state
        if state == "cold":
            _search_state = state = "loading"
            threading.Thread(target=_load_search, name="search-load",
                             daemon=True).start()
    if state != "ready":
        return {"state": state, "error": _search_error, "items": [],
                "message": "Loading the embedder and FAISS index (~600MB); "
                           "try again in a moment."}
    hits = _search_lib.search(q, k=k, only_untranslated_works=only_untranslated,
                              language_stage=(stage or None))
    return {"state": "ready", "items": [{
        "score": h.score, "doc_id": h.document.id, "title": h.document.title,
        "author": h.document.author, "latin": h.segment.latin_text,
        "english": h.segment.english_text, "source_loc": h.segment.source_loc,
    } for h in hits]}


# ---------------------------------------------------------------------------
# Static frontend (mounted last so /api/* wins)
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
