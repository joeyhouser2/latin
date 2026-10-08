"""Persisted job queue for the long-running corpus work.

A job is one invocation of an existing CLI script (``scripts/translate_pending.py``
and friends) run as a *subprocess*, with its stdout tee'd to a log file and its
``overall N/M`` progress lines parsed back into the queue row. Three reasons it
is a subprocess and not a thread:

* the scripts already exist, are resumable mid-document, and are the code path
  that has actually been run over this corpus for months;
* a CUDA OOM or a hung tokenizer takes down one worker process, not the web app;
* cancelling means killing a pid, which is the only reliable way to stop a
  half-finished ``generate()`` call.

The queue lives in its own SQLite file (``data/jobs.db``), deliberately *not* in
``corpus.db``: a translation run holds corpus.db open for hours from another
process, and adding the web app as a second writer to that same WAL database is
exactly the failure this project has been bitten by before.

Scheduling: one job per GPU, one corpus writer at a time
--------------------------------------------------------
Each job kind declares whether it needs a GPU and whether it writes
``corpus.db``. The scheduler runs as many jobs as there are free GPUs, pinning
each to its own card via ``CUDA_VISIBLE_DEVICES``, with one hard rule on top:
**corpus-writing jobs must not overlap in what they write**. Two jobs scoped to
different single documents (``doc_id``) may run at once, one per card -- each
commits short per-chunk transactions under a 30 s busy timeout, so they
interleave safely. Anything unscoped (a whole source, a language, an ingest, a
reindex) is exclusive and waits for every other writer. So an unscoped
translation still queues behind others, but a summarization job
-- which only *reads* corpus.db and writes its own ``summaries.db`` -- starts on
whichever card the translation is not using. Queued jobs that are blocked are
skipped over, not waited on, so a summary queued behind ten translations does
not sit idle while a GPU does.

GPUs are addressed by **UUID**, not index. ``nvidia-smi`` numbers cards by PCI
bus; CUDA, by default, numbers them fastest-first; Ollama's docs say outright
that numeric ids "may vary". On this machine the two orders disagree (a 4060 Ti
and a 4070 SUPER), so an index-pinned job could land on exactly the card another
job was already using. A UUID means the same card to every consumer.

One worker per queue, enforced
------------------------------
Only one process may run the worker for a given jobs.db, enforced by an OS file
lock next to it. Before this existed, double-clicking the launcher while the app
was already open produced a second server that failed to bind its port but kept
its worker thread polling the same queue -- two workers, no cross-process claim,
and the same job could run twice. Now a second instance serves pages but runs
nothing, and claiming a job is additionally a conditional UPDATE, atomic across
processes.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.models import LANGUAGES


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = Path(os.environ.get("LATIN_JOBS_DB") or REPO_ROOT / "data" / "jobs.db")
LOG_DIR = Path(os.environ.get("LATIN_JOB_LOGS") or REPO_ROOT / "data" / "joblogs")

# Terminal states: a job in one of these will never run again on its own.
FINISHED = ("done", "failed", "cancelled", "interrupted")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT    NOT NULL,
    label       TEXT    NOT NULL,
    params      TEXT    NOT NULL,          -- JSON dict passed to the kind's argv builder
    status      TEXT    NOT NULL,          -- queued|running|done|failed|cancelled|interrupted
    not_before  TEXT,                      -- ISO-8601 UTC; NULL = run as soon as a slot is free
    created_at  TEXT    NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    pid         INTEGER,
    log_path    TEXT,
    done        INTEGER NOT NULL DEFAULT 0,   -- units processed so far (parsed from stdout)
    total       INTEGER NOT NULL DEFAULT 0,   -- units the script said it would process
    exit_code   INTEGER,
    error       TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, not_before, id);
"""

# The scripts print e.g. "... | overall 12,345/98,765 3.2 seg/s ETA 1.4h".
# translate_pending.py, stylize_library.py and summarize.py share this format,
# so one regex drives the progress bar for all of them.
_PROGRESS_RE = re.compile(r"overall\s+([\d,]+)\s*/\s*([\d,]+)")
# The banner printed before any work: "=== Translate pending: 12 docs, 98,765 segments ==="
_TOTAL_RE = re.compile(r"([\d,]+)\s+(?:segments|chunks)")


# ---------------------------------------------------------------------------
# GPUs
# ---------------------------------------------------------------------------

@dataclass
class Gpu:
    index: str        # nvidia-smi's (PCI bus order) index -- for display only
    uuid: str         # what CUDA_VISIBLE_DEVICES actually gets
    name: str
    memory_total: int   # MiB
    memory_used: int    # MiB

    @property
    def memory_free(self) -> int:
        return self.memory_total - self.memory_used


def detect_gpus() -> List[Gpu]:
    """The NVIDIA GPUs on this machine, via nvidia-smi (no torch import).

    ``LATIN_GPUS`` narrows the set: a comma list of nvidia-smi indices, or
    ``none`` to run every job on CPU, one at a time.
    """
    wanted = os.environ.get("LATIN_GPUS", "").strip().lower()
    if wanted == "none":
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            gpus.append(Gpu(parts[0], parts[1], parts[2], int(parts[3]), int(parts[4])))
        except ValueError:
            continue
    if wanted:
        keep = {w.strip() for w in wanted.split(",")}
        gpus = [g for g in gpus if g.index in keep]
    return gpus


# ---------------------------------------------------------------------------
# Job kinds: what a queued job actually runs
# ---------------------------------------------------------------------------

def _flag(argv: List[str], flag: str, value: Any) -> None:
    """Append ``--flag value`` unless value is empty/None (or ``--flag`` if True)."""
    if value is None or value == "" or value is False:
        return
    if value is True:
        argv.append(flag)
    else:
        argv.extend([flag, str(value)])


def _translate_argv(p: Dict[str, Any]) -> List[str]:
    argv = [sys.executable, "-u", "scripts/translate_pending.py"]
    _flag(argv, "--doc-id", p.get("doc_id"))
    _flag(argv, "--source-prefix", p.get("source_prefix"))
    _flag(argv, "--language", p.get("language"))
    _flag(argv, "--batch-size", p.get("batch_size"))
    _flag(argv, "--chunk", p.get("chunk"))
    _flag(argv, "--max-length", p.get("max_length"))
    _flag(argv, "--skip-translated", p.get("skip_translated"))
    # Re-translate German editorial segments already translated as Latin
    # (see ingest/german_detect.py); leaves every other segment untouched.
    _flag(argv, "--retranslate-german", p.get("retranslate_german"))
    if p.get("section_first") and p.get("section_last"):
        argv.extend(["--section-range", str(p["section_first"]), str(p["section_last"])])
    return argv


def _stylize_argv(p: Dict[str, Any]) -> List[str]:
    argv = [sys.executable, "-u", "scripts/stylize_library.py"]
    _flag(argv, "--doc-id", p.get("doc_id"))
    _flag(argv, "--source-prefix", p.get("source_prefix"))
    _flag(argv, "--language", p.get("language"))
    _flag(argv, "--preset", p.get("preset"))
    _flag(argv, "--backend", p.get("backend"))
    _flag(argv, "--batch-size", p.get("batch_size"))
    _flag(argv, "--limit", p.get("limit"))
    _flag(argv, "--skip-poetry", p.get("skip_poetry"))
    if p.get("section_first") and p.get("section_last"):
        argv.extend(["--section-range", str(p["section_first"]), str(p["section_last"])])
    return argv


# What the UI calls an OCR engine -> the options the scan connectors understand.
# `mdz` reads `ocr`, `iiif` reads `mode`; each ignores the other's key.
OCR_ENGINES = {
    "auto": {},                                     # let the connector decide
    "library": {"ocr": "mdz"},                      # the library's own OCR (MDZ hOCR)
    "print": {"ocr": "tesseract", "mode": "print"},  # Tesseract (Latin / Fraktur)
    "htr": {"mode": "htr", "ocr": "tesseract"},     # handwriting recognition (Kraken)
}


def ocr_fragment(p: Dict[str, Any]) -> str:
    """``#key=val&...`` for the scan connectors from the job's ocr/pages params."""
    engine = p.get("ocr") or "auto"
    if engine not in OCR_ENGINES:
        raise ValueError(f"ocr must be one of {sorted(OCR_ENGINES)}, not {engine!r}")
    opts = dict(OCR_ENGINES[engine])
    if p.get("pages"):
        pages = str(p["pages"]).strip()
        if not re.fullmatch(r"\d*-?\d*", pages):
            raise ValueError(f"pages must look like 1-40, not {pages!r}")
        opts["pages"] = pages
    return "#" + "&".join(f"{k}={v}" for k, v in opts.items()) if opts else ""


def _ingest_argv(p: Dict[str, Any]) -> List[str]:
    ident = p["identifier"]
    if "#" not in ident:                 # an explicit fragment from the user wins
        ident += ocr_fragment(p)
    argv = [sys.executable, "-u", "scripts/ingest.py", p["source"], ident]
    _flag(argv, "--discover", p.get("discover"))
    _flag(argv, "--limit", p.get("limit"))
    _flag(argv, "--author", p.get("author"))
    _flag(argv, "--title", p.get("title"))
    _flag(argv, "--genre", p.get("genre"))
    _flag(argv, "--century", p.get("century"))
    _flag(argv, "--stage", p.get("stage"))
    _flag(argv, "--language", p.get("language"))
    _flag(argv, "--translation-status", p.get("translation_status"))
    return argv


def _reindex_argv(p: Dict[str, Any]) -> List[str]:
    return [sys.executable, "-u", "scripts/reindex.py"]


def _summarize_argv(p: Dict[str, Any]) -> List[str]:
    argv = [sys.executable, "-u", "scripts/summarize.py"]
    ids = list(p.get("doc_ids") or [])
    if p.get("doc_id"):
        ids.append(p["doc_id"])
    for doc_id in ids:
        argv.extend(["--doc-id", str(int(doc_id))])
    if not ids and not p.get("all"):
        raise ValueError("summarize needs doc_id(s) or all=true")
    _flag(argv, "--all", p.get("all"))
    _flag(argv, "--model", p.get("model"))
    _flag(argv, "--source", p.get("source"))
    _flag(argv, "--min-translated", p.get("min_translated"))
    _flag(argv, "--limit", p.get("limit"))
    _flag(argv, "--force", p.get("force"))
    return argv


@dataclass(frozen=True)
class KindSpec:
    build: Callable[[Dict[str, Any]], List[str]]
    description: str
    writes_corpus: bool     # at most one of these runs at a time
    uses_gpu: bool          # gets its own card, pinned by UUID


JOB_KINDS: Dict[str, KindSpec] = {
    "translate": KindSpec(_translate_argv, "Translate untranslated segments", True, True),
    "stylize":   KindSpec(_stylize_argv,   "Stylize translated segments", True, True),
    # Ingest writes the corpus *and* embeds on the GPU as it goes.
    "ingest":    KindSpec(_ingest_argv,    "Ingest from a source connector", True, True),
    "reindex":   KindSpec(_reindex_argv,   "Rebuild the FAISS index", True, True),
    # Reads corpus.db, writes data/summaries.db: free to run beside a translation.
    "summarize": KindSpec(_summarize_argv, "Summarize translated documents (local LLM)",
                          False, True),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: int
    kind: str
    label: str
    params: Dict[str, Any]
    status: str
    not_before: Optional[str]
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    pid: Optional[int]
    log_path: Optional[str]
    done: int
    total: int
    exit_code: Optional[int]
    error: Optional[str]
    gpu: Optional[str] = None       # nvidia-smi index of the card it ran on
    note: Optional[str] = None      # e.g. "nothing to do" -- neither error nor silence

    @property
    def percent(self) -> Optional[float]:
        if not self.total:
            return None
        return min(100.0, 100.0 * self.done / self.total)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["percent"] = self.percent
        return d


def write_scope(kind: str, params: Dict[str, Any]) -> Optional[frozenset]:
    """The documents a corpus-writing job may touch; ``None`` = anything (exclusive).

    Only translate/stylize scoped to one document have a narrow scope. Re-translating
    German makes a backup of the whole database first, so it stays exclusive.
    """
    if kind in ("translate", "stylize") and params.get("doc_id")             and not params.get("retranslate_german"):
        return frozenset({int(params["doc_id"])})
    return None


def scopes_conflict(a: Optional[frozenset], b: Optional[frozenset]) -> bool:
    return a is None or b is None or bool(a & b)


@dataclass
class _Running:
    job_id: int
    kind: str
    gpu: Optional[Gpu]
    scope: Optional[frozenset] = None
    proc: Optional[subprocess.Popen] = None
    thread: Optional[threading.Thread] = None


class JobQueue:
    """The queue itself: enqueue/list/cancel, plus the scheduler.

    Thread-safe for its callers (FastAPI handlers, the scheduler thread, one
    thread per running job) via one lock around a single connection --
    ``check_same_thread=False`` plus a mutex, rather than a pool, because the
    write volume is a few rows a minute.
    """

    def __init__(self, db_path: str | os.PathLike = DEFAULT_DB,
                 log_dir: str | os.PathLike = LOG_DIR,
                 repo_root: str | os.PathLike = REPO_ROOT,
                 gpus: Optional[List[Gpu]] = None):
        self.db_path = str(db_path)
        self.log_dir = Path(log_dir)
        self.repo_root = Path(repo_root)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

        self.gpus: List[Gpu] = detect_gpus() if gpus is None else gpus
        self._stop = threading.Event()
        self._scheduler: Optional[threading.Thread] = None
        self._running: Dict[int, _Running] = {}
        self._worker_lock_fh = None
        self.worker_active = False

    def _migrate(self) -> None:
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(jobs)")}
        for col in ("gpu", "note"):
            if col not in cols:
                self.conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} TEXT")

    # -- lifecycle -----------------------------------------------------------

    def _acquire_worker_lock(self) -> bool:
        """Take the exclusive per-queue worker lock; False if another process has it.

        An OS-level lock on a file beside jobs.db, released by the OS when the
        holder exits however it exits -- so a crashed server never leaves the
        queue permanently locked, which a lock *file* (exists/doesn't) would.
        """
        path = self.db_path + ".worker.lock"
        fh = open(path, "a+")
        try:
            fh.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._worker_lock_fh = fh
        return True

    def _adopt_orphans(self) -> None:
        """Mark jobs left 'running' by a previous worker as interrupted.

        Only ever called while holding the worker lock: a second instance must
        not do this, because the 'running' rows it would see belong to the
        first instance's live jobs, not to a dead predecessor.
        """
        with self._lock:
            self.conn.execute(
                "UPDATE jobs SET status='interrupted', finished_at=?, "
                "error='interrupted when the web app stopped' WHERE status='running'",
                (_now(),),
            )
            self.conn.commit()

    def start_worker(self) -> bool:
        """Start the scheduler if this process can take the worker lock.

        Returns whether it did. When it cannot, this process still serves the
        UI and can enqueue -- the process holding the lock runs the jobs.
        """
        if self._scheduler and self._scheduler.is_alive():
            return True
        if not self._acquire_worker_lock():
            print("[jobs] another instance holds the worker lock for "
                  f"{self.db_path}; this one will serve pages but run no jobs.",
                  flush=True)
            self.worker_active = False
            return False
        self._adopt_orphans()
        self.worker_active = True
        gpus = ", ".join(f"{g.index}:{g.name}" for g in self.gpus) or "none (CPU)"
        print(f"[jobs] worker running; GPUs: {gpus}", flush=True)
        self._stop.clear()
        self._scheduler = threading.Thread(target=self._run_loop, name="job-scheduler",
                                           daemon=True)
        self._scheduler.start()
        return True

    def shutdown(self, kill_running: bool = False) -> None:
        """Stop picking up new jobs; optionally kill the ones in flight.

        Default is to leave running jobs alone: they are separate processes
        doing hours of GPU work, and shutting down the web app should not throw
        that away. Note the limit of that promise on Windows -- children share
        this process's console, so closing the launcher *window* still ends them,
        since Windows signals every process attached to a console being closed.
        Either way nothing is lost: the rows are reconciled as "interrupted" on
        the next startup and the work resumes from the last committed chunk.
        """
        self._stop.set()
        if kill_running:
            for job_id in list(self._running):
                self.cancel_running(job_id)
        if self._scheduler:
            self._scheduler.join(timeout=5)

    # -- queue operations ----------------------------------------------------

    def enqueue(self, kind: str, params: Dict[str, Any], label: str = "",
                not_before: Optional[str] = None) -> Job:
        if kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {kind!r}; choose from {sorted(JOB_KINDS)}")
        # Fail fast on a bad param set: build the argv now, so a typo surfaces as
        # a 400 on the button click rather than as a job that dies seconds in.
        JOB_KINDS[kind].build(params)
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO jobs (kind, label, params, status, not_before, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (kind, label or describe(kind, params), json.dumps(params),
                 "queued", not_before, _now()),
            )
            self.conn.commit()
            job_id = cur.lastrowid
        return self.get(job_id)

    def get(self, job_id: int) -> Optional[Job]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def list(self, limit: int = 100, status: str = "") -> List[Job]:
        sql = "SELECT * FROM jobs"
        args: List[Any] = []
        if status:
            sql += " WHERE status = ?"
            args.append(status)
        # Active work first, then most recent -- the order you want at a glance.
        sql += (" ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 "
                "ELSE 2 END, id DESC LIMIT ?")
        args.append(limit)
        with self._lock:
            rows = self.conn.execute(sql, args).fetchall()
        return [_row_to_job(r) for r in rows]

    def cancel(self, job_id: int) -> bool:
        """Cancel a queued job, or kill a running one."""
        job = self.get(job_id)
        if job is None or job.status in FINISHED:
            return False
        if job.status == "running":
            return self.cancel_running(job_id)
        with self._lock:
            self.conn.execute(
                "UPDATE jobs SET status='cancelled', finished_at=? WHERE id=? AND status='queued'",
                (_now(), job_id),
            )
            self.conn.commit()
        return True

    def cancel_running(self, job_id: int) -> bool:
        run = self._running.get(job_id)
        if run is None or run.proc is None:
            return False
        # The job's thread sees the non-zero return code and writes the final
        # status; marking it here first keeps the UI from flashing "failed".
        self._mark(job_id, status="cancelled")
        proc = run.proc
        try:
            if os.name == "nt":
                # CTRL_BREAK reaches the child's own handler (a clean exit
                # mid-chunk, so the chunk's work is committed, and summarize.py
                # gets to shut its private Ollama down); kill if it ignores us.
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            return False
        return True

    def requeue(self, job_id: int) -> Optional[Job]:
        """Queue a fresh copy of a finished job (same kind + params)."""
        job = self.get(job_id)
        if job is None:
            return None
        return self.enqueue(job.kind, job.params, label=job.label,
                            not_before=job.not_before)

    def clear_finished(self) -> int:
        with self._lock:
            cur = self.conn.execute(
                f"DELETE FROM jobs WHERE status IN ({','.join('?' * len(FINISHED))})",
                FINISHED,
            )
            self.conn.commit()
            return cur.rowcount

    def log_tail(self, job_id: int, lines: int = 200) -> str:
        job = self.get(job_id)
        if job is None or not job.log_path or not os.path.isfile(job.log_path):
            return ""
        # Logs top out in the low megabytes for a multi-hour run; reading the
        # whole file and slicing is simpler than seeking and fast enough.
        with open(job.log_path, "r", encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:])

    def gpu_status(self) -> List[Dict[str, Any]]:
        """Each card, fresh memory figures, and which job (if any) holds it."""
        fresh = {g.uuid: g for g in detect_gpus()}
        by_gpu = {r.gpu.uuid: r.job_id for r in self._running.values() if r.gpu}
        out = []
        for g in self.gpus:
            cur = fresh.get(g.uuid, g)
            out.append({"index": g.index, "name": g.name,
                        "memory_total": cur.memory_total, "memory_used": cur.memory_used,
                        "job_id": by_gpu.get(g.uuid)})
        return out

    # -- scheduler -----------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                claimed = self._claim_next()
            except Exception as exc:                        # noqa: BLE001
                print(f"[jobs] scheduler error: {type(exc).__name__}: {exc}", flush=True)
                claimed = None
            if claimed is None:
                self._stop.wait(2.0)
            # else: loop straight round -- another slot may be free too.

    def _claim_next(self) -> Optional[int]:
        """Start the oldest queued job that can run right now, if any.

        "Can run" = its start time has passed, it does not need a GPU that is
        taken, and it is not a corpus writer while another writer is running.
        """
        now = _now()
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE status='queued' "
                "AND (not_before IS NULL OR not_before <= ?) ORDER BY id",
                (now,),
            ).fetchall()
        if not rows:
            return None

        running = list(self._running.values())
        writers = [r for r in running
                   if r.kind in JOB_KINDS and JOB_KINDS[r.kind].writes_corpus]
        busy = {r.gpu.uuid for r in running if r.gpu}
        cpu_busy = any(r.gpu is None and JOB_KINDS.get(r.kind, None) is not None
                       and JOB_KINDS[r.kind].uses_gpu for r in running)

        for row in rows:
            spec = JOB_KINDS.get(row["kind"])
            if spec is None:
                self._mark(row["id"], status="failed", finished_at=now,
                           error=f"unknown job kind {row['kind']!r}")
                continue
            params = json.loads(row["params"])
            scope = write_scope(row["kind"], params)
            if spec.writes_corpus and any(scopes_conflict(scope, w.scope) for w in writers):
                continue
            gpu: Optional[Gpu] = None
            if spec.uses_gpu:
                if self.gpus:
                    gpu = self._pick_gpu(busy, params)
                    if gpu is None:
                        continue
                elif cpu_busy:
                    continue        # no GPUs: GPU-class jobs share the CPU one at a time

            # Conditional UPDATE: atomic across processes, so even a second
            # worker (which the lock should prevent) could not double-claim.
            with self._lock:
                cur = self.conn.execute(
                    "UPDATE jobs SET status='running', started_at=?, gpu=? "
                    "WHERE id=? AND status='queued'",
                    (now, gpu.index if gpu else None, row["id"]),
                )
                self.conn.commit()
            if cur.rowcount != 1:
                continue
            run = _Running(job_id=row["id"], kind=row["kind"], gpu=gpu, scope=scope)
            self._running[row["id"]] = run
            run.thread = threading.Thread(target=self._run_job_thread, args=(run,),
                                          name=f"job-{row['id']}", daemon=True)
            run.thread.start()
            return row["id"]
        return None

    def _pick_gpu(self, busy: set, params: Dict[str, Any]) -> Optional[Gpu]:
        """A free card for a job: an explicit pin if the job asked, else the free
        card with the most free memory right now."""
        free = [g for g in self.gpus if g.uuid not in busy]
        want = params.get("gpu")
        if want not in (None, "", "any", "alternate"):
            match = [g for g in free if g.index == str(want) or g.uuid == str(want)]
            return match[0] if match else None
        if not free:
            return None
        fresh = {g.uuid: g for g in detect_gpus()}
        return max(free, key=lambda g: fresh.get(g.uuid, g).memory_free)

    def _run_job_thread(self, run: _Running) -> None:
        try:
            self._run_job(run)
        except Exception as exc:                            # noqa: BLE001
            self._mark(run.job_id, status="failed", finished_at=_now(),
                       error=f"{type(exc).__name__}: {exc}")
        finally:
            self._running.pop(run.job_id, None)

    def _run_job(self, run: _Running) -> None:
        job = self.get(run.job_id)
        argv = JOB_KINDS[job.kind].build(job.params)
        log_path = self.log_dir / f"job{job.id:05d}-{job.kind}.log"
        self._mark(job.id, log_path=str(log_path))

        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   CUDA_DEVICE_ORDER="PCI_BUS_ID")
        if run.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = run.gpu.uuid
            env["LATIN_GPU_PINNED"] = "1"     # ingest.htr must not re-pick a card

        creationflags = 0
        if os.name == "nt":
            # Needed for CTRL_BREAK_EVENT to be deliverable to the child on cancel.
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            where = (f"GPU {run.gpu.index} ({run.gpu.name}, {run.gpu.uuid})"
                     if run.gpu else "CPU")
            log.write(f"$ {' '.join(argv)}\n# on {where}\n\n")
            log.flush()
            proc = subprocess.Popen(
                argv, cwd=str(self.repo_root), env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1, creationflags=creationflags,
            )
            run.proc = proc
            self._mark(job.id, pid=proc.pid)

            last_db = last_log = 0.0
            latest = None
            for line in proc.stdout:                       # type: ignore[union-attr]
                log.write(line)
                done, total = _parse_progress(line)
                if done is not None:
                    latest = (done, total)
                    # Throttle: these scripts print a progress line per chunk,
                    # several a second on a fast GPU -- no DB write for each.
                    if time.time() - last_db > 1.0:
                        self._mark(job.id, done=done, total=total)
                        last_db = time.time()
                if time.time() - last_log > 5.0:
                    log.flush()
                    last_log = time.time()
            # The throttle above almost always swallows the *last* progress line,
            # which is the one that reads 100%. Write it now, or every finished
            # job sits at "40/50" in the UI forever.
            if latest is not None:
                self._mark(job.id, done=latest[0], total=latest[1])
            code = proc.wait()

        final = self.get(job.id)
        # cancel() already wrote 'cancelled'; don't overwrite it with 'failed'.
        if final and final.status == "cancelled":
            self._mark(job.id, finished_at=_now(), exit_code=code, pid=None)
            return
        note = None
        if code == 0 and latest is not None and latest[1] == 0:
            # Succeeded at doing nothing -- e.g. stylizing a document that has no
            # translations yet. Say so, rather than showing it as a green "done"
            # that looks like work happened.
            note = "nothing to do: 0 pending"
        self._mark(
            job.id, status=("done" if code == 0 else "failed"),
            finished_at=_now(), exit_code=code, pid=None, note=note,
            error=None if code == 0 else f"exited with code {code}; see the log",
        )

    def _mark(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self.conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?",
                              [*fields.values(), job_id])
            self.conn.commit()


def _parse_progress(line: str):
    """Pull (done, total) out of a script's progress line, if it has one."""
    m = _PROGRESS_RE.search(line)
    if m:
        return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
    # The opening banner carries the total before any work has been done, so the
    # bar can show 0 / 98,765 rather than nothing at all for the first minute.
    if line.startswith("==="):
        m = _TOTAL_RE.search(line)
        if m:
            return 0, int(m.group(1).replace(",", ""))
    return None, None


def describe(kind: str, params: Dict[str, Any]) -> str:
    """A short human label for a job, used when the caller supplies none."""
    pin = f" [GPU {params['gpu']}]" if params.get("gpu") not in (None, "", "any") else ""
    return _describe(kind, params) + pin


def _describe(kind: str, params: Dict[str, Any]) -> str:
    if kind in ("translate", "stylize"):
        verb = "Translate" if kind == "translate" else f"Stylize ({params.get('preset', 'victorian_prose')})"
        if kind == "translate" and params.get("retranslate_german"):
            verb = "Re-translate German"
        if params.get("doc_id"):
            scope = f"doc {params['doc_id']}"
            if params.get("section_first"):
                scope += f" §{params['section_first']}–{params['section_last']}"
        elif params.get("source_prefix"):
            scope = f"source '{params['source_prefix']}'"
        elif params.get("language"):
            scope = "all " + LANGUAGES.get(params["language"], params["language"])
        else:
            scope = "the whole library"
        return f"{verb}: {scope}"
    if kind == "summarize":
        ids = list(params.get("doc_ids") or []) + ([params["doc_id"]] if params.get("doc_id") else [])
        if ids:
            scope = f"doc {ids[0]}" if len(ids) == 1 else f"{len(ids)} docs"
        else:
            scope = "all translated docs"
        return f"Summarize: {scope}" + (f" ({params['model']})" if params.get("model") else "")
    if kind == "ingest":
        what = params.get("identifier", "")
        ocr = params.get("ocr")
        extra = ((" (discover)" if params.get("discover") else "")
                 + (f" [OCR: {ocr}]" if ocr and ocr != "auto" else "")
                 + (f" pp.{params['pages']}" if params.get("pages") else ""))
        return f"Ingest {params.get('source')}: {what[:60]}{extra}"
    spec = JOB_KINDS.get(kind)
    return spec.description if spec else kind


def _row_to_job(row: sqlite3.Row) -> Job:
    keys = row.keys()
    return Job(
        id=row["id"], kind=row["kind"], label=row["label"],
        params=json.loads(row["params"]), status=row["status"],
        not_before=row["not_before"], created_at=row["created_at"],
        started_at=row["started_at"], finished_at=row["finished_at"],
        pid=row["pid"], log_path=row["log_path"], done=row["done"],
        total=row["total"], exit_code=row["exit_code"], error=row["error"],
        gpu=row["gpu"] if "gpu" in keys else None,
        note=row["note"] if "note" in keys else None,
    )
