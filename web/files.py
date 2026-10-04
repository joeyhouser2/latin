"""Read-only browser over the project's data directory.

The library's SQLite corpus is only half the picture: raw OCR dumps, id lists
for scoped stylizer passes, harvest logs and PDF exports all live as files under
``data/`` and are the things you actually want to look at when deciding what to
ingest or re-run next. This module lists them and serves previews/downloads.

Two guardrails:

* every path is resolved and checked to be inside a configured root, so a
  ``../..`` in a query string cannot walk out of ``data/``;
* the SQLite corpus files are listed but never served. Copying ``corpus.db``
  byte-for-byte while a job holds it open in WAL mode produces a torn file that
  looks fine until it isn't -- taking a backup is ``sqlite3``'s backup API's
  job, not a download link's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Named roots the browser can see. Everything else on disk is out of reach.
ROOTS: Dict[str, Path] = {
    "data": REPO_ROOT / "data",
    "models": REPO_ROOT / "models",
    "docs": REPO_ROOT / "docs",
}

# Never served, whatever the extension filter says (see the module docstring).
BLOCKED_SUFFIXES = {".db", ".db-wal", ".db-shm", ".sqlite", ".sqlite3", ".faiss"}
BLOCKED_PATTERNS = ("corpus.db.bak-",)

TEXT_SUFFIXES = {".txt", ".json", ".md", ".log", ".csv", ".tsv", ".xml", ".py",
                 ".ps1", ".yaml", ".yml", ".err"}
PREVIEW_BYTES = 200_000        # ~200KB is plenty to judge an OCR dump by


class OutsideRoot(ValueError):
    """Raised when a requested path escapes its root."""


@dataclass
class Entry:
    name: str
    path: str                  # root-relative, forward slashes: the API's handle
    is_dir: bool
    size: int
    modified: str
    kind: str                  # "dir" | "text" | "pdf" | "binary" | "blocked"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "path": self.path, "is_dir": self.is_dir,
            "size": self.size, "modified": self.modified, "kind": self.kind,
            "downloadable": self.kind not in ("dir", "blocked"),
        }


def resolve(root: str, rel: str = "") -> Path:
    """Map (root, relative path) to an absolute path inside that root."""
    if root not in ROOTS:
        raise OutsideRoot(f"unknown root {root!r}")
    base = ROOTS[root].resolve()
    target = (base / rel.lstrip("/\\")).resolve() if rel else base
    if target != base and base not in target.parents:
        raise OutsideRoot(f"{rel!r} is outside {root!r}")
    return target


def _kind(path: Path) -> str:
    if path.is_dir():
        return "dir"
    name, suffix = path.name.lower(), path.suffix.lower()
    if suffix in BLOCKED_SUFFIXES or any(p in name for p in BLOCKED_PATTERNS):
        return "blocked"
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix == ".pdf":
        return "pdf"
    return "binary"


def _entry(path: Path, root: str) -> Entry:
    stat = path.stat()
    rel = path.relative_to(ROOTS[root].resolve()).as_posix()
    return Entry(
        name=path.name, path=rel, is_dir=path.is_dir(),
        size=0 if path.is_dir() else stat.st_size,
        modified=datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                         .isoformat(timespec="seconds"),
        kind=_kind(path),
    )


def listing(root: str = "data", rel: str = "", q: str = "") -> Dict[str, Any]:
    """One directory's contents: directories first, then files by recency.

    Recency rather than name because this directory is a work log -- the file
    you want is almost always the one the last run wrote.
    """
    target = resolve(root, rel)
    if not target.exists():
        return {"root": root, "path": rel, "entries": [], "missing": True}
    entries: List[Entry] = []
    with os.scandir(target) as it:
        for de in it:
            if de.name.startswith(".") or de.name == "__pycache__":
                continue
            if q and q.lower() not in de.name.lower():
                continue
            try:
                entries.append(_entry(Path(de.path), root))
            except OSError:
                continue          # vanished or locked mid-scan; skip it
    entries.sort(key=lambda e: (not e.is_dir, e.is_dir and e.name.lower() or ""))
    dirs = [e for e in entries if e.is_dir]
    files = sorted((e for e in entries if not e.is_dir),
                   key=lambda e: e.modified, reverse=True)
    parent = None if not rel else str(Path(rel).parent.as_posix()).replace(".", "")
    return {
        "root": root, "roots": sorted(ROOTS), "path": rel, "parent": parent,
        "entries": [e.to_dict() for e in dirs + files],
    }


def preview(root: str, rel: str) -> Dict[str, Any]:
    """The head of a text file, for a quick look without downloading it."""
    target = resolve(root, rel)
    if not target.is_file():
        return {"error": "not a file"}
    kind = _kind(target)
    if kind == "blocked":
        return {"error": "SQLite/index files are not served; use the sqlite3 "
                         "backup API to copy the corpus safely", "kind": kind}
    if kind not in ("text",):
        return {"kind": kind, "size": target.stat().st_size,
                "error": "no text preview for this file type"}
    size = target.stat().st_size
    with open(target, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read(PREVIEW_BYTES)
    return {"kind": kind, "size": size, "truncated": size > PREVIEW_BYTES,
            "text": text, "path": rel, "root": root}


def download_path(root: str, rel: str) -> Optional[Path]:
    """The absolute path to serve, or None if this file must not be served."""
    target = resolve(root, rel)
    if not target.is_file() or _kind(target) == "blocked":
        return None
    return target
