"""Connector for local plain-text files.

fetch(path) reads one .txt file as a single work; discover(dir) lists .txt files
in a directory for bulk ingestion. Lets you bring any text you already have on
disk into the library.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .base import Connector, RawWork


class FileConnector(Connector):
    name = "file"

    def fetch(self, path: str, **meta_overrides) -> RawWork:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        meta = {
            "title": p.stem,
            "source": f"local file ({p.name})",
            "language_stage": "unknown",
            "has_existing_translation": False,
        }
        meta.update(meta_overrides)
        return meta, [("Text", text)]

    def discover(self, directory: str, limit: int = 500) -> List[str]:
        paths = sorted(str(p) for p in Path(directory).rglob("*.txt"))
        return paths[:limit]
