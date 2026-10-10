"""Known-word scoring for Greek transcriptions (the Greek counterpart of ``abbrev.vocab_hit_rate``).

The vocabulary is every accent-and-case-folded Greek word occurring at least twice in the
corpus's Greek documents (read-only; cached in ``data/cache/greek_vocab.json``). A
transcription from a model that reads the hand scores ~0.5+; Latin-trained models on Greek
letters, or any model on a script it does not know, score near 0.1.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent
_CACHE = _REPO / "data" / "cache" / "greek_vocab.json"
_WORD = re.compile(r"[Ͱ-Ͽἀ-῿]{3,}")


def fold(word: str) -> str:
    """Lowercase, strip accents/breathings, final sigma -> sigma."""
    d = unicodedata.normalize("NFD", word.lower())
    d = "".join(c for c in d if not unicodedata.combining(c))
    return d.replace("ς", "σ")


def build_vocab(db_path: Optional[str] = None, min_count: int = 2, refresh: bool = False) -> Counter:
    if _CACHE.exists() and not refresh:
        return Counter(json.loads(_CACHE.read_text(encoding="utf-8")))
    path = Path(db_path or _REPO / "data" / "corpus.db")
    if not path.exists():
        return Counter()
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=30)
    counts: Counter = Counter()
    for (text,) in conn.execute(
            """SELECT g.latin_text FROM segments g JOIN sections s ON s.id = g.section_id
               JOIN documents d ON d.id = s.doc_id WHERE d.language = 'grc'"""):
        counts.update(fold(w) for w in _WORD.findall(text))
    conn.close()
    vocab = Counter({w: n for w, n in counts.items() if n >= min_count})
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    return vocab


def hit_rate(text: str, vocab: Counter) -> float:
    """Share of 3+ letter Greek tokens found in the vocabulary."""
    toks = [fold(w) for w in _WORD.findall(text)]
    return sum(1 for w in toks if vocab.get(w, 0) > 0) / len(toks) if toks else 0.0
