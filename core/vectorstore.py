"""Persisted FAISS vector index keyed by segment id.

Unlike the original in-memory store, vectors are added *with* their segment id
(via IndexIDMap), so a search returns segment ids directly and the index can be
saved to / loaded from disk. The SQLite store remains the source of truth; this
index can always be rebuilt from it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss

from .embedder import EMBEDDING_DIM


class VectorStore:
    """Cosine-similarity search over normalized embeddings, keyed by segment id."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        # IndexFlatIP + normalized vectors == cosine similarity.
        # IndexIDMap lets us attach segment ids as the vector ids.
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    def add(self, segment_ids: List[int], embeddings: np.ndarray) -> None:
        if len(segment_ids) == 0:
            return
        embeddings = np.ascontiguousarray(embeddings.astype("float32"))
        faiss.normalize_L2(embeddings)
        ids = np.asarray(segment_ids, dtype="int64")
        self.index.add_with_ids(embeddings, ids)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Return up to k (segment_id, score) pairs, best first."""
        if self.index.ntotal == 0:
            return []
        q = np.ascontiguousarray(query_embedding.reshape(1, -1).astype("float32"))
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        return [
            (int(sid), float(score))
            for sid, score in zip(ids[0], scores[0])
            if sid != -1
        ]

    @property
    def size(self) -> int:
        return self.index.ntotal

    def save(self, path: str = "data/index.faiss") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path)

    def load(self, path: str = "data/index.faiss") -> None:
        if Path(path).exists():
            self.index = faiss.read_index(path)
