"""Multilingual sentence embeddings for semantic search.

A multilingual model lets a user query in English OR Latin and still match Latin
passages, because both land near each other in the same vector space.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # for the default model


class Embedder:
    """Wraps a SentenceTransformer. Loaded lazily on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str], batch_size: int = 32,
              show_progress: bool = False, device: Optional[str] = None) -> np.ndarray:
        """Embed a list of texts -> [n, dim] float32 array.

        ``device`` (e.g. "cpu") runs this one call there. SentenceTransformer.encode
        moves the model to that device and leaves it there, so the model is moved
        back afterwards; if that move fails (GPU still full) it stays put and the
        next call runs on the fallback device too.
        """
        model = self.model
        home = model.device
        try:
            embeddings = model.encode(
                texts, batch_size=batch_size, device=device,
                show_progress_bar=show_progress, convert_to_numpy=True,
            )
        finally:
            if device is not None and model.device != home:
                try:
                    model.to(home)
                except Exception as exc:  # e.g. CUDA OOM moving back
                    print(f"Embedder: staying on {model.device}, could not return "
                          f"to {home}: {exc}")
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query -> [dim] float32 vector."""
        return self.embed([query])[0]
