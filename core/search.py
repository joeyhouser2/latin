"""Semantic search over the corpus, returning segment-level hits.

Each hit carries enough context to deep-link into the Reader: the segment, its
score, and its parent document. Metadata filtering (century, genre, whether an
English translation already exists) is layered on top of the vector search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .models import Segment, Document
from .store import Store
from .embedder import Embedder
from .vectorstore import VectorStore


@dataclass
class SearchHit:
    segment: Segment
    score: float
    document: Document


class SearchService:
    def __init__(self, store: Store, embedder: Embedder, vectors: VectorStore):
        self.store = store
        self.embedder = embedder
        self.vectors = vectors

    def search(
        self,
        query: str,
        k: int = 5,
        language_stage: Optional[str] = None,
        only_untranslated_works: bool = False,
    ) -> List[SearchHit]:
        if not query.strip() or self.vectors.size == 0:
            return []

        # Over-fetch so metadata filtering still leaves ~k results.
        raw = self.vectors.search(self.embedder.embed_query(query), k=k * 4)

        hits: List[SearchHit] = []
        doc_cache: dict = {}
        for segment_id, score in raw:
            segment = self.store.get_segment(segment_id)
            if segment is None:
                continue
            doc_id = self.store.document_id_for_segment(segment_id)
            if doc_id is None:
                continue
            if doc_id not in doc_cache:
                doc_cache[doc_id] = self.store.get_document(doc_id)
            doc = doc_cache[doc_id]
            if doc is None:
                continue

            if language_stage and doc.language_stage != language_stage:
                continue
            if only_untranslated_works and doc.has_existing_translation:
                continue

            hits.append(SearchHit(segment=segment, score=score, document=doc))
            if len(hits) >= k:
                break
        return hits
