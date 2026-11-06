"""Self-test: ingest survives a CUDA out-of-memory error while embedding.

A fake embedder raises OOM on the GPU path; the document must still end up in the
index (via the CPU retry). An embedder that fails outright must raise
IndexingError naming the document, not leave it silently unindexed.

Uses a temporary database and index, so it touches no real data.

Run:  python scripts/embed_oom_selftest.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
import torch

from core.embedder import EMBEDDING_DIM
from pipeline import IndexingError, Library

TEXT = "Ave maris stella Dei mater alma.\nQui tollis peccata mundi.\n15."


def _vecs(n):
    return np.random.default_rng(0).random((n, EMBEDDING_DIM), dtype="float32")


class OOMOnGPU:
    """Raises CUDA OOM unless asked to run on CPU."""
    def __init__(self, exc):
        self.exc, self.devices = exc, []

    def embed(self, texts, device=None, **_):
        self.devices.append(device)
        if device != "cpu":
            raise self.exc
        return _vecs(len(texts))


class AlwaysFails:
    def __init__(self, exc):
        self.exc, self.calls = exc, 0

    def embed(self, texts, **_):
        self.calls += 1
        raise self.exc


def indexed_ids(lib):
    return set(faiss.vector_to_array(lib.vectors.index.id_map).astype("int64").tolist())


def make_lib(tmp, embedder):
    return Library(db_path=os.path.join(tmp, "c.db"),
                   index_path=os.path.join(tmp, "i.faiss"), embedder=embedder)


def main():
    # 1. OOM (torch type and old RuntimeError message form) -> CPU retry -> indexed.
    for exc in (torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 20 MiB"),
                RuntimeError("CUDA error: out of memory")):
        with tempfile.TemporaryDirectory() as tmp:
            emb = OOMOnGPU(exc)
            lib = make_lib(tmp, emb)
            doc = lib.add_document(TEXT, "OOM test", verse=True)
            assert emb.devices == [None, "cpu"], emb.devices
            want = {s.id for s in doc.iter_segments() if s.latin_text != "15."}
            assert len(want) == 2
            assert indexed_ids(lib) == want, "doc must be indexed after CPU retry"
            lib.store.close()
            # The saved index on disk has it too.
            lib2 = make_lib(tmp, emb)
            assert indexed_ids(lib2) == want, "saved index must include the doc"
            lib2.store.close()

    # 2. Failure that survives the retry -> IndexingError naming the doc.
    #    3. A non-OOM error is not retried on CPU, but is reported the same way.
    for exc, calls in ((torch.cuda.OutOfMemoryError("CUDA out of memory"), 2),
                       (ValueError("tokenizer exploded"), 1)):
        with tempfile.TemporaryDirectory() as tmp:
            emb = AlwaysFails(exc)
            lib = make_lib(tmp, emb)
            try:
                lib.add_document(TEXT, "Fail test", verse=True)
            except IndexingError as e:
                msg = str(e)
            else:
                raise AssertionError("expected IndexingError")
            assert emb.calls == calls, (exc, emb.calls)
            docs = lib.store.list_documents()
            assert len(docs) == 1, "doc stays in the store"
            assert f"document {docs[0].id} " in msg and "NOT in" in msg, msg
            assert lib.vectors.size == 0
            lib.store.close()

    print("embed OOM selftest: OK")


if __name__ == "__main__":
    main()
