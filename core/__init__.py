"""Core package for the Latin reading + discovery library.

Modules:
    models      - Document / Section / Segment data structures
    store       - SQLite persistence
    embedder    - multilingual sentence embeddings
    vectorstore - persisted FAISS index keyed by segment id
    segmenter   - Latin sentence segmentation (CLTK)
    translator  - pluggable Latin->English translation (NLLB default)
    search      - semantic + metadata search over the corpus
"""

# Load-order guard (Windows). If Python's ssl module is loaded -- by any HTTPS
# request -- before pyarrow, then pyarrow's later import (pulled in by
# sentence-transformers -> transformers -> pandas the first time the embedder
# loads) dies with an access violation that takes the whole process down: no
# exception, just a segfault. Bisected 2026-09-22 (HTTPS then embed crashes;
# XML parsing, threads, faiss-first alone do not), and importing pyarrow (or
# torch) first prevents it. Every entry point imports `core` before touching
# the network, so doing it here fixes the order everywhere at once: the ingest
# CLI (which fetches, *then* embeds, for every network connector), job
# subprocesses, and the web server (which fetches catalogues over HTTPS and
# loads the embedder lazily on the first search). Costs ~0.2s at import.
try:
    import pyarrow  # noqa: F401  (imported for its side effect: DLL load order)
except ImportError:
    pass
