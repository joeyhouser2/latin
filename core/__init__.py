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
