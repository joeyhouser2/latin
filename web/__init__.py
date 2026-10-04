"""Web layer: a FastAPI browser/reader over the library plus a job queue.

Talks to ``pipeline.Library`` / ``core.store.Store`` for reading and to
``web.jobs`` for anything long-running. Nothing here loads a model: translation
and stylization run as separate subprocesses so a CUDA OOM kills a worker, not
the site.
"""
