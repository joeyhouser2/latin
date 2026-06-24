"""Rebuild the FAISS index from the store, backfilling markup-stripped embed_text.

Run after changing the embedding or normalization logic, or after ingesting data
created by an older version that lacked embed_text.

Run:  python scripts/reindex.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import Library

if __name__ == "__main__":
    lib = Library()
    n = lib.reindex()
    lib.close()
    print(f"Done. {n} segments indexed.")
