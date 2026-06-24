"""End-to-end smoke test of the vertical slice on a tiny inline text:
ingest -> segment -> embed -> translate -> search -> read.
Run: python scripts/smoke_e2e.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use a throwaway data dir so we don't touch the real corpus.
for p in ("data/_smoke.db", "data/_smoke.faiss"):
    if os.path.exists(p):
        os.remove(p)

from pipeline import Library

lib = Library(db_path="data/_smoke.db", index_path="data/_smoke.faiss")

text = (
    "Quid est ergo tempus? Si nemo ex me quaerat, scio; si quaerenti explicare "
    "velim, nescio. In principio creavit Deus caelum et terram. Terra autem erat "
    "inanis et vacua."
)
doc = lib.add_document(
    text, title="Confessiones (excerpt)", author="Augustinus",
    century=4, genre="philosophy", language_stage="late_antique",
    source="smoke-test", has_existing_translation=False,
)
print(f"INGESTED doc {doc.id}, {len(list(doc.iter_segments()))} segments")

n = lib.translate_document(doc.id)
print(f"TRANSLATED {n} segments")

print("\n--- READER VIEW (side by side) ---")
for seg in lib.get_document(doc.id).iter_segments():
    print(f"[LA] {seg.latin_text}")
    print(f"[EN] {seg.english_text}\n")

print("--- SEARCH 'what is time' ---")
for hit in lib.search("what is time", k=2):
    print(f"  {hit.score:.3f} | {hit.document.author}: {hit.segment.latin_text[:60]}")

print("\n--- SEARCH (English-query cross-lingual) 'creation of the world' ---")
for hit in lib.search("creation of the world", k=2):
    print(f"  {hit.score:.3f} | {hit.segment.latin_text[:60]}")

lib.close()
for p in ("data/_smoke.db", "data/_smoke.faiss"):
    if os.path.exists(p):
        os.remove(p)
print("\nOK")
