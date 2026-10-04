"""Run the latin repo's long-s repair over free text, under its own venv.

    latinvenv/Scripts/python long_s_repair.py <raw.txt> <out.txt>

The repo's own scripts drive this by corpus --doc-id; the modules underneath
take plain text, which is what a freshly OCR'd page is. Two tiers, in the order
the repo applies them: the dictionary tier resolves what the corpus vocabulary
can confirm, and the trained classifier takes what is left.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from core.store import Store
from ingest.ocr_fix import build_reference_vocab, build_prefix_index, correct_long_s
from ingest.ocr_fix_model import load_model, apply_model_tier, MODEL_PATH

raw_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
raw = raw_path.read_text(encoding="utf-8")

store = Store("data/corpus.db")
vocab = build_reference_vocab(store)
prefixes = build_prefix_index(vocab)
print(f"reference vocabulary: {len(vocab):,} forms")

dict_pass = correct_long_s(raw, vocab, prefixes)
text = dict_pass.text if hasattr(dict_pass, "text") else str(dict_pass)
print(f"dictionary tier: {getattr(dict_pass, 'n_changed', '?')} token(s) changed")

try:
    pipeline = load_model(MODEL_PATH)
    model_pass = apply_model_tier(text, pipeline, vocab=vocab, vocab_prefixes=prefixes)
    text = model_pass.text if hasattr(model_pass, "text") else text
    print(f"model tier: {getattr(model_pass, 'n_changed', '?')} token(s) changed")
except Exception as exc:  # noqa: BLE001 - the dictionary tier alone is still useful
    print(f"model tier skipped: {type(exc).__name__}: {exc}")

out_path.write_text(text, encoding="utf-8")
print(f"wrote {len(text):,} chars -> {out_path}")
