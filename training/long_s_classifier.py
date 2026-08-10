"""Train a character-context classifier that decides, for a given 'f' in a
word, whether it's a genuine f or a misread long-s (ſ) that should become s.

This complements ingest.ocr_fix's whole-word dictionary approach. The
dictionary approach requires the *entire corrected word* to be independently
attested elsewhere in the corpus, which fails for rare-but-legitimate
inflected forms (e.g. "usurarum", the central term of the Salmasius
treatise). A context classifier instead asks a *local* question -- does the
handful of characters around this f look like a long-s environment? -- which
can also cover forms the dictionary never sees.

Training data is entirely synthetic, generated from clean, non-OCR corpus
text (every document except doc 376 and anything else flagged by
scripts/check_ocr_corruption.py):

  * genuine-f examples (label 0): every 'f' that's already in a real,
    unmodified word (facere, figulus, fuit, ...).
  * long-s examples (label 1): every non-final 's' in a real word, flipped to
    'f' (the actual historical corruption -- long s was used for every s
    except the word-final one).
  * adversarial ct->ft examples (label 0): every 'c' immediately before a
    't', flipped to 'f' (contractus -> contraftus). This document has BOTH
    error classes producing the same output letter 'f' from two different
    source letters, and a model trained only on the long-s pattern would
    learn to treat "f before t" as unconditional evidence for s -- exactly
    backwards for words like "contraftus" (really contractus) or "diftum"
    (really dictum). Without this negative class the model cannot tell the
    two apart, since "genuine f before t" essentially never occurs in clean
    Latin on its own.

Usage:
    python training/long_s_classifier.py --train
    python training/long_s_classifier.py --eval-only   # reuse saved model
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.store import Store
from ingest.ocr_fix import build_reference_vocab

MODEL_PATH = "models/long_s_classifier.joblib"
WINDOW = 5  # characters of context on each side


def _windows_for_word(word: str, positions_labels):
    """Yield (context_string, label) for each (position, label) in a word."""
    padded = ("^" * WINDOW) + word + ("$" * WINDOW)
    for pos, label in positions_labels:
        p = pos + WINDOW
        left = padded[p - WINDOW: p]
        right = padded[p + 1: p + 1 + WINDOW]
        yield f"{left}@{right}", label


def generate_examples(words):
    """words: iterable of lowercase alphabetic word types (deduped).

    Returns list of (context_str, label, source_word, kind) for inspection.
    """
    examples = []
    for w in words:
        if len(w) < 2:
            continue

        # genuine-f: every native f, word untouched.
        f_positions = [(i, 0) for i, c in enumerate(w) if c == "f"]
        if f_positions:
            for ctx, label in _windows_for_word(w, f_positions):
                examples.append((ctx, label, w, "genuine_f"))

        # long-s: flip every non-final s -> f in one rendering of the word.
        s_idx = [i for i, c in enumerate(w) if c == "s" and i != len(w) - 1]
        if s_idx:
            corrupted = list(w)
            for i in s_idx:
                corrupted[i] = "f"
            corrupted = "".join(corrupted)
            for ctx, label in _windows_for_word(corrupted, [(i, 1) for i in s_idx]):
                examples.append((ctx, label, w, "long_s"))

        # adversarial ct->ft: flip every c-before-t -> f.
        ct_idx = [i for i in range(len(w) - 1) if w[i] == "c" and w[i + 1] == "t"]
        if ct_idx:
            corrupted = list(w)
            for i in ct_idx:
                corrupted[i] = "f"
            corrupted = "".join(corrupted)
            for ctx, label in _windows_for_word(corrupted, [(i, 0) for i in ct_idx]):
                examples.append((ctx, label, w, "ct_adversarial"))

    return examples


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("--exclude-doc-ids", type=int, nargs="*", default=[376])
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    store = Store(args.db)
    print("Building word-type vocabulary (excluding", args.exclude_doc_ids, ")...")
    vocab = build_reference_vocab(store, exclude_doc_ids=args.exclude_doc_ids)
    words = [w for w in vocab.keys() if w.isalpha()]
    print(f"  {len(words):,} distinct word types")

    random.seed(args.seed)
    random.shuffle(words)
    n_test = int(len(words) * args.test_frac)
    test_words, train_words = words[:n_test], words[n_test:]
    print(f"  split: {len(train_words):,} train words / {len(test_words):,} test words")

    train_ex = generate_examples(train_words)
    test_ex = generate_examples(test_words)
    print(f"  train examples: {len(train_ex):,}   test examples: {len(test_ex):,}")

    from collections import Counter
    print("  train label/kind breakdown:", Counter((l, k) for _, l, _, k in train_ex))
    print("  test  label/kind breakdown:", Counter((l, k) for _, l, _, k in test_ex))

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    import joblib

    Xtr = [ex[0] for ex in train_ex]
    ytr = [ex[1] for ex in train_ex]
    Xte = [ex[0] for ex in test_ex]
    yte = [ex[1] for ex in test_ex]

    pipe = Pipeline([
        ("vec", CountVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")),
    ])
    print("\nTraining...")
    pipe.fit(Xtr, ytr)

    print("\n=== Overall test set ===")
    pred = pipe.predict(Xte)
    print(classification_report(yte, pred, target_names=["not-s (0)", "was-s (1)"]))

    print("=== Per-kind breakdown (test set) ===")
    kinds = sorted(set(ex[3] for ex in test_ex))
    for kind in kinds:
        idx = [j for j, ex in enumerate(test_ex) if ex[3] == kind]
        yk = [yte[j] for j in idx]
        pk = [pred[j] for j in idx]
        acc = sum(1 for a, b in zip(yk, pk) if a == b) / len(idx)
        print(f"  {kind:16} n={len(idx):>7,}  accuracy={acc:.3%}  "
              f"(true label is always {yk[0]})")

    os.makedirs("models", exist_ok=True)
    joblib.dump({"pipeline": pipe, "window": WINDOW}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
