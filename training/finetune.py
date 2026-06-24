"""Fine-tune an open-source NLLB-200 model on mined parallel pairs.

Reads the JSONL produced by scripts/mine_parallel.py and fine-tunes
facebook/nllb-200-distilled-600M (Latin lat_Latn / Greek ell_Grek -> English
eng_Latn). The result loads as a drop-in Translator via:
    NLLBTranslator(model_name="models/<your-output-dir>")

Runs on CPU but is slow — a GPU is strongly recommended. Optional `sacrebleu`
enables a chrF eval metric (pip install sacrebleu).

Run:
    python training/finetune.py --data data/parallel/*.jsonl --out models/nllb-latin-classical
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Dict

# Make the repo root importable so `core.*` resolves when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset


SRC_LANG_BY_CODE = {"la": "lat_Latn", "grc": "ell_Grek"}  # ell_Grek = Greek script proxy
TGT_LANG = "eng_Latn"


def load_pairs(patterns: List[str]) -> List[Dict]:
    pairs = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        pairs.append(json.loads(line))
    return pairs


class PairDataset(Dataset):
    """Pre-tokenizes all pairs once, batched per source language.

    Tokenizing in __getitem__ (and resetting NLLB's src_lang per item) rebuilds
    the tokenizer's processor on every step and starves the GPU. Doing it once,
    grouped by language, removes that bottleneck entirely.
    """

    def __init__(self, pairs, tokenizer, max_length=256, normalize_grc=False):
        from collections import defaultdict
        by_lang = defaultdict(list)
        for p in pairs:
            by_lang[p.get("src_lang", "la")].append(p)

        self.examples = []
        for lang, group in by_lang.items():
            tokenizer.src_lang = SRC_LANG_BY_CODE.get(lang, "lat_Latn")
            srcs = [p["src"] for p in group]
            if normalize_grc and lang == "grc":
                from core.normalize import strip_greek_diacritics
                srcs = [strip_greek_diacritics(s) for s in srcs]
            enc = tokenizer(
                srcs,
                text_target=[p["tgt"] for p in group],
                truncation=True, max_length=max_length,
            )
            for i in range(len(group)):
                self.examples.append({
                    "input_ids": enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                    "labels": enc["labels"][i],
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="JSONL path(s)/glob(s)")
    ap.add_argument("--out", required=True, help="output model directory")
    ap.add_argument("--base", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--optim", default="adamw_torch",
                    help="optimizer; 'adafactor' frees ~5-7GB vs Adam, enabling a much larger batch")
    ap.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="cap the corpus to this many (shuffled) pairs; 0 = use all")
    ap.add_argument("--normalize-grc", action="store_true",
                    help="strip polytonic diacritics from Greek sources (ancient Greek)")
    args = ap.parse_args()

    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)

    pairs = load_pairs(args.data)
    if not pairs:
        raise SystemExit("No pairs found. Run scripts/mine_parallel.py first.")
    # Fixed-seed shuffle so the eval split is representative (the files are
    # concatenated, so without this the eval set would be one author).
    import random
    random.Random(42).shuffle(pairs)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    print(f"Loaded {len(pairs)} parallel pairs.")

    n_eval = max(1, int(len(pairs) * args.eval_frac))
    eval_pairs, train_pairs = pairs[:n_eval], pairs[n_eval:]
    print(f"Train: {len(train_pairs)}  Eval: {len(eval_pairs)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    metrics_fn = _build_metrics(tokenizer)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim=args.optim,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=args.max_length,
        group_by_length=True,          # batch similar lengths -> far less padding waste
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=PairDataset(train_pairs, tokenizer, args.max_length, args.normalize_grc),
        eval_dataset=PairDataset(eval_pairs, tokenizer, args.max_length, args.normalize_grc),
        data_collator=collator,
        compute_metrics=metrics_fn,
    )
    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU detected — training will be very slow on CPU.")

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved fine-tuned model to {args.out}")
    print(f'Use it:  NLLBTranslator(model_name="{args.out}")')


def _build_metrics(tokenizer):
    try:
        import sacrebleu
    except ImportError:
        print("(sacrebleu not installed; skipping chrF metric)")
        return None

    import numpy as np

    def compute(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dec_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        chrf = sacrebleu.corpus_chrf(dec_preds, [dec_labels]).score
        return {"chrf": round(chrf, 2)}

    return compute


if __name__ == "__main__":
    main()
