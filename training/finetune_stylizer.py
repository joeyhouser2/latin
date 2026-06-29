"""Fine-tune a small seq2seq into a Victorian-prose stylizer (Workstream C).

Monolingual English->English: learns to rewrite plain modern English into 19th-c.
translation register, from the (modern, victorian) pairs distilled by
scripts/build_victorian_corpus.py. This is a far easier task than the cross-lingual
verse fine-tune (which failed), and the targets are clean and consistent — so a
small model has a real chance here.

Default base flan-t5-base with a task prefix; the result loads via:
    Seq2SeqStylizer(model_name="models/stylizer-victorian", prefix="victorianize: ")

Run (use a free GPU):
    CUDA_VISIBLE_DEVICES=1 python training/finetune_stylizer.py \
        --data data/parallel/victorian_pairs.jsonl --out models/stylizer-victorian
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset


def load_pairs(patterns):
    pairs = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        if rec.get("src") and rec.get("tgt"):
                            pairs.append(rec)
    return pairs


class StyleDataset(Dataset):
    """Pre-tokenized (prefix+modern -> victorian) examples."""

    def __init__(self, pairs, tokenizer, prefix, max_length=256):
        srcs = [prefix + p["src"] for p in pairs]
        enc = tokenizer(srcs, text_target=[p["tgt"] for p in pairs],
                        truncation=True, max_length=max_length)
        self.examples = [
            {"input_ids": enc["input_ids"][i],
             "attention_mask": enc["attention_mask"][i],
             "labels": enc["labels"][i]}
            for i in range(len(pairs))
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/parallel/victorian_pairs.jsonl"],
                    help="JSONL path(s)/glob(s) of {src: modern, tgt: victorian}")
    ap.add_argument("--out", default="models/stylizer-victorian")
    ap.add_argument("--base", default="google/flan-t5-base")
    ap.add_argument("--prefix", default="victorianize: ", help="T5 task prefix")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="gradient accumulation steps (effective batch = batch-size * this)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--optim", default="adamw_torch_fused",
                    help="'adafactor' is far lighter on VRAM (no m/v states) — needed "
                         "to fit flan-t5-large on a 12GB GPU")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="trade compute for VRAM (needed for large on 12GB)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in --out (auto-detected)")
    ap.add_argument("--eval-gen", action="store_true",
                    help="run slow generation-based chrF eval each epoch (default: fast "
                         "loss-only eval; evaluate checkpoints with scripts/eval_stylizer.py)")
    args = ap.parse_args()

    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)

    pairs = load_pairs(args.data)
    if not pairs:
        raise SystemExit("No pairs found. Run scripts/build_victorian_corpus.py first.")
    import random
    random.Random(42).shuffle(pairs)
    n_eval = max(1, int(len(pairs) * args.eval_frac))
    eval_pairs, train_pairs = pairs[:n_eval], pairs[n_eval:]
    print(f"Loaded {len(pairs)} pairs. Train {len(train_pairs)} / Eval {len(eval_pairs)}.")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=6,               # keep recent epochs to pick the best register/fidelity tradeoff
        # Generation-based eval (chrF) costs ~8 min/epoch on flan-t5-large; we instead
        # evaluate saved checkpoints externally with scripts/eval_stylizer.py (richer
        # metrics). In-training eval stays loss-only (seconds) unless --eval-gen.
        predict_with_generate=args.eval_gen,
        generation_max_length=args.max_length,
        group_by_length=True,
        logging_steps=50,
        bf16=torch.cuda.is_available(),   # T5 is unstable in fp16 (NaN); bf16 is safe on Ada
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=StyleDataset(train_pairs, tokenizer, args.prefix, args.max_length),
        eval_dataset=StyleDataset(eval_pairs, tokenizer, args.prefix, args.max_length),
        data_collator=collator,
        compute_metrics=_build_metrics(tokenizer) if args.eval_gen else None,
    )
    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU — training will be slow on CPU.")

    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    # Guard against the truncated-root-save bug: verify the just-saved root model
    # reloads and matches the trainer's weights; if not, fall back to the last good
    # checkpoint's shard.
    _verify_or_repair_save(args.out, trainer)
    print(f"Saved Victorian stylizer to {args.out}")
    print(f'Use it:  Seq2SeqStylizer(model_name="{args.out}", prefix="{args.prefix}")')


def _verify_or_repair_save(out, trainer):
    """Load the root model.safetensors and sanity-check it; on failure copy the
    newest checkpoint's model.safetensors + config.json over the root."""
    import shutil
    from transformers import AutoModelForSeq2SeqLM
    try:
        AutoModelForSeq2SeqLM.from_pretrained(out)
        print("Root model verified (loads cleanly).")
        return
    except Exception as e:                       # truncated / corrupt root shard
        print(f"Root model failed to load ({e}); repairing from last checkpoint.")
    ckpts = sorted(glob.glob(os.path.join(out, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[1]))
    if not ckpts:
        raise SystemExit("No checkpoint to repair from.")
    last = ckpts[-1]
    for f in ("model.safetensors", "config.json", "generation_config.json"):
        src = os.path.join(last, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out, f))
    AutoModelForSeq2SeqLM.from_pretrained(out)    # re-verify
    print(f"Repaired root model from {last}.")


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
        dp = tokenizer.batch_decode(preds, skip_special_tokens=True)
        dl = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {"chrf": round(sacrebleu.corpus_chrf(dp, [dl]).score, 2)}

    return compute


if __name__ == "__main__":
    main()
