"""Pluggable Latin -> English translation.

Everything downstream depends only on the Translator interface, so swapping NLLB
for an LLM-backed or fine-tuned translator later is a one-class change with no
schema or pipeline churn. NLLB is the free, local default; note it is trained
mostly on ecclesiastical/modern Latin and is weaker on classical/medieval Latin.
"""

from __future__ import annotations

from typing import List
from abc import ABC, abstractmethod


class Translator(ABC):
    """Translate Latin text to English."""

    @abstractmethod
    def translate(self, latin_text: str) -> str:
        ...

    def translate_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Default: translate one at a time. Backends may override for speed."""
        return [self.translate(t) for t in texts]


class NLLBTranslator(Translator):
    """Meta's NLLB-200. Loaded lazily; uses GPU if available."""

    SRC_LANG = "lat_Latn"   # NLLB source code: lat_Latn (Latin) or ell_Grek (Greek)
    TGT_LANG = "eng_Latn"

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M",
                 src_lang: str = None, tgt_lang: str = None, preprocess=None,
                 max_length: int = 512, nllb_tokenizer: bool = False):
        self.model_name = model_name
        self.src_lang = src_lang or self.SRC_LANG
        self.tgt_lang = tgt_lang or self.TGT_LANG
        self.max_length = max_length      # caps tokenization + generation length
        # Optional source-text normalizer (e.g. strip_greek_diacritics); must
        # match whatever normalization the model was trained with.
        self.preprocess = preprocess
        # transformers 5.0's AutoTokenizer loads NLLB checkpoints as a generic
        # TokenizersBackend that silently ignores src_lang: no source-language
        # token is emitted at all, so the model has to guess the input
        # language (for German it often just copies the input back).
        # nllb_tokenizer=True loads NllbTokenizerFast directly, which emits the
        # proper src_lang prefix. Opt-in so existing Latin/Greek translation
        # (and the fine-tuned models trained under the old behavior) is
        # unchanged; lat_Latn isn't an NLLB-200 language code anyway.
        # Measured on held-out Greek (scripts/eval_translation.py, 1332 pairs
        # unseen by all of them): the tag makes every fine-tuned Greek model
        # WORSE -- v1 chrF 25.3 -> 23.2, v2 27.9 -> 26.5, v3 28.8 -> 27.1
        # (stock 18.7 -> 18.3, a wash) -- because training/finetune.py used
        # AutoTokenizer too, so they all trained on untagged input. Don't turn
        # this on for grc without retraining the models with the tag.
        self.nllb_tokenizer = nllb_tokenizer
        self._tokenizer = None
        self._model = None
        self._device = None
        self._tgt_id = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"Loading translation model: {self.model_name}")
        if self.nllb_tokenizer:
            from transformers import NllbTokenizerFast
            self._tokenizer = NllbTokenizerFast.from_pretrained(self.model_name, src_lang=self.src_lang)
        else:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang=self.src_lang)
            except AttributeError:
                # transformers 5.0.0's AutoTokenizer cannot resolve the NLLB/m2m_100
                # tokenizer class (saved as "TokenizersBackend") and raises
                # AttributeError deep in auto-resolution — for the stock model too.
                # NLLB-derived models share the NLLB tokenizer, so load it directly.
                from transformers import NllbTokenizerFast
                self._tokenizer = NllbTokenizerFast.from_pretrained(self.model_name, src_lang=self.src_lang)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._tgt_id = self._tokenizer.convert_tokens_to_ids(self.tgt_lang)

    def translate(self, latin_text: str) -> str:
        if not latin_text or not latin_text.strip():
            return ""
        return self.translate_batch([latin_text], batch_size=1)[0]

    def translate_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        import torch

        self._ensure_loaded()
        if self.preprocess:
            texts = [self.preprocess(t) for t in texts]

        # Length-aware batching: group similar-length texts so one long segment
        # doesn't pad a whole batch to its size (the main VRAM waste). Results are
        # mapped back to the original order before returning.
        order = sorted(range(len(texts)), key=lambda j: len(texts[j]))
        results: List[str] = [""] * len(texts)
        i = 0
        bs = max(1, batch_size)
        while i < len(order):
            idx = order[i:i + bs]
            batch = [texts[j] for j in idx]
            try:
                eng = self._generate(batch, torch)
                for j, e in zip(idx, eng):
                    results[j] = e
                i += len(idx)
                if bs < batch_size:        # recovered — grow back toward target
                    bs = min(batch_size, bs * 2)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                if bs == 1:                # a single segment won't fit even capped
                    results[idx[0]] = ""    # leave untranslated; resume can retry
                    i += 1
                else:
                    bs = max(1, bs // 2)    # back off and retry this slice
        return results

    def _generate(self, batch: List[str], torch) -> List[str]:
        inputs = self._tokenizer(batch, return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = self._model.generate(
                **inputs, forced_bos_token_id=self._tgt_id,
                max_length=self.max_length, num_beams=4, early_stopping=True,
            )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)
