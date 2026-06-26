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
                 src_lang: str = None, tgt_lang: str = None, preprocess=None):
        self.model_name = model_name
        self.src_lang = src_lang or self.SRC_LANG
        self.tgt_lang = tgt_lang or self.TGT_LANG
        # Optional source-text normalizer (e.g. strip_greek_diacritics); must
        # match whatever normalization the model was trained with.
        self.preprocess = preprocess
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
        out: List[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tgt_id,
                    max_length=512, num_beams=4, early_stopping=True,
                )
            out.extend(self._tokenizer.batch_decode(generated, skip_special_tokens=True))
        return out
