"""Optional post-translation style / verse layer.

A Stylizer takes the *literal* English crib produced by a Translator and rewrites
it into a target register — 19th-century scholarly prose, English blank verse,
rhymed couplets — optionally informed by the source text and its meter. It mirrors
the Translator interface: everything downstream depends only on the abstract
``Stylizer``, so the local-LLM backend here can be swapped for another (or an
API-backed one) without pipeline or schema churn.

Design notes
------------
* The stylized text is stored *alongside* the literal translation, never instead
  of it (see ``Segment.english_styled``). Style transfer can embellish; a reading
  tool must keep the faithful crib and let the reader toggle.
* Alignment is sacred. The reader renders source segment *i* next to English
  segment *i*, so a Stylizer must return exactly one styled string per input unit.
  ``stylize_units`` prefers a single coherent passage-level generation (better
  verse/prose flow) but *guarantees* count by falling back to per-unit rewriting
  when the model's line count doesn't match.
* Heavy deps (torch/transformers) are imported lazily, exactly like NLLBTranslator,
  so importing this module is cheap and the model only loads on first use.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class StyleUnit:
    """One thing to stylize: the literal English plus optional context for it."""

    literal: str                      # literal English crib (required)
    source: Optional[str] = None      # original Latin/Greek line/sentence
    scansion: Optional[str] = None    # metrical pattern, for verse presets


@dataclass
class StylePreset:
    """A named rewriting target: how to instruct the model and how to frame work."""

    name: str
    description: str
    system: str                       # persona / standing instructions
    # Per-unit instruction; {source}/{literal}/{scansion}/{meta} are filled in.
    instruction: str
    verse: bool = False               # affects passage framing + scansion hints


# ---------------------------------------------------------------------------
# Presets — the actual quality lever. Each keeps two hard rules: stay faithful to
# the literal meaning (no invented content), and return text only (no commentary).
# ---------------------------------------------------------------------------

_FAITHFUL = (
    "Stay strictly faithful to the meaning of the literal translation: do not add, "
    "drop, or invent content, names, or imagery. Output only the rewritten text "
    "with no preamble, notes, or quotation marks. Your output must be English "
    "prose -- never Latin, Greek, or any other language, even if the literal "
    "translation given to you looks incomplete, garbled, or hard to follow; do "
    "your best with what's there rather than reverting to the source language."
)

PRESETS = {
    "victorian_prose": StylePreset(
        name="victorian_prose",
        description="19th-century scholarly English prose (Loeb / Jowett register).",
        system=(
            "You are an English stylist rewriting plain modern translations of "
            "classical and medieval texts into the register of a fine 19th-century "
            "scholarly translation."
        ),
        instruction=(
            "Rewrite the following literal translation as a learned Victorian "
            "translator would: elevated but lucid diction, periodic sentences, "
            "measured cadence, lightly archaic vocabulary (sparing use of forms "
            "like 'thou' only where natural). " + _FAITHFUL + "\n\n"
            "{meta}Literal translation:\n{literal}"
        ),
    ),
    "verse_blank": StylePreset(
        name="verse_blank",
        description="English blank verse (unrhymed iambic pentameter).",
        verse=True,
        system=(
            "You are a verse translator rendering classical and medieval poetry "
            "into English blank verse — unrhymed iambic pentameter — in the high "
            "tradition of English poetic translation."
        ),
        instruction=(
            "Render the following literal prose translation as English blank verse: "
            "unrhymed iambic pentameter, one English line per source line where the "
            "sense allows, dignified poetic diction. " + _FAITHFUL + "\n\n"
            "{meta}Literal translation:\n{literal}"
        ),
    ),
    "verse_couplet": StylePreset(
        name="verse_couplet",
        description="Rhymed heroic couplets (Dryden / Pope register).",
        verse=True,
        system=(
            "You are a verse translator rendering classical poetry into rhymed "
            "English heroic couplets in the manner of Dryden and Pope."
        ),
        instruction=(
            "Render the following literal translation as rhymed heroic couplets: "
            "pairs of iambic-pentameter lines rhyming aa bb, polished and witty but "
            "faithful. " + _FAITHFUL + "\n\n"
            "{meta}Literal translation:\n{literal}"
        ),
    ),
    # Reverse direction, used to BUILD training data (not for the reader): turn
    # authentic 19th-c. translation prose into plain modern English, yielding
    # (modern, victorian) pairs to distill a standalone Victorian stylizer model.
    # Modernizing is a lower-hallucination LLM task than inventing archaism.
    "modernize": StylePreset(
        name="modernize",
        description="Plain contemporary English (for building (modern, victorian) pairs).",
        system=(
            "You rewrite ornate 19th-century English prose into plain, clear, "
            "contemporary English."
        ),
        instruction=(
            "Rewrite the following passage in the plainest possible modern English, "
            "as a contemporary news report or a children's encyclopedia would put it. "
            "Use only the most common everyday words. Break long periodic sentences "
            "into short, direct ones. Use normal subject-verb-object order — undo every "
            "inversion. Remove all rhetorical flourish, apostrophe ('O thou...'), and "
            "elevated diction. No archaic words (no thee/thou/hath/whilst/ere), no "
            "semicolons-as-clause-glue. Keep proper names. Preserve the exact meaning "
            "but make the wording as different from the original's ornate style as the "
            "meaning allows. " + _FAITHFUL + "\n\n"
            "{meta}Passage:\n{literal}"
        ),
    ),
}

DEFAULT_PRESET = "victorian_prose"

# Cheap function-word discriminator: when the "literal translation" fed to the
# stylizer is itself incoherent (garbled OCR source -> a bad NLLB translation),
# the LLM sometimes gives up on the English-rewriting task entirely and just
# paraphrases the source-language text quoted in its own prompt (see
# _meta_block) -- but keeps writing in *that* language instead of English.
# Found via a real case: ~5% of one document's styled output turned out to be
# fluent-looking Latin, not English, because the underlying literal
# translations were repetition-loop garbage the model had nothing to work
# with. Exact-word-boundary matching against a short, high-frequency function
# word list is enough to catch this cheaply without a real language-ID model.
_EN_FUNCTION_WORDS = frozenset(
    "the and of to is in that which be was were are it as with for on this by "
    "an a his her their they he she we you not have has had".split()
)
_LATIN_FUNCTION_WORDS = frozenset(
    "et in ad est qui quae quod sunt non cum sed ut si de ex per atque ac vel "
    "aut qua quam quum quia ita eis ei hic haec hoc his ille illa illud iam "
    "enim autem tamen nam quoque etiam esse sit sic suos suas suus sua eorum "
    "omnia omnes magis potius".split()
)
_WORD_RE = re.compile(r"[A-Za-z']+")


def looks_non_english(text: str, min_signal: int = 3) -> bool:
    """True if function-word evidence favors Latin over English. Requires at
    least `min_signal` total function-word hits before judging (too little
    text either way -> not flagged, since a short phrase can't be
    distinguished reliably and false alarms would just waste a retry)."""
    words = _WORD_RE.findall((text or "").lower())
    if not words:
        return False
    en = sum(1 for w in words if w in _EN_FUNCTION_WORDS)
    la = sum(1 for w in words if w in _LATIN_FUNCTION_WORDS)
    return (en + la) >= min_signal and la > en


def _meta_block(unit: StyleUnit, preset: StylePreset, context: Optional[dict]) -> str:
    """Optional context lines (source text, author/era, meter) prepended to the
    instruction so the model can hew closer to the original."""
    lines: List[str] = []
    ctx = context or {}
    if ctx.get("author") or ctx.get("era"):
        who = " · ".join(str(x) for x in (ctx.get("author"), ctx.get("era")) if x)
        lines.append(f"Source work: {who}.")
    if unit.source:
        lang = ctx.get("source_lang", "original")
        lines.append(f"{lang} text: {unit.source}")
    if preset.verse and unit.scansion:
        lines.append(f"Source metre/scansion: {unit.scansion}")
    return ("\n".join(lines) + "\n\n") if lines else ""


class Stylizer(ABC):
    """Rewrite literal English into a target register, preserving alignment."""

    @abstractmethod
    def _generate(self, system: str, user: str) -> str:
        """Backend-specific single completion. Returns model text only."""

    def stylize_units(
        self,
        units: Sequence[StyleUnit],
        preset: str = DEFAULT_PRESET,
        context: Optional[dict] = None,
    ) -> List[str]:
        """Stylize a passage, returning exactly ``len(units)`` strings (aligned).

        Tries one coherent passage-level generation first (numbered output, parsed
        back); on any count mismatch falls back to per-unit rewriting so alignment
        is never broken."""
        units = list(units)
        if not units:
            return []
        spec = PRESETS[preset]
        passage = self._try_passage(units, spec, context)
        results = (passage if passage is not None
                   else [self._stylize_one(u, spec, context) for u in units])

        # Guard against the LLM reverting to the source language when the
        # literal translation it was given was too garbled to work with (see
        # looks_non_english's docstring) -- retry just the affected unit(s)
        # individually with a firmer one-off reminder; if that still fails,
        # fall back to the literal translation itself. Imperfect English
        # prose beats confidently-wrong Latin silently occupying the styled
        # English column.
        for i, (unit, text) in enumerate(zip(units, results)):
            if not looks_non_english(text):
                continue
            retried = self._stylize_one(unit, spec, context, extra_reminder=True)
            results[i] = unit.literal.strip() if looks_non_english(retried) else retried
        return results

    def stylize(
        self, literal: str, *, source: Optional[str] = None,
        preset: str = DEFAULT_PRESET, context: Optional[dict] = None,
    ) -> str:
        """Stylize a single crib (convenience over ``stylize_units``)."""
        return self.stylize_units(
            [StyleUnit(literal=literal, source=source)], preset, context
        )[0]

    # -- internals -----------------------------------------------------------

    def _stylize_one(
        self, unit: StyleUnit, spec: StylePreset, context: Optional[dict],
        extra_reminder: bool = False,
    ) -> str:
        if not unit.literal or not unit.literal.strip():
            return ""
        user = spec.instruction.format(
            literal=unit.literal.strip(),
            meta=_meta_block(unit, spec, context),
        )
        if extra_reminder:
            user += ("\n\nReminder: respond in English. Do not quote or continue in "
                     "Latin, Greek, or any other source language.")
        return _clean(self._generate(spec.system, user))

    def _try_passage(
        self, units: Sequence[StyleUnit], spec: StylePreset, context: Optional[dict]
    ) -> Optional[List[str]]:
        """Stylize all units in one shot with numbered I/O; return None on mismatch.

        A single unit needs no numbering — go straight to per-unit (which the
        caller's fallback handles), so only multi-unit passages take this path."""
        if len(units) < 2:
            return None
        numbered = "\n".join(
            f"{i + 1}. {u.literal.strip()}" for i, u in enumerate(units)
        )
        # Reuse the preset's instruction but swap in the numbered list + a strict
        # one-line-per-number contract.
        head = spec.instruction.split("{meta}")[0].rstrip()
        meta = _meta_block(units[0], spec, context) if not _per_unit_sources(units) else ""
        user = (
            f"{head}\n\n{meta}"
            "Below are numbered literal lines. Return the rewritten version of each, "
            "keeping the SAME numbering and exactly one entry per number "
            f"(1..{len(units)}), in order:\n\n{numbered}"
        )
        parsed = _parse_numbered(self._generate(spec.system, user), len(units))
        return parsed

    # Allow concrete backends to release GPU memory etc.
    def close(self) -> None:  # pragma: no cover - optional
        pass


def _per_unit_sources(units: Sequence[StyleUnit]) -> bool:
    return any(u.source for u in units)


def _clean(text: str) -> str:
    text = (text or "").strip()
    # Strip wrapping quotes the model sometimes adds.
    if len(text) >= 2 and text[0] in "\"'“" and text[-1] in "\"'”":
        text = text[1:-1].strip()
    return text


_NUM_LINE = re.compile(r"^\s*(\d+)[.)\]]\s+(.*)$")


def _parse_numbered(text: str, n: int) -> Optional[List[str]]:
    """Parse '1. ... 2. ...' back into n strings; None if the count doesn't match.

    Lines without a leading number are appended to the current entry (so a styled
    line may itself span multiple physical lines, e.g. a couplet)."""
    out: dict = {}
    current = None
    for raw in (text or "").splitlines():
        m = _NUM_LINE.match(raw)
        if m:
            current = int(m.group(1))
            out[current] = m.group(2).strip()
        elif current is not None and raw.strip():
            out[current] = (out[current] + "\n" + raw.strip()).strip()
    if sorted(out) != list(range(1, n + 1)):
        return None
    return [_clean(out[i]) for i in range(1, n + 1)]


class LocalLLMStylizer(Stylizer):
    """Local open-source instruct model (transformers ``AutoModelForCausalLM``).

    Uses the model's chat template, GPU if available. Lazy-loaded. Default is a
    capable mid-size instruct model; override ``model_name`` for something lighter
    (e.g. a 3B) or heavier per your VRAM."""

    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading stylizer model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype
        ).to(self._device)

    def _generate(self, system: str, user: str) -> str:
        import torch

        self._ensure_loaded()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        # Decode only the newly generated continuation.
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def close(self) -> None:  # pragma: no cover - optional
        self._model = None
        self._tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class Seq2SeqStylizer(Stylizer):
    """A trained, monolingual English->English style model (T5/BART-family).

    Workstream C: a small fine-tuned seq2seq that rewrites a literal English crib
    into Victorian register directly, with no prompt and no big LLM at inference
    (fast, offline). Trained on (modern, victorian) pairs distilled by the LLM via
    the 'modernize' preset (see scripts/build_victorian_corpus.py).

    The style is baked into the weights, so presets/context are ignored and each
    unit is rewritten independently (alignment is trivially preserved). Heavy deps
    are imported lazily, like the other backends."""

    def __init__(self, model_name: str, max_length: int = 256, num_beams: int = 4,
                 prefix: str = ""):
        self.model_name = model_name
        self.max_length = max_length
        self.num_beams = num_beams
        self.prefix = prefix          # e.g. "victorianize: " for a T5 task prefix
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        print(f"Loading stylizer seq2seq model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def stylize_units(self, units, preset=DEFAULT_PRESET, context=None):
        """Rewrite each unit's literal English through the trained model (batched).
        preset/context are ignored — the register is in the weights."""
        units = list(units)
        if not units:
            return []
        return self._run([u.literal for u in units])

    def _run(self, texts, batch_size: int = 16):
        import torch
        self._ensure_loaded()
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [self.prefix + (t or "") for t in texts[i:i + batch_size]]
            inputs = self._tokenizer(batch, return_tensors="pt", padding=True,
                                     truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                gen = self._model.generate(**inputs, max_length=self.max_length,
                                           num_beams=self.num_beams, early_stopping=True)
            out.extend(_clean(s) for s in
                       self._tokenizer.batch_decode(gen, skip_special_tokens=True))
        return out

    def _generate(self, system: str, user: str) -> str:  # ABC contract; single unit
        return self._run([user])[0]
