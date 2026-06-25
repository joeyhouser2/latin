# Roadmap: verse-translation models, Homeric Greek, and a standalone Victorian stylizer

Future-work plan agreed 2026-06-23. Three connected workstreams. All reuse the
existing mining/fine-tune/align infrastructure (`training/parallel.py`,
`training/aligner.py`, `training/finetune.py`) and plug into the existing
`Translator` / `Stylizer` interfaces — no architectural change.

A crucial distinction runs through all of this: **fidelity vs. register.** A
literal translator and a *poetic* renderer want opposite things from their training
data. Verse translations (Dryden, Pope) are gorgeous but loose — they reorder,
expand, and embellish to serve metre and rhyme. That makes them:
- **bad** training data for a *faithful* translator (the model would learn to
  paraphrase and invent), but
- **good** training data for a *poetic/stylistic* layer (which is allowed to be
  interpretive, and is stored alongside the literal crib, never replacing it).

So the literal translators stay trained on literal pairs; verse pairs train the
poetic layer. Keep them separate.

---

## Workstream A — Verse-translation model (Dryden / Pope type)

**Goal:** a model that renders Latin/Greek into English *verse-register* output —
the poetic backend for the `Stylizer` (or a clearly-marked interpretive
Translator), distinct from the literal crib.

**Data:** public-domain verse translations on Project Gutenberg, aligned against
the Perseus source. Unlike the prose miner, these editions do **not** share a CTS
citation scheme with the source, so we use the comparable-text aligner
(`training/aligner.py`), not `PerseusParallelMiner`.

Candidate pairs are in [`training/verse_targets.py`](../training/verse_targets.py).
Core set: Dryden's *Aeneid*, Pope's *Iliad* and *Odyssey*, Garth/Dryden *Ovid*.

**Alignment approach (the hard part):**
1. Fetch source (Perseus CTS) and the English verse text (Gutenberg).
2. Prefer **book-level anchoring**: both editions preserve book boundaries
   ("BOOK I" … in the English; book `div`s in the source). Align book-by-book to
   shrink the DP and stop cross-book leakage. (v0 of `scripts/mine_verse.py` does
   whole-text; per-book is the first tuning step.)
3. Within a book, run the monotonic-DP aligner. **Translate-then-align (`--via-mt`)
   is recommended**: render the source to rough English with the existing routed
   translator, align English↔English (robust, no LaBSE download), but emit the
   ORIGINAL source ↔ gold verse. Kept pairs are never MT output.
4. Keep a **conservative `threshold`** and spot-check. Expect **low yield** — verse
   is paraphrastic, so many beads fall below threshold. That is correct: we want
   clean poetic exemplars, not coverage.

**Output:** JSONL `{src, tgt, citation, src_lang, era, style: "verse", source}` in
`data/parallel/` (gitignored). Add a `style` field (default `"literal"`).

**Training:** `training/finetune.py` on verse JSONL → `models/nllb-latin-verse` /
`models/nllb-greek-verse`. Honest caveat: a 600M NLLB trained on loose verse may
produce mushy output; the larger lever is still a prompted LLM `Stylizer` with the
verse presets. Treat the fine-tune as an experiment, and compare against the
LLM-stylizer baseline before committing.

---

## Workstream B — Homeric / archaic Greek (a gap we never trained on)

Our Greek models cover Attic/classical (`perseus_greek`) and patristic/late-antique
(`first1k`). **None saw Homeric (archaic, ~8th c. BCE) Greek**, whose dialect and
morphology differ sharply (e.g. `-οιο` genitives, tmesis, Ionic forms). Homer,
Hesiod, the Homeric Hymns, and Apollonius all live in this register.

**Key win — this is mostly free with existing tooling:** Perseus's `canonical-greekLit`
ships **Murray's prose English `-eng` editions for the Iliad and Odyssey**, sharing
the CTS scheme. So faithful Homeric pairs come straight from the *existing*
`PerseusParallelMiner`:

```
python scripts/mine_parallel.py tlg0012.tlg001 --greek --era archaic \
    --out data/parallel/homer_iliad.jsonl
python scripts/mine_parallel.py tlg0012.tlg002 --greek --era archaic \
    --out data/parallel/homer_odyssey.jsonl
```

Then fine-tune a Homeric model (normalize diacritics, same as other Greek models):

```
python training/finetune.py --data data/parallel/homer_*.jsonl \
    --out models/nllb-greek-archaic --normalize-grc --epochs 3
```

**Plumbing added now (inert until the model exists):**
- `"archaic"` added to `LANGUAGE_STAGES` (`core/models.py`).
- `("grc", "archaic") -> models/nllb-greek-archaic` route in
  `pipeline.TRANSLATOR_MODELS` (falls back to the general Greek model until trained).

`strip_greek_diacritics` already handles polytonic Homeric text unchanged — the
dialect difference is lexical/morphological and is learned from the data, not
preprocessed. The **same Homer source** then doubles as the Greek side of the
Workstream-A verse pairs (Pope's Iliad/Odyssey), so mine it once, use it twice.

---

## Workstream C — Standalone Victorian-prose stylizer model

**Goal:** a *monolingual* English→English editor that lifts **any** literal
translation into Victorian register, independent of source language — a trained,
local `Stylizer` backend that doesn't depend on a big prompted LLM at inference.

Today's `victorian_prose` works because our fine-tune *targets* (Loeb/ANF/NPNF,
1880s–1900s) are already Victorian. But that only styles *our* translations. A
broad stylizer should lift arbitrary English.

**Data problem:** there is no parallel (plain ↔ Victorian) corpus. Bootstrap it.
Two routes, blendable:
- **Distill from our LLM** — run the prompted `LocalLLMStylizer(victorian_prose)`
  over a large set of plain modern English → synthetic (plain, victorian) pairs;
  train plain→victorian. Cheap, fully open-source, but inherits the LLM's idea of
  "Victorian."
- **Mine + modernize (recommended)** — collect *real* public-domain Victorian prose
  (Gutenberg: Gibbon, Macaulay, Jowett, Loeb-era translations), use the LLM to
  *modernize* each passage → (modern, victorian) pairs, train the **victorian
  direction**. Real Victorian targets give authentic register; modernizing is the
  easier, lower-hallucination LLM direction than inventing archaism. The
  verse-English from Workstream A is itself good Victorian-register material.

**Train:** a small seq2seq (BART/T5) or a small causal model on the (plain →
victorian) pairs; wrap as a `Stylizer` subclass (`Seq2SeqStylizer`) alongside
`LocalLLMStylizer`, selectable behind the same interface. `scripts/build_victorian_corpus.py`
(to write) generates the pairs; `training/finetune.py` (or a small variant) trains.

**Why separate from the translator:** it composes with *every* translator and every
source language, and it's swappable/testable in isolation — same reasoning as
keeping `Stylizer` separate from `Translator` in the first place.

---

## Suggested order

1. **B first** — highest value, lowest risk; reuses existing tooling end-to-end and
   fills a real coverage gap. Validates Homeric mining + gives the Greek source for A.
2. **A next** — build/tune `scripts/mine_verse.py` (book anchoring, threshold
   sweep) on one book (e.g. Aeneid I) before scaling; compare the fine-tune to the
   LLM-stylizer baseline.
3. **C last** — most open-ended (synthetic-data design); the prompted `Stylizer`
   already covers the need in the meantime.

See [[translation-models-plan]] and [[stylizer-verse-layer]] in memory.
