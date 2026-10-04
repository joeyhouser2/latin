# Roadmap: Basinio da Parma's *Hesperis* — a manuscript HTR workstream

> ## ⚠️ Superseded for the purpose of getting a reading text — 2026-10-03
>
> **A printed edition exists, and I had not looked for one before writing this.**
>
> > **Basinii Parmensis poetae opera praestantiora**, vol. 1, ed. Lorenzo Drudi
> > (with Francesco Gaetano Battaglini), Rimini, *ex typographia Albertiniana*,
> > 1794 — *Hesperis* at pp. 1–288.
> >
> > Google Books [`id=AuzlAAAAMAAJ`](https://books.google.com/books/about/Basini_Parmensis_poetae_opera_praestanti.html?id=AuzlAAAAMAAJ),
> > **full view, PDF and EPUB download.**
>
> 1794 is unambiguously public domain, it is *print* — so `GT4HistOCR` reads it,
> as it reads the 1513 de Brie page — and it is the text Christian Peters' German
> translation works from, so it is not a second-rate witness.
>
> That makes **workstreams A, B and D below unnecessary** for a reading text.
> No IIIF download, no Transkribus credits, no Kraken virtualenv, no ten pages
> of hand transcription. One manual download, then the ordinary OCR path.
>
> **What is still worth taking from this document:** workstream C, which is the
> same post-correction machinery either way — but read the caveat below, because
> re-measurement changed what C is worth. And the closing section on `btv1b`
> manuscripts stands on its own: that argument was never really about Basinio.
>
> **When the manuscript would still matter:** a critical edition collating
> Ms-630 against Drudi, or any case where the 1794 editor's choices are the
> object of study rather than a means to the text. Not for a translation.
>
> ### Caveat on workstream C, measured 2026-10-03
>
> Post-correction is not a safety net for a bad scan. Against the folded
> 392,757-form vocabulary:
>
> | | before | after |
> |---|---|---|
> | Ocland, keyed transcription | 82.1% | **88.6%** (1,217 tokens fixed) |
> | de Brie, Gallica 1513 via GT4HistOCR | 73.2% | **73.2%** (**0** tokens fixed) |
>
> On de Brie `correct()` fixed nothing — 440 tokens had two or more equally near
> candidates, 305 had none, and it refused every one. That refusal is the
> behaviour to keep. But it means edit-distance correction only pays when the
> recogniser is already nearly right, which is an argument *for* the clean 1794
> print and against expecting C to rescue a hard page.

---

Plan drafted 2026-10-02, from a sourcing survey done for the `book_creator`
side. Everything here is new ground for this repo: every connector in `ingest/`
assumes a text already exists somewhere as characters. This one does not exist
as characters anywhere, and that is the whole problem.

*(That last sentence is what the note above corrects. It was true of the
manuscript and false of the poem.)*

---

## Why this work is worth doing

The *Hesperis* is a 13-book Neo-Latin epic on Sigismondo Malatesta's wars,
written for the Rimini court around 1455 and deliberately Virgilian in scale —
the **first large-scale Neo-Latin epic ever completed**. It has never been
translated into English and has no digital text.

It is not undigitised for lack of interest. The Ludwig Boltzmann Institute ran a
funded project on it from 2018 and produced a born-digital edition of **2 books
out of 13**, which were never published because the IT funding ran out. That is
the realistic measure of the effort: a funded institute with Latinists stalled
at 15%.

What makes it tractable here is that this repo already has four of the five
pieces — IIIF downloading, OCR post-correction, a 392k-form reference
vocabulary, and a Latin translation model. The missing piece is the recogniser.

---

## What exists, and why OCR cannot read it

| | |
|---|---|
| **Shelfmark** | BnF, Bibliothèque de l'Arsenal, **Ms-630 réserve** |
| **IIIF manifest** | `https://gallica.bnf.fr/iiif/ark:/12148/btv1b525024658/manifest.json` |
| **Date** | 1451–1500 |
| **Support** | parchment, 137 leaves, 340 × 230 mm |
| **Hand** | *écriture italienne du XVe siècle* — humanist minuscule |
| **Images** | 285 canvases |
| **Contents** | "Basinii Parmensis Hesperidos libri XIII" — the complete poem |

This repo's own `ingest/gallica.py` already states the rule that governs this:

> `bpt6k` = printed monograph (has OCR behind the bot check), `btv1b` =
> manuscript or image-only scan (**never has OCR at all**).

Basinio is `btv1b`. There is no text layer to fetch, and `.texteBrut` would
return the ALTCHA shell even if there were. Tesseract is also the wrong tool:
`GT4HistOCR_2000000.traineddata` (which `book_creator` now uses, and which reads
a 1513 *printed* page well) is trained on type, not on a pen. Running it on a
humanist minuscule produces noise, not errors.

The **IIIF Image API is not bot-checked** — `/iiif/ark:/12148/<ark>/f<N>/full/full/0/native.jpg`
serves real JPEGs. That is how `book_creator` pulled 22 pages of a 1513 printing
earlier today. It does rate-limit: a tight loop earns `429 Trop de requêtes`, and
— the nasty part — Gallica returns the 429 page **with an image content-type**,
so a downloader must check the JPEG magic bytes (`\xff\xd8\xff`) rather than
trusting the status code. `iiif_downloader.py` already has delay and retry; it
needs that one check added.

---

## Workstream A — get the images

Smallest step, and independently useful: 285 images is a 20-minute download.

```bash
python iiif_downloader.py \
  --manifest https://gallica.bnf.fr/iiif/ark:/12148/btv1b525024658/manifest.json \
  --output data/manuscripts/basinio-hesperis
```

**To add first:** validate each download's magic bytes, so a throttled 429 is
retried rather than saved as a corrupt `.jpg` that breaks the next stage.

Request full resolution. Parchment at 340 × 230 mm scanned well gives far more
to work with than a downsampled copy, and HTR is more sensitive to resolution
than print OCR is.

---

## Workstream B — choose a recogniser

Two realistic routes. The decision turns on whether an existing model already
reads this hand, because training from scratch is where the Boltzmann project
died.

### B1 — Transkribus public models (try this first)

Transkribus hosts public HTR models, several trained on 15th-century Italian
humanist hands. If one of them reads Ms-630 at a usable character error rate,
this workstream collapses to an afternoon.

**Do this before anything else:** upload ten representative pages, run two or
three candidate public models, and measure CER against a hand-transcription of
those same ten pages. Ten pages of ground truth is a couple of hours and decides
the entire shape of the project.

Caveats to settle before committing: Transkribus is a hosted service with credit
pricing, and the export licence on model output needs reading if the resulting
text is to be published or sold.

### B2 — Kraken, locally

Fully local, no service, and it fits this repo's existing habits — but it needs
ground truth. `book_creator/cache/kraken/` already holds three Greek models from
[AjaxMultiCommentary](https://github.com/AjaxMultiCommentary/OCR-kraken-models)
(5.2% CER on polytonic print), downloaded but unused; none of them is for a
Latin hand.

Realistic path: start from a general Latin-manuscript model on
[Zenodo](https://zenodo.org/) or via eScriptorium, fine-tune on 30–50
transcribed pages of Ms-630 itself. Fine-tuning on the specific hand is what
makes this work — a mixed model gets to roughly the right letterforms, and the
scribe's own abbreviations are what the fine-tune learns.

**Note the environment risk.** `pip install kraken` pulls its own torch build.
This repo's `latinvenv` has a working torch and so does `book_creator`'s
(upgraded to 2.6.0+cu124 on 2026-09-28 so Opus-MT could load). Kraken belongs in
a **third virtualenv**, for the same reason `requirements-audio.txt` keeps TTS
out of the main one.

---

## Workstream C — post-correction, which already exists

Whatever the recogniser emits goes through machinery this repo already has, and
this is the part that needs no new thinking:

1. **Abbreviation expansion.** A 15th-century scribe abbreviates far more
   heavily than a printer: `ꝑ`, `ꝗ`, the nasal bar, `-ꝫ`, suspension strokes.
   `book_creator/ocr_correct.ABBREVIATIONS` covers the printed set and needs
   extending for the manuscript ones.
2. **Vocabulary correction.** `ingest/ocr_fix.build_reference_vocab` gives
   392,757 forms; `book_creator/ocr_correct.correct` matches unknown tokens at
   edit distance one and **refuses when two candidates are equally near**. Keep
   that refusal. On the Ocland EEBO scan it took in-vocabulary coverage from
   77.6% to 86.6% without inventing a word.
3. **Garble detection.** `ingest/garble_detect.py` and
   `scripts/mark_garbled_segments.py` already flag segments not worth
   translating. A manuscript will produce more of these than any print source.
4. **Long-s and ct repair** — `scripts/fix_long_s_ocr*.py` — applies if the
   scribe's `ſ` survives into the recogniser output as `f`.

---

## Workstream D — structure and ingest

The poem is 13 books. Whatever comes out needs book boundaries before it is
worth anything downstream.

- Book divisions in a manuscript are rubrics, not headings: `LIBER PRIMUS` in
  red, often with a decorated initial. The recogniser may drop or garble them,
  so expect to mark them by hand against the page images. Thirteen marks.
- Ingest as a normal `documents` row with `language_stage` set appropriately and
  `source` recording the shelfmark. `translation_status` starts `untranslated`;
  `scripts/translate_pending.py --doc-id N` then runs `models/nllb-latin` over
  it like any other work.
- `book_creator` can then build it with `--corpus-id N`, which skips alignment
  entirely because the corpus stores one segment per row with its English
  alongside.

---

## Decision order

1. **Download the 285 images.** Cheap, useful whatever follows, and the only
   step with no open questions.
2. **Transcribe ten pages by hand.** This is the real investment and everything
   else depends on it: without ground truth there is no way to measure a model,
   and measuring is what tells you whether this is an afternoon or a season.
3. **Measure Transkribus public models against those ten pages.** If CER is
   under about 10%, take that route and skip B2 entirely.
4. **Only if that fails**, fine-tune Kraken on 30–50 pages in its own
   virtualenv.

Stop after step 3 if the numbers are bad. A manuscript edition built on a
recogniser that guesses is worth less than no edition — the same argument that
kept a vision LLM out of `book_creator`'s OCR path, where it offered
`defendite armis` for *defendite ab oris*: a real Latin word, not on the page,
and one no reader would catch.

---

## What this unlocks beyond Basinio

If B1 or B2 lands, the method generalises immediately. Every `btv1b` item on
Gallica — the manuscripts `ingest/gallica.py` currently has to skip — becomes
reachable, and that is a far larger body of untranslated Latin than the printed
material this repo has been working from.
