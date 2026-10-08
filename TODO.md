# Roadmap / TODO

Open work for the Latin/Greek reading + translation library, roughly grouped.
Newest decisions at the top of each section.

## Translation models & training data

- [x] **Greek language-tag question — CLOSED, no retraining needed** (measured 2026-10-04). Under transformers 5.0 `AutoTokenizer` returns a generic `TokenizersBackend` that silently ignores `src_lang`/`tgt_lang`, so v1/v2/v3/archaic all trained on untagged `text </s> <unk>` on *both* sides. Enabling `NLLBTranslator(nllb_tokenizer=True)` at inference makes them *worse* (held-out chrF over 1332 pairs unseen by all: v1 25.3→23.2, v2 27.9→26.5, v3 28.8→27.1; stock 18.7→18.3, a wash), so the flag stays off for `grc`. Retraining with the tag was then A/B'd — two arms, identical recipe, differing only by `finetune.py --nllb-tokenizer` — and came out **dead even**: 26.4 chrF / 5.8 BLEU either way on the full held-out set (300-pair subset 29.6 vs 29.7; in-training eval 33.16 vs 32.83). Reason: training is one direction only (grc→eng), so a constant tag carries no information; the German win came from the *stock* model needing to identify its input language, which a one-pair fine-tune doesn't. **Don't retrain v2/v3 for this.** Latin is moot — `lat_Latn` isn't an NLLB-200 code. For future fine-tunes: use the 4060 Ti (`CUDA_VISIBLE_DEVICES=1`), which runs batch 8 at ~1.4 it/s; the 4070 SUPER spills at the same settings and crawls at 42 s/it.
- [ ] **Patristic-Latin parallel corpus** (to train a real late-antique/medieval Latin model — the 200s–900s "Loeb gap"). The straightforward approaches failed (see below); pick one:
  - **(a) Shelve it** — `nllb-latin` already has ~32k Vulgate pairs, so it reads the period acceptably. Lowest effort; recommended unless quality proves insufficient.
  - **(b) LaBSE/LASER-class aligner** — proper cross-lingual sentence mining (à la Bertalign) over CCEL English (ANF/NPNF) ↔ Patrologia Latina. The "correct" tool, but a real build + heavy new model dep, and uncertain payoff (Latin is poorly supported even by big multilingual models; the translations are paraphrastic).
  - **(c) Hand-source chapter-structured Latin** per work and align to CCEL chapters. Reliable per work, but manual and doesn't scale.
  - *Why it's hard:* no pre-aligned dual editions (the Greek win relied on First1KGreek's CTS editions); our embedder is weak on Latin; Wikisource/Migne Latin is flat (no chapter markers). Reusable pieces already built: `training/ccel.py` (CCEL ThML extractor), `training/align.py` (alignment scaffolding).
- [ ] **Finish & evaluate Greek v3** (patristic, training now): when `models/nllb-greek-v3` lands, add it to the front of `TRANSLATOR_MODELS["grc"]` and run `scripts/eval_translation.py` v2-vs-v3 on a held-out patristic slice.
- [ ] **Run the chrF eval at scale** — Latin fine-tune vs stock, and Greek v1/v2/v3 — to put numbers on the wins (harness built: `scripts/eval_translation.py`).
- [ ] **Stage-aware translator routing** — route by `language_stage` (classical vs late/medieval), not just `language`, so era-specific models are used per document. (Currently `TRANSLATOR_MODELS` keys on language only.)
- [ ] **Greek epic weakness** — Homeric/verse Greek stays rough. Add aligned verse data and/or upsample it; diacritic-stripping already helps tokenization.

## Corpus & connectors

- [ ] **Bulk-ingest the 200s–900s reading library** — DigilibLT (late-antique) + Patrologia Latina via Corpus Corporum (patristic→Carolingian) + First1KGreek (patristic Greek). Run ingests with `CUDA_VISIBLE_DEVICES=""` while GPUs train.
- [ ] **Re-translate existing stock-translated docs** with the fine-tuned models (Einhard etc. still show seed-time stock NLLB; currently *deferred* by choice).
- [ ] **More connectors** — Documenta Catholica Omnia (patristic), EDH (epigraphy). MGH, CAMENA/CroALa, DBBE, archive.org and `treatises`/`gallica` (financial & commercial prose) are done; the generic `tei` connector covers many TEI sources given a URL.
- [ ] **Improve DigilibLT extraction** — some works (e.g. scholia) parse to ~1 segment; the TEI parsing needs work for unusual structures. Prefer specific `DLT…` ids over bulk `canone` for now.

## Reader / UI

- [x] **Chapter / section navigation** — done in the web app (`web/`): a section picker plus paged segment loading. The Gradio reader (`app.py`) is still one long scroll.
- [ ] **Reader controls** — "show original only" toggle, bookmarks. Plain-text export is done (`/api/documents/{id}/export`); .docx is not.
- [x] **Bulk-translate action** — done in the web app: select works and queue them, or queue one job covering a whole filter. Jobs run one at a time via `web/jobs.py`.
- [ ] **Manuscript image viewer** — show the IIIF page image alongside the text (pairs with Phase 5 below).

- [x] **Scan connectors** — `mdz`, `ocrimages`, `vd` (VD17/18), `europeana`, `googlebooks`, generic `iiif` (print via Tesseract, manuscripts via Kraken HTR + `ingest/abbrev.py`). Walled sources (HathiTrust, ISTC/CERL, Biblissima, USTC) deliberately not scraped.
- [ ] **HTR quality** — evaluate CATMuS on more hands/centuries; fine-tune on Latin-only ground truth; abbreviation expansion by a model rather than rules; Fraktur/German-mixed prints need `frk` for Tesseract (VD17 German-heavy items come out garbled); drop-cap and neume noise.
- [x] **HTR model search** — benchmarked CATMuS Medieval (old + 1.6.0), Manicule, Frolat, CATMuS-Print, Reichenau; pick-by-trial is wired into `iiif`. Whole manuscript done: St Gall 195 (9549 words, 82% known).
- [ ] **No model reads uncial / rustic capitals / papyrus** (St Gall 226). Needs fine-tuning on hand-corrected pages, or a Greek/Latin late-antique model if one appears. Also untested: Greek minuscule model (`greek_minuscule_s9-12`), German handwriting model, Gallicorpora+ for Old French.
- [ ] **Run HTR for real** — pick a manuscript, transcribe whole thing, ingest, translate, and read the result critically.
- [ ] **Gallica full text** — the SRU catalogue is open but every text endpoint is behind an ALTCHA bot check, so `gallica` can only catalogue. Options: match Gallica hits to archive.org copies automatically, or hand-download and ingest with `file`.

## Search / discovery

- [ ] **Hybrid search** (BM25 + dense) so exact names/terms rank well alongside semantic matches.
- [ ] **Discovery facets** — filter by century range and genre in the Discover tab (metadata already stored).

## Infrastructure / robustness

- [ ] **Ingest embedding robustness** — embedding currently OOMs/crashes if the GPU is busy training (the docs land in the store but not the index; fixed only by `reindex.py`). Make `_embed_document` fall back to CPU on CUDA OOM, or embed on CPU by default during ingest.
- [ ] **Tests** — unit tests for `core/` (store, vectorstore, segmenter, normalize) and a smoke test per connector.
- [ ] **Metadata enrichment** — populate `century`/dates more consistently from sources (drives discovery filters).

## Manuscripts (Phase 5)

- [ ] **IIIF → HTR → ingest pipeline** — wire `iiif_downloader.py` (IIIF images) → handwritten-text recognition (Transkribus/eScriptorium/Kraken/TrOCR) → text → ingest, with `image_region` linking segments to page images. Hardest phase; reads texts that exist only as unparsed manuscript images.

## Licensing & attribution

- [ ] **Corpus license audit** — sources carry different terms; document them and decide what can be redistributed (texts, derived translations, trained models):
  - Wikisource **CC BY-SA**, Perseus / First1KGreek **CC BY-SA**, grosenthal (check), CCEL/ANF-NPNF **public domain**, DigilibLT **CC BY-NC-ND** (non-commercial, no-derivatives — careful), Corpus Corporum (per-text terms vary), EDCS (per terms), Latin Library (public domain per site).
  - Open question: what license applies to **fine-tuned models** trained on this mix (esp. the CC-BY-NC-ND DigilibLT and CC-BY-SA share-alike sources), and to the **mined parallel corpora**. Affects whether models/corpora can be published.
  - Add per-`source` attribution to the reader UI and a `LICENSES.md`.

## Productionization (if it grows past a personal tool)

- [x] **Real web app** — `web/` is a FastAPI backend + a no-build frontend (`web/static/`), over the unchanged `core/` + `pipeline.py` layer: document browser with progress, reader, file browser, catalogue search, job queue. Launch with `python scripts/serve.py` or `Latin Library.bat`. Still single-user and unauthenticated — bind beyond localhost only behind something that authenticates.
- [ ] **Scalable storage** — move from SQLite + a single FAISS file to Postgres + pgvector (or Qdrant/LanceDB) once the corpus grows large; persistent index updates instead of full rebuilds.
- [ ] **Accounts & state** — user accounts, saved/bookmarked passages, reading history, per-user collections.
- [ ] **Serving the models** — host the fine-tuned translators behind a small inference service (batched, cached) rather than loading them in-process.
- [ ] **Deployment** — containerize, pick hosting. Background workers exist locally (`web/jobs.py`: one job at a time, persisted in `data/jobs.db`, resumable, cancellable); a hosted version would need a real broker and auth.

## Housekeeping

- [ ] **Rename `iiif_downloader.py` → `iiif_downloader.py`** — the filename is a typo (its own docstring/CLI call it `iiif_downloader.py`).
- [ ] **Retire legacy `rag_ui.py`** — superseded by `app.py` (the README already notes this).
- [ ] **README cleanup** — the embedded `ManuscriptProcessor`/TrOCR code blocks reference a file that doesn't exist; mark them clearly as illustrative examples.
