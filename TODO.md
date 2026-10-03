# Roadmap / TODO

Open work for the Latin/Greek reading + translation library, roughly grouped.
Newest decisions at the top of each section.

## Translation models & training data

- [ ] **Retrain the Greek models with a tagged tokenizer.** Under transformers 5.0 `AutoTokenizer` returns a generic `TokenizersBackend` that silently ignores `src_lang`/`tgt_lang`, so `training/finetune.py` trained v1/v2/v3/archaic on untagged `text </s> <unk>` on *both* sides — no `ell_Grek`, no `eng_Latn`. Measured consequence: enabling `NLLBTranslator(nllb_tokenizer=True)` at inference makes every Greek model *worse* (held-out chrF, 1332 pairs unseen by all: v1 25.3→23.2, v2 27.9→26.5, v3 28.8→27.1; stock 18.7→18.3, a wash), so the flag stays off for `grc` and the only real fix is retraining. `finetune.py` now takes `--nllb-tokenizer` to train with the tags. Order of work: (1) cheap A/B — 25k pairs × 1 epoch tagged vs untagged, ~45 min//run on a free GPU, eval both with `scripts/eval_translation.py`; (2) only if tagged wins, retrain v2 (50k × 4 epochs, ~6h) and v3 (perseus_greek+first1k × 3 epochs, ~2.2h/epoch ≈ 6.6h) and re-eval before switching any route. Latin is unaffected: `lat_Latn` isn't an NLLB-200 code, so it's `<unk>` either way.
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
- [ ] **More connectors** — MGH (medieval), CAMENA/CroALa (Neo-Latin), Documenta Catholica Omnia (patristic), EDH (epigraphy). The generic `tei` connector already covers many TEI sources given a URL.
- [ ] **Improve DigilibLT extraction** — some works (e.g. scholia) parse to ~1 segment; the TEI parsing needs work for unusual structures. Prefer specific `DLT…` ids over bulk `canone` for now.

## Reader / UI

- [ ] **Chapter / section navigation** in the reader (jump by book/chapter; currently one long scroll).
- [ ] **Reader controls** — "show original only" toggle, bookmarks, export (e.g. to .docx/.txt).
- [ ] **Bulk-translate action** — "translate this whole document / all untranslated docs" from the UI (currently per-doc button).
- [ ] **Manuscript image viewer** — show the IIIF page image alongside the text (pairs with Phase 5 below).

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

- [ ] **Real web app** — replace Gradio with a FastAPI backend + a proper reader frontend (React), keeping the existing `core/` + `pipeline.py` service layer unchanged.
- [ ] **Scalable storage** — move from SQLite + a single FAISS file to Postgres + pgvector (or Qdrant/LanceDB) once the corpus grows large; persistent index updates instead of full rebuilds.
- [ ] **Accounts & state** — user accounts, saved/bookmarked passages, reading history, per-user collections.
- [ ] **Serving the models** — host the fine-tuned translators behind a small inference service (batched, cached) rather than loading them in-process.
- [ ] **Deployment** — containerize, pick hosting, add background workers for ingest/translation jobs.

## Housekeeping

- [ ] **Rename `iiif_downloader.py` → `iiif_downloader.py`** — the filename is a typo (its own docstring/CLI call it `iiif_downloader.py`).
- [ ] **Retire legacy `rag_ui.py`** — superseded by `app.py` (the README already notes this).
- [ ] **README cleanup** — the embedded `ManuscriptProcessor`/TrOCR code blocks reference a file that doesn't exist; mark them clearly as illustrative examples.
