# Latin RAG + Translation Pipeline

A complete system for searching Latin texts and translating them to English. Query in English or Latin, retrieve relevant passages from your corpus, and get automatic translations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR LATIN CORPUS                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │Manuscript│  │ Patrologia│  │  Perseus │  │  Custom  │        │
│  │  Images  │  │  Latina   │  │  Texts   │  │  Texts   │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │               │
│       ▼             │             │             │               │
│  ┌──────────┐       │             │             │               │
│  │   HTR    │       │             │             │               │
│  │Transkribus       │             │             │               │
│  │ /TrOCR   │       │             │             │               │
│  └────┬─────┘       │             │             │               │
│       │             │             │             │               │
│       ▼             ▼             ▼             ▼               │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Raw Latin Text (.txt)                   │       │
│  └────────────────────────┬────────────────────────────┘       │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                      INDEXING PIPELINE                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│  │  Chunk   │───▶│  Embed   │───▶│  Store   │                 │
│  │  Text    │    │(Multiling)│    │ (FAISS) │                 │
│  └──────────┘    └──────────┘    └──────────┘                 │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                           │
│                                                                │
│  User Query ──▶ Embed ──▶ Search ──▶ Retrieve ──▶ Translate   │
│  (EN or LA)              (FAISS)    (Top-k LA)    (NLLB-200)  │
│                                                                │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                         OUTPUT                                 │
│  • Latin passage with source citation                         │
│  • English translation                                         │
│  • Relevance score                                            │
└───────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Reading Interface

```bash
# 1. Seed the library with a sample medieval text (Einhard's Vita Karoli Magni)
python scripts/seed_demo.py

# 2. Launch the reader + discovery UI
python app.py
# Open http://localhost:7860
```

The **Read** tab shows a document with Latin and English side by side, aligned
sentence by sentence; the **Discover** tab does cross-lingual semantic search
(query in English or Latin) with filters for language stage and untranslated
works. Data persists in `data/corpus.db` (text + metadata) and
`data/index.faiss` (embeddings), both rebuildable from the sources.

> The original RAG demo (`rag_ui.py`) is still present but superseded by `app.py`.

### Run the Web App (browse, read, queue jobs)

```bash
python scripts/serve.py          # http://127.0.0.1:8000, opens a browser
```

On Windows, double-click **`Latin Library.bat`** instead, or put a shortcut on the
Desktop once:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1
powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1 -StartMenu
```

The launcher prefers `latinvenv\Scripts\python.exe` when it exists, because the
interpreter that runs the server is also the one that runs the jobs — and the
jobs want the CUDA build of torch.

The logo — a rubricated versal **L** on parchment, the illuminated initial a scribe
would put at the head of a text — is generated, not hand-drawn twice:

```bash
python scripts/make_logo.py                 # logo.svg, favicon.svg, latin-library.ico, logo-256.png
python scripts/make_logo.py --concept codex # or: versal (default), scriptorium, pilcrow
python scripts/make_logo.py --sheet out.png # compare every concept at 256/64/32/16 px
```

[scripts/make_logo.py](scripts/make_logo.py) defines the mark once as primitive shapes
in a 64×64 grid and renders it twice: to SVG for the page and favicon, and through
Pillow to a multi-size `.ico` for the Windows shortcut. Each concept also declares a
*small* variant — the versal drops its keyline and gold corner dots below 24px, where
they would otherwise turn the letterform to mush. After changing the logo, re-run
`scripts/install_shortcut.ps1` so the Desktop shortcut picks up the new icon.

What it does that `app.py` does not:

| Tab | |
|---|---|
| **Documents** | Every work, filtered by language, era, source, or *our* progress (untranslated / partial / translated / unstyled), with a per-document translated bar. Select works and queue them, or queue one job covering the whole filter. |
| **Reader** | Latin/English side by side, literal or stylized, with a section picker — and a "translate sections 2–5" button, so a 181-part folio can be sampled before committing the GPU to all of it. |
| **Files** | Read-only browse of `data/`, `models/`, `docs/`: raw OCR dumps, id lists, harvest logs, PDF exports, with previews and downloads. SQLite and FAISS files are listed but never served — see the warning below. |
| **Find texts** | Search a connector's catalogue *without ingesting*, then queue the ones you want (see `treatises` below). |
| **Summaries** | Search LLM-written summaries of every summarized work *and each part of it* — keyword and meaning combined. Hits open the reader at the right part. |
| **Jobs** | The queue: which GPU each job is on, live progress parsed from each script's own output, log tail, cancel, requeue. |
| **Search** | The same cross-lingual semantic search as the Gradio app. The embedder and the ~600MB index load lazily on the first query. |

**Jobs** are subprocess invocations of the existing scripts
(`translate_pending.py`, `stylize_library.py`, `ingest.py`, `reindex.py`,
`summarize.py`), queued in `data/jobs.db` with logs under `data/joblogs/`. A job can
carry a "start after" time, so an overnight run can be lined up during the day.
Because the scripts are resumable, cancelling and requeueing is always safe. Closing
the browser does not stop a running job; closing the server's console window does
(the job shares that console), and it comes back as **interrupted** with a requeue
button that resumes from the last committed chunk.

**Scheduling: one job per GPU, one corpus writer at a time.** Every job is pinned
to its own card (`CUDA_VISIBLE_DEVICES` set to the card's *UUID* — `nvidia-smi` and
CUDA number the cards in different orders on this machine, so an index could put two
jobs on one card). On top of that, only one job that writes `corpus.db` runs at a
time. So translations still queue behind each other, but a summary — which only
reads the corpus — starts on whichever card the translation isn't using. Blocked jobs
are skipped over rather than waited on. Only one app instance runs the queue: a
second launch just opens the running one, and a second server on another port serves
pages but runs no jobs (an OS lock beside `jobs.db` enforces it).

> The queue lives in its own SQLite file on purpose: a translation run holds
> `corpus.db` open for hours from another process, and the web app must never be a
> second writer on it. For the same reason the file browser refuses to serve
> `corpus.db` — a byte copy taken while WAL is active is a torn file. Use the
> `sqlite3` backup API.

### Summaries (local LLM)

`scripts/summarize.py` has a local LLM read each translated work — the original and
the machine translation line by line, since the MT is weak and a 12B model reads Latin
well enough to catch its mistakes — and write one summary per part plus one for the
whole work, into `data/summaries.db`. Queue it from the reader (**Summarize**), the
Documents tab (**Summarize selected**), or the Summaries tab (**Summarize everything
translated**).

```bash
python scripts/summarize.py --doc-id 397                 # what the queue runs
python scripts/summarize.py --all --min-translated 0.9   # every mostly-translated work
python scripts/summarize.py --doc-id 397 --model qwen3:14b --source english
```

It uses [Ollama](https://ollama.com) (default model `gemma4:12b`, plus
`nomic-embed-text` for meaning search), but not the Ollama service you already
run: each job starts a **private** `ollama serve` on a free port, pinned to the job's
card, and shuts it down afterwards. Two things had to be right for that pinning to
be real, both found by testing rather than assumed:

* **Vulkan off.** Ollama 0.34 also enumerates GPUs through Vulkan, which ignores
  `CUDA_VISIBLE_DEVICES` — a server "pinned" to the 4070 SUPER loaded onto the 4060
  Ti. The private server runs with `OLLAMA_VULKAN=false`.
* **No orphaned runners.** Killing `ollama serve` on Windows does not kill its
  `llama-server.exe`, which then holds ~8 GB of VRAM with no owner. The server runs
  inside a kill-on-close Job Object, so the runner dies with the job *however* it
  dies (verified with a hard kill).

Parts are saved as they finish, so a cancelled job resumes where it stopped. It opens
`corpus.db` read-only, which is what lets it run beside a translation.

> **Sharing a card with other Ollama work.** The private server controls where *its*
> model goes, not where the shared Ollama service puts yours. If another project loads
> a large model through the shared service onto the card a summary is using, both
> models together overflow VRAM, Windows pages memory to system RAM, and generation
> drops ~40× (seen in testing: 7.6 s per chunk became 240 s). Cancel and requeue: the
> scheduler picks the card with the most free memory at claim time, and finished parts
> are kept.


### Use as a Library

```python
from pipeline import Library

lib = Library()  # persists to data/corpus.db + data/index.faiss

# Add a text: it is sentence-segmented, embedded, and stored
doc = lib.add_document(
    "Gallia est omnis divisa in partes tres...",
    title="De Bello Gallico", author="Caesar",
    language_stage="classical", has_existing_translation=True,
)

# Translate it (fills in English for each segment; cached in the DB)
lib.translate_document(doc.id)

# Read it back, side by side
for seg in lib.get_document(doc.id).iter_segments():
    print(seg.latin_text, "->", seg.english_text)

# Cross-lingual semantic search (query in English or Latin)
for hit in lib.search("what does Caesar say about Gaul?", k=3):
    print(f"{hit.score:.3f}  {hit.document.author}: {hit.segment.latin_text}")

lib.close()
```

---

## Adding Texts: Sources & Connectors

Texts come in through **connectors** — one per source — all driven by a single CLI
(`scripts/ingest.py`). Each connector turns a source into structured documents
that are segmented, embedded, and stored. A connector can also *discover* many
works at once (a category, an index page, a directory).

```bash
python scripts/ingest.py list          # show available sources
```

| Source | `name` | Pull one work by… | `--discover` lists… |
|---|---|---|---|
| The Latin Library | `latinlibrary` | page URL | works linked from an author/index page |
| Latin Wikisource | `wikisource` | page title | full-text search results, or a `Categoria:` |
| **Perseus** (classical Latin canon) | `perseus` | CTS urn / `group.work` | works under a textgroup |
| **Perseus Greek** (classical Greek canon) | `perseus_greek` | CTS urn / `group.work` | works under a textgroup |
| **First1KGreek** (post-classical / patristic Greek, 2nd–6th c.) | `first1k_greek` | CTS urn / `group.work` | works under a textgroup |
| Generic TEI-XML (Patrologia, EpiDoc, CroALa, PTA) | `tei` | XML URL or local path | `.xml` files in a directory |
| **DigilibLT** (late-antique Latin) | `digiliblt` | `DLT…` id | an author's (`AUT…`) works, or `canone` |
| **Corpus Corporum** (Patrologia Latina, medieval) | `corpuscorporum` | text idno | text idnos under a corpus idno |
| **Corpus Thomisticum** (complete Aquinas) | `corpusthomisticum` | page id / URL | work pages from an index |
| **EDCS** (~542k Latin inscriptions) | `edcs` | search query (one Document per query) | — |
| **Treatises** (financial/fiscal/commercial Latin & Greek) | `treatises` | `ia:<archive.org id>` | works matching a theme (`usury`, `money`, `exchange`, `commerce`, `tax`, `weights`, `accounting`, `economy`) or free text |
| **Gallica** (BnF) — *catalogue only* | `gallica` | — (see below) | Latin works matching an SRU query |
| **MDZ** (Bayerische Staatsbibliothek, scanned early-modern prints) | `mdz` | `bsb12188295`, or `bsb…#ocr=tesseract&pages=1-40` | — (see below) |
| **Page images → Tesseract OCR** (folder from `iiif_downloader.py`, or any IIIF manifest URL) | `ocrimages` | directory, or manifest URL `#pages=1-40` | — |
| **VD17 / VD18** (German-region imprints 1601–1800, K10plus SRU) | `vd` | `vd17:<PPN>` / `vd18:<PPN>` | free words (title), or raw `pica.` CQL; `vd17:` / `vd18:` prefix picks one |
| **Europeana** (aggregator; open-licence Latin text) | `europeana` | `/1613/item_…` | free-text query (set `EUROPEANA_API_KEY`) |
| **Google Books** (catalogue → archive.org `bub_gb_` mirror) | `googlebooks` | Books id or URL | free-text query (set `GOOGLE_BOOKS_API_KEY`) |
| **IIIF** (any manifest: Vatican, e-codices, BL, Parker, Heidelberg, Göttingen…; print *and* manuscripts) | `iiif` | manifest URL, `vatlib:Vat.lat.3773`, `ecodices:csg-0390`, `bnf:…`, `bodleian:…` | — |
| **Capitularia** (Frankish royal capitularies, 507–9th c.) ✓ *verified translation status* | `capitularia` | `BK.139` / `Mordek.12` | `untranslated`, a reign (`pre814`, `ldf` = 814–840, `post840`), `all`, or title words |
| **CELT** (Hiberno-Latin, Cork) ✓ *verified translation status* | `celt` | `L100003` | `untranslated`, `all`, or title/author words |
| **Vernacular classics** (de/fr/it/nl/pl/hu/ru, medieval–Renaissance) | `vernacular` | catalogue key `pl:rej-zywot`, or `ws:<lang>:<Wikisource page>` | a language code or `all` |
| Local plain text | `file` | `.txt` path | `.txt` files in a directory |

```bash
# One work
python scripts/ingest.py wikisource "Confessiones (ed. Migne)/1" \
    --author Augustinus --stage late_antique --genre philosophy
python scripts/ingest.py tei \
    https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data/phi0448/phi001/phi0448.phi001.perseus-lat2.xml

# Late-antique (DigilibLT) and Patrologia Latina (Corpus Corporum)
python scripts/ingest.py digiliblt DLT000001 --genre agrimensores
python scripts/ingest.py corpuscorporum 10821 --stage medieval

# Perseus classical canon (CTS urn or group.work), and Aquinas
python scripts/ingest.py perseus phi0474.phi013                    # Cicero, In Catilinam
python scripts/ingest.py perseus urn:cts:latinLit:phi0448.phi001  # Caesar, De bello Gallico
python scripts/ingest.py corpusthomisticum sth0000

# Ancient Greek (the Greek module) — stored with language=grc, read with a Greek column
python scripts/ingest.py perseus_greek tlg0020.tlg001             # Hesiod, Theogony

# EDCS inscriptions matching a query (one Document, one segment per inscription)
python scripts/ingest.py edcs "Augustus"
python scripts/ingest.py edcs "province=Roma"

# Bulk: discover then ingest
python scripts/ingest.py wikisource "Beda" --discover --limit 5 --stage medieval
python scripts/ingest.py latinlibrary https://www.thelatinlibrary.com/aug.html --discover --limit 10
python scripts/ingest.py digiliblt canone --discover --limit 20           # DigilibLT catalogue
python scripts/ingest.py corpuscorporum 38 --discover --limit 10          # corpus 38 = Patrologia Latina
python scripts/ingest.py file ./my_texts --discover

# Optionally translate the first N segments of each doc now (NLLB; slow on CPU)
python scripts/ingest.py wikisource "..." --translate 20
```

Metadata flags (`--author`, `--title`, `--century`, `--genre`, `--stage`,
`--has-translation`) are stored with the work and power the discovery filters.

> **EDCS note:** each inscription becomes its own segment (no sentence-splitting).
> The connector calls EDCS's JSON API directly with `requests`; Playwright was
> only used once to *discover* that endpoint, so it is not a runtime dependency.
> Inscriptions carry heavy epigraphic markup (`Imp(erator)`, `[Aug]ustus`); the
> display text keeps it, but a markup-stripped copy (`Imperator Augustus`) is what
> gets embedded, so inscriptions search well. See *Embedding & re-indexing* below.

### The Dull Books: `treatises`

The corpus leans towards material people translate because they want to read it —
poetry, liturgy, patristics. The `treatises` connector goes after the opposite:
early-modern Latin (and some Greek) technical prose on money, interest, exchange,
taxation, weights and accounting. *De usuris*, *De monetis*, *De cambiis*, *De
vectigalibus populi Romani*. These are untranslated in the strong sense — nobody
has ever wanted to read them in English, which is exactly why machine translation
is the only way they ever will be.

```bash
python scripts/ingest.py treatises usury --discover --limit 10 --stage early_modern
python scripts/ingest.py treatises ia:bub_gb_D2hqS7meY7YC        # Budel, De monetis (1591)
python scripts/ingest.py treatises "de ponderibus" --discover --limit 5
```

Or use the web app's **Find texts** tab, which shows the catalogue with dates and
a "has text" flag before you queue anything.

It searches two catalogues, which do different jobs:

* **archive.org** — searched by title against a built-in vocabulary per theme, and
  the only one of the two that serves text. Its `bub_gb_*` items are Google Books
  scans of exactly this literature. The fetch path goes through `/metadata/<id>`
  first to find the item's real text derivative: roughly one item in five here is
  an image-only scan with no OCR at all, and the naive `<id>_djvu.txt` URL that
  `archiveorg` uses 404s on those.
* **Gallica (BnF)** — catalogue only. The SRU search API is open (a single search
  for *de usuris* in Latin returns 461 works), but every full-text endpoint now
  sits behind an ALTCHA bot check that serves a JavaScript shell to scripts. So
  `gallica`'s `fetch()` **raises** rather than returning text: the failure it
  prevents is ingesting 50KB of French navigation chrome as a Latin treatise and
  queueing it for translation. Use it to find works, then look for the same
  edition on archive.org or download it by hand and ingest with `file`.
* **MDZ (Munich)** — scanned early-modern prints, and unlike Gallica it is open to
  scripts. `mdz` reads the library's own per-page hOCR by default (`ocr=auto`),
  and falls back to local Tesseract (`lat` model) only for pages with no usable
  text; `#ocr=tesseract` forces local OCR, `#pages=a-b` limits the range. Try both
  engines on a few pages before committing to a whole book. There is no
  `--discover`: find items on digitale-sammlungen.de (filter *Latin*) and pass
  their `bsb` ids. Both connectors share `ingest/pagetext.py`, which rejoins
  hyphenated line ends, drops folio/signature lines and catchwords, rejoins
  sentences cut by a page break, and folds long s (ſ) to s. Tesseract is for
  **print only** — on handwriting it produces fluent-looking noise, so both
  connectors refuse text with almost no Latin function words. Manuscripts need
  HTR (see the Manuscripts section).

* **Catalogues that point at scans (VD17/18, Europeana, Google Books)** — these
  record *where* a book is and let `ingest/copies.py` route each digital-copy link
  to a connector that can read that host (MDZ, archive.org, or a IIIF manifest on
  Heidelberg / Göttingen / e-codices / Vatican / Goobi viewers / MPI). A record
  whose only link is a viewer page we cannot read raises `NoReadableCopy` and lists
  the links. In a 60-record VD17 sample about a third routed straight to MDZ.
  Options pass through after `#`: `vd17:005436001#pages=1-30`.
* **`iiif` picks its engine from the pages.** It OCRs three sample pages with
  Tesseract; if they read as Latin it is print and Tesseract does the rest,
  otherwise it is treated as handwriting and goes to HTR (`ingest/htr.py`).
  `#mode=print|htr` overrides. Handwriting runs **Kraken + the CATMuS Medieval
  model** in an isolated `.venv-htr` (own torch), on whichever GPU has the most
  free memory (~15 s/page on a 4070-class card; CPU works but takes minutes per
  page). Output is graphematic, so `ingest/abbrev.py` then expands it: unambiguous
  glyphs (ȩ ꝑ ⁊ ꝓ), macron/`&` endings (resolved against the corpus vocabulary),
  nomina sacra, scribal run-togethers ("inmulieribus"), and drops the junk lines
  the model invents over neumes and stains. The raw transcription stays in
  `data/raw/iiif_*/htr.json`. Still heuristic — it makes text translatable, not
  edited. Setup is in the `ingest/htr.py` docstring.
* **Which model reads what** (`scripts/htr_benchmark.py`, known-word rate on three
  sample pages each): Latin print 1744 — CATMuS-Print 0.88, Reichenau 0.87, Tesseract
  0.68; Carolingian minuscule — CATMuS Medieval 1.6.0 ≈ Manicule 0.79; 14th-c.
  Gothic — Manicule 0.65 vs CATMuS 0.56; the Frolat models trail (0.4–0.5);
  **uncial on papyrus — every model ~0.30 (unreadable)**. The `iiif` connector tries
  each installed candidate (`ingest/htr.py: HAND_MODELS / PRINT_MODELS`) on three
  pages and keeps the best, and refuses handwriting output under 50% known words.
  Models live in `models/htr/` (Zenodo, CC-BY/CC0; record ids in `htr.py`).
* **Scan viewer and corrections.** On the OCR page (or a reader header) **Pages** opens the page image beside its transcription; edit and Ctrl+S saves a per-page correction to `corrections.json` next to the images, and **Re-ingest with corrections** makes a corrected copy (corrections never touch corpus.db directly). Greek minuscule is detected automatically (`#script=greek|latin` forces it). `python -m ingest.ocr_audit --all` scores every long or scan-derived Latin document by known-word rate; results show on the OCR page.
* **Doubtful words.** The viewer's *Find doubtful words* re-reads the page for per-word confidence (Tesseract word scores, or Kraken character confidences for handwriting; new handwriting ingests record it automatically in `<cache>.lines.json`) and marks suspect words red and weak ones amber on the image and in a clickable list (`ingest/linecheck.py`). Measured with `python scripts/linecheck_eval.py` on two St Gall 195 pages read against the images (226 words, 37 wrong, 16%): the flags cover 28% of words and catch 95% of the errors at 55% precision (about 3.4x better than chance). Caveats: the reference readings are Claude's, not an edition; the split-word rule was designed after seeing the second page's misses; and the unknown-word and split-word signals use a vocabulary built from the corpus itself. Confidence alone catches about 3 in 10 errors; most of the rest are run-together or split words, which come from the vocabulary rules.
* **Testing OCR.** `python scripts/ocr_selftest.py` runs the real engines on known text; `python scripts/ui_smoke_ocr.py` clicks through your running app (queues and cancels one job); `python scripts/ui_e2e_ocr.py [--show]` starts a throwaway app with a scratch library and runs a full scan -> viewer -> correction -> re-ingest cycle in a real browser.
* **Known OCR gaps and hand-download sources** are tracked in
  [`docs/ocr-gaps.md`](docs/ocr-gaps.md), generated from `ingest/ocr_notes.py` and
  shown on the web app's **OCR** page, which also lists scan-derived documents and
  takes a downloaded PDF or image folder (`ocrimages` accepts PDFs; it uses the same
  engine selection as `iiif`).
* **OCR in the web app.** Find-texts rows are flagged **text**, **scan · library OCR**
  or **scan · OCR needed**; the OCR row above the table picks the engine (auto /
  library / Tesseract / handwriting) and a page range for every ingest you queue,
  and *Ingest by identifier* takes a `bsb…`, `ecodices:…`, `vatlib:…` or manifest
  URL directly. Documents made from page images carry an **OCR** badge in the
  library (the connectors stamp `[OCR: engine]` into `source`). Handwriting jobs
  run on the card the queue pinned, not whichever is freest.
* **Not built, and why.** HathiTrust (Cloudflare challenge on every endpoint),
  ISTC/CERL, Biblissima and USTC (bot-check / login walls) cannot be read by a
  script without defeating those checks, which this project does not do — find
  the copy there by hand and pass the manifest or archive.org id to `iiif` /
  `treatises`. VD16 is not exposed on the SRU endpoint.

Long OCR blobs are split into numbered ~1200-word sections. That is not tidiness:
every scoped pass in this project (`--section-range`) works in sections, so a
600-page folio arriving as one section is all-or-nothing — 40,000 segments or
none. Chunking is what makes "translate part 1 and see whether the OCR is good
enough" possible.

> Expect noisy OCR. These are 16th–18th-century prints with long s, heavy
> abbreviation and mixed Greek; the garble detector (`ingest/garble_detect.py`)
> and the long-s repair tooling (`scripts/fix_long_s_ocr*.py`) exist for exactly
> this material.

### Verified translation status: `capitularia` and `celt`

"Untranslated" is the claim this library exists to make, so these two connectors
don't guess it — each checks the source's own scholarly record and stores the
evidence with the document (`documents.translation_evidence`, shown under the title
in the reader and in the Find-texts tab):

```bash
python scripts/ingest.py capitularia untranslated --discover --limit 400
python scripts/ingest.py celt untranslated --discover
```

**Capitularia** (Cologne, `github.com/cceh/capitularia`) — 340 Merovingian and
Carolingian capitularies, 264 of them 9th-century. Every capitulary record lists its
published translations; each is resolved against the project bibliography and
classified by language (the entry's own note — "enthält dt. Übersetzungen" — then
its title, then its place of publication). English printed before 1929 →
`translated` (public domain: Munro's 1900 *Laws of Charles the Great*); later English
→ `translated_paywalled` (King 1987, Loyn 1975, Dutton 2004…); only non-English →
`untranslated`. Result: **233 untranslated**, 54 in copyright, 17 public domain,
36 unknown. The unknowns are honest: 68 capitularies cite *Domínguez 2014*, which is
missing from the project's bibliography and couldn't be identified, so its language
isn't guessed. Four standard English readers the bibliography omits (Hillgarth 1986,
Fouracre & Gerberding 1996, McNamara 1992, Ehler & Morrall 1954) are supplied by
hand and marked as such in the evidence. The text comes from the Boretius–Krause
edition where the project has transcribed it and it is near-complete (211
capitularies), otherwise from the fullest manuscript witness, with scribal deletions
dropped, corrections kept and the editors' German notes removed.

**CELT** (University College Cork) — 28 Hiberno-Latin texts. Two checks: CELT's own
English twin (`T201040` translates `L201040`: 9 texts), then the text's bibliography
— counting only *editions* lists, since every false positive in an audit of all 28
came from secondary literature (a lecture titled *Translations and Adaptations in
Irish*; unrelated Galen translations). Result: 13 translated, 3 in copyright, **12
untranslated** — the Irish annals, *Vita Ite*, the hymn *Adelphus adelpha mater*, the
*Regimen na Sláinte* texts, and others.

> **Windows crash, fixed in `core/__init__.py`.** Any HTTPS request made before the
> embedder first loads used to segfault the process when the embedder then pulled in
> pyarrow (a DLL load-order clash) — no traceback, just exit 139. That hit
> `scripts/ingest.py` for *every* network connector (it fetches, then embeds), and
> would have hit the web server (Find texts, then Search). `core` now imports pyarrow
> first; it is imported before any network activity by every entry point.

### Training Era-Specific Translators

Off-the-shelf Latin/Greek MT (NLLB, OPUS-MT) is trained mostly on classical/
ecclesiastical text and is weak on medieval and late-antique Latin. To improve
that, you can mine your own parallel corpus and fine-tune an open model — no paid
API required.

```bash
# 1. Build the parallel corpus (one-time; pure Python, no GPU)
python scripts/mine_parallel.py --all --era classical --out data/parallel/perseus_latin.jsonl   # ~15k Latin pairs
python scripts/mine_parallel.py --all --greek --era ancient --out data/parallel/perseus_greek.jsonl  # ~64k Greek pairs
python scripts/load_grosenthal.py --out data/parallel/grosenthal.jsonl                          # ~99k Latin pairs (incl. Vulgate)

# 2. Fine-tune NLLB-200 on the Latin pairs (needs a CUDA GPU; see note below)
python training/finetune.py --data data/parallel/perseus_latin.jsonl data/parallel/grosenthal.jsonl \
    --out models/nllb-latin

# 3. Done — the reader uses it automatically (see below)
```

**The reader auto-routes by language.** `Library.translate_document()` picks the
translator for each document's `language` via `TRANSLATOR_MODELS` in `pipeline.py`:
a Latin doc uses `models/nllb-latin`, a Greek doc uses `models/nllb-greek-v2`
(with diacritic-stripping applied to match training), and anything without a
trained model falls back to stock NLLB. Drop a fine-tuned model into `models/`
and the "Translate" button starts using it — no code change.

**Quantify a model** against stock on a held-out slice:

```bash
python scripts/eval_translation.py --data data/parallel/perseus_latin.jsonl data/parallel/grosenthal.jsonl \
    --lang la --holdout-from 25000 --by-era --model "stock=stock" --model "tuned=models/nllb-latin"
```

The miner aligns a work's `-lat`/`-grc` edition against its `-eng` edition at the
deepest CTS citation level they share (`--all` does the whole repo in one git-tree
call). Each pair is `{src, tgt, citation, src_lang, era, source}`, tagged with
`era` (`language_stage`). Together the sources give ~114k Latin pairs (classical +
late-antique via the Vulgate) and ~64k ancient-Greek pairs. Mined corpora live in
`data/parallel/` and checkpoints in `models/` (both gitignored, rebuildable).

> **GPU note:** `training/finetune.py` is CPU-runnable but impractically slow for
> the full corpus. Install a CUDA build of PyTorch first
> (`pip install torch --index-url https://download.pytorch.org/whl/cu121`, matching
> your CUDA version); the script auto-enables fp16 when a GPU is present.

### Greek Module

Documents carry a `language` field (`la` Latin, `grc` ancient Greek). Greek texts
are segmented with Greek punctuation (`·`, `;`), stored with `language=grc`, and the
reader labels the original column "Greek". Use the `perseus_greek` connector for the
ancient Greek canon.

> The embedding model handles Greek script, so **Greek-query** search is strong, but
> **English→ancient-Greek** cross-lingual search is weak (a model limitation). High-
> quality Greek translation is future work (the translator is pluggable).

### Vernacular Module

`vernacular` brings famous medieval/Renaissance works in German, French, Italian,
Dutch, Polish, Hungarian and Russian into the library, so English readers can
reach what's canonical in those literatures but rarely translated. It is a curated
catalogue (`ingest/vernacular.py::CATALOG`) pinned to Wikisource editions; add a
work by adding an entry, and run `VernacularConnector().check()` to confirm each
page resolves and isn't a stub. Documents carry `language` = `de`/`fr`/`it`/`nl`/
`pl`/`hu`/`ru`, are segmented on `.?!` only (medieval `;`/`:` are clause marks), and
translate via stock NLLB. Old stages (Middle High German, Middle Dutch, Old
Hungarian, Old East Slavic) are far outside NLLB's training data, so treat output
there as a rough gloss. `translation_status` is left `unknown` until
`scripts/enrich_translation_status.py` checks it.

For the old stages NLLB cannot read, `--engine llm` translates with a local Ollama
model given a per-stage briefing (spelling conventions, archaic grammar) and the
work's own metadata (`core/llm_translator.py`). It runs on a private Ollama pinned to
whichever card `CUDA_VISIBLE_DEVICES` names; use the GPU's UUID from `nvidia-smi -L`,
since index 0 is not always the same card to CUDA and to nvidia-smi.

```bash
CUDA_VISIBLE_DEVICES=GPU-xxxx python scripts/translate_pending.py --doc-id 13408     --engine llm --redo          # --redo replaces the existing NLLB translation
```

Trial results: good on Old East Slavic (the Slovo), usable on Early New High German
and Old Polish, but it cannot read 12th-century Hungarian and will produce confident
filler for it. The model's own "low confidence" flag was never raised in testing, so
do not rely on it; spot-check anything pre-1300.

```bash
python scripts/ingest.py vernacular pl:rej-zywot --stage early_modern
python scripts/ingest.py vernacular all --discover
python scripts/translate_pending.py --language pl
```

### Embedding & Re-indexing

Each segment stores the display text plus an optional `embed_text` — a
markup-stripped copy (editorial brackets removed, letters kept) used for semantic
search. After changing the embedding/normalization logic, or ingesting older data,
rebuild the index from the store:

```bash
python scripts/reindex.py    # backfills embed_text and rebuilds data/index.faiss
```

**Add your own source:** subclass `Connector` (implement `fetch`, optionally
`discover`) in `ingest/`, then register it in `ingest/registry.py`.

---

## Parsing Manuscript Documents

Manuscripts come as images, not text. You need an HTR (Handwritten Text Recognition) pipeline to convert them to searchable text.

### Overview

```
Manuscript Image (.jpg/.tiff)
         │
         ▼
    ┌─────────┐
    │   HTR   │  Transkribus, eScriptorium, or TrOCR
    └────┬────┘
         │
         ▼
    Raw Latin Text (with errors)
         │
         ▼
    ┌─────────────┐
    │ Normalize   │  Expand abbreviations, fix OCR errors
    └──────┬──────┘
         │
         ▼
    Clean Latin Text → Feed to RAG pipeline
```

### Option 1: Transkribus (Recommended for Beginners)

**Best for:** Medieval manuscripts, has pre-trained Latin models

1. **Create account:** https://readcoop.eu/transkribus/
2. **Upload images** of your manuscript
3. **Select a model:**
   - Search "Latin" in public models
   - Good options: "Medieval Latin", "Caroline Minuscule", "Gothic"
4. **Run recognition**
5. **Export as plain text or PAGE XML**

```bash
# After export, you'll have files like:
manuscript_page_001.txt
manuscript_page_002.txt
...
```

### Option 2: eScriptorium (Open Source, Self-Hosted)

**Best for:** Large-scale processing, full control

```bash
# Install with Docker
git clone https://gitlab.com/scripta/escriptorium.git
cd escriptorium
docker-compose up -d
# Access at http://localhost:8000
```

Uses Kraken engine under the hood. You can train custom models.

### Option 3: TrOCR (Programmatic, Fine-Tunable)

**Best for:** Integration into Python pipelines

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load model (fine-tune on Latin manuscripts for best results)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Process image
image = Image.open("manuscript_page.jpg").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values

# Generate text
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
```

**Note:** Base TrOCR is trained on modern handwriting. For medieval Latin, you'll need to fine-tune on labeled manuscript data.

---

## Processing IIIF Manuscript Images

Vatican, Bodleian, and other libraries serve images via IIIF. This repo includes a
ready-to-use downloader, **`iiif_downloader.py`**, with a command-line interface.

### Download Images with `iiif_downloader.py`

The script handles IIIF 2.x and 3.x manifests, retries, and rate limiting, and has
shortcuts for several major repositories (Vatican, Bodleian, BnF Gallica, e-codices).

```bash
# Download a Vatican manuscript by shelfmark
python iiif_downloader.py --vatican "Vat.lat.3773" --output vat_lat_3773

# Download from any IIIF manifest URL directly
python iiif_downloader.py --manifest https://example.com/iiif/manifest.json --output out_dir

# Download the first 10 pages only
python iiif_downloader.py --vatican "Vat.lat.3773" --output vat_lat_3773 --max 10

# Download at full resolution (slower, larger files)
python iiif_downloader.py --vatican "Vat.lat.3773" --output vat_lat_3773 --size full

# Other repositories
python iiif_downloader.py --bodleian "MS. Bodl. 264" --output bodl_264
python iiif_downloader.py --bnf "ark:/12148/btv1b8432895r" --output bnf_manuscript
python iiif_downloader.py --ecodices "csg-0390" --output stgallen_390

# Search a few known Vatican manuscripts
python iiif_downloader.py --search-vatican "Virgil"
```

Useful flags: `--max` (page limit), `--start` (skip pages), `--size` (IIIF size, default
`1000,`), and `--delay` (seconds between requests). Images are saved as
`page_NNNN.jpg` and the manifest is saved alongside them as `manifest.json`.

### Use the Downloader as a Library

```python
from iiif_downloader import IIIFDownloader, get_vatican_manifest

downloader = IIIFDownloader(delay=0.5)
manifest_url = get_vatican_manifest("Vat.lat.3773")
downloader.download_manifest(manifest_url, "vat_lat_3773", max_images=10)
```

### Complete Manuscript-to-RAG Pipeline

```python
"""
Full pipeline: IIIF images → HTR → RAG
"""

import os
from pathlib import Path
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from latin_rag_pipeline import LatinRAG

class ManuscriptProcessor:
    """Process manuscript images into searchable text."""
    
    def __init__(self):
        # Load TrOCR (or use Transkribus API)
        print("Loading HTR model...")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    def transcribe_image(self, image_path: str) -> str:
        """Transcribe a single manuscript page."""
        image = Image.open(image_path).convert("RGB")
        
        # TrOCR works best on line-level images
        # For full pages, you may need to segment first
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values, max_length=512)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return text
    
    def transcribe_manuscript(self, image_dir: str, source_name: str) -> str:
        """Transcribe all pages in a directory."""
        image_dir = Path(image_dir)
        pages = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        
        full_text = []
        for i, page_path in enumerate(pages):
            print(f"Transcribing {page_path.name} ({i+1}/{len(pages)})")
            try:
                text = self.transcribe_image(str(page_path))
                full_text.append(f"[Page {i+1}]\n{text}")
            except Exception as e:
                print(f"  Error: {e}")
        
        return "\n\n".join(full_text)


def process_manuscript_to_rag(
    image_dir: str,
    source_name: str,
    rag: LatinRAG = None
) -> LatinRAG:
    """
    Complete pipeline: manuscript images → searchable RAG index.
    
    Args:
        image_dir: Directory containing manuscript page images
        source_name: Name for citation (e.g., "Vatican, Vat.lat.3773")
        rag: Existing RAG instance to add to, or None to create new
    
    Returns:
        LatinRAG instance with manuscript indexed
    """
    # Initialize
    processor = ManuscriptProcessor()
    if rag is None:
        rag = LatinRAG()
    
    # Transcribe
    print(f"\n{'='*60}")
    print(f"Processing: {source_name}")
    print(f"{'='*60}")
    
    text = processor.transcribe_manuscript(image_dir, source_name)
    
    # Save transcription
    output_file = Path(image_dir) / "transcription.txt"
    output_file.write_text(text, encoding="utf-8")
    print(f"Saved transcription to {output_file}")
    
    # Index in RAG
    print("Indexing in RAG...")
    rag.index_texts([(text, source_name)])
    
    return rag


# Example usage:
if __name__ == "__main__":
    # Process a downloaded manuscript
    rag = process_manuscript_to_rag(
        image_dir="manuscript_images/vat_lat_3773",
        source_name="Vatican, Vat.lat.3773 (Virgil)"
    )
    
    # Query it
    results = rag.query("Arma virumque cano", k=3)
    for r in results:
        print(f"\n{r.passage.source}")
        print(f"Latin: {r.passage.text[:200]}...")
        print(f"English: {r.translation}")
```

---

## Processing Different Text Sources

### Plain Text Files

```python
rag = LatinRAG()

# Single file
with open("augustine_confessions.txt") as f:
    rag.index_texts([(f.read(), "Augustine, Confessions")])

# Multiple files
texts = []
for path in Path("latin_corpus").glob("*.txt"):
    texts.append((path.read_text(), path.stem))
rag.index_texts(texts)
```

### TEI-XML (Perseus, Patrologia Latina)

```python
from lxml import etree

def extract_text_from_tei(tei_path: str) -> tuple[str, str]:
    """Extract text and title from TEI-XML file."""
    tree = etree.parse(tei_path)
    root = tree.getroot()
    
    # Handle namespace
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    
    # Get title
    title_elem = root.find(".//tei:title", ns) or root.find(".//title")
    title = title_elem.text if title_elem is not None else Path(tei_path).stem
    
    # Get body text
    body = root.find(".//tei:body", ns) or root.find(".//body")
    if body is not None:
        text = " ".join(body.itertext())
    else:
        text = " ".join(root.itertext())
    
    # Clean up whitespace
    text = " ".join(text.split())
    
    return text, title

# Process Patrologia Latina repo
texts = []
for xml_file in Path("patrologia_latina-dev/data").glob("**/*.xml"):
    try:
        text, title = extract_text_from_tei(str(xml_file))
        if text.strip():
            texts.append((text, f"Patrologia Latina: {title}"))
    except Exception as e:
        print(f"Error processing {xml_file}: {e}")

rag.index_texts(texts)
```

### EpiDoc XML

```python
def extract_text_from_epidoc(epidoc_path: str) -> tuple[str, str]:
    """Extract text from EpiDoc XML (used by many classics projects)."""
    tree = etree.parse(epidoc_path)
    root = tree.getroot()
    
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    
    # Get title
    title = root.findtext(".//tei:title", default="Unknown", namespaces=ns)
    
    # Get text from edition div
    edition = root.find(".//tei:div[@type='edition']", ns)
    if edition is not None:
        text = " ".join(edition.itertext())
    else:
        body = root.find(".//tei:body", ns)
        text = " ".join(body.itertext()) if body is not None else ""
    
    return " ".join(text.split()), title
```

### Corpus Corporum Dump

```python
# If you downloaded from Corpus Corporum or HuggingFace
from datasets import load_dataset

# Load the HuggingFace version
ds = load_dataset("Fece228/latin-literature-dataset-170M", split="train")

texts = []
for item in ds:
    texts.append((item["text"], item.get("source", "Corpus Corporum")))

rag.index_texts(texts)
```

---

## Fixing Corrupted OCR Sources

Scanned/OCR'd sources (as opposed to clean digital editions) can carry two
distinct kinds of corruption, each with its own tooling:

- **Long-s and similar character-level misreads** (ſ→f, ct→ft, letter-spaced
  title-page runs split into fake word breaks) — dictionary + trained-classifier
  pipeline in [`ingest/ocr_fix.py`](ingest/ocr_fix.py), [`ingest/ocr_fix_model.py`](ingest/ocr_fix_model.py),
  [`ingest/ocr_fix_ct.py`](ingest/ocr_fix_ct.py). Run `scripts/check_ocr_corruption.py`
  to scan the whole corpus for candidates, then `scripts/fix_long_s_ocr.py` /
  `scripts/fix_long_s_ocr_model.py` / `scripts/fix_ct_ocr.py` per document
  (dry-run by default, `--apply` to write).
- **Embedded non-Latin script the original OCR never recognized at all**
  (e.g. Greek quotations force-fit into Latin-alphabet noise) — this needs
  re-OCR from the original page images, not text-level correction. See
  [`docs/embedded-nonlatin-script-recovery.md`](docs/embedded-nonlatin-script-recovery.md)
  for the full runbook (finding archive.org page scans, setting up
  language-aware Tesseract, aligning segments to pages, splicing recovered
  text back in, and — importantly — why translating the recovered
  non-Latin text word-by-word causes model repetition-loop degeneration).

Either way, segments too corrupted to translate reliably get an honest
`[untranslatable: ...]` placeholder rather than a fabricated translation —
see [`ingest/garble_detect.py`](ingest/garble_detect.py). This is wired into
`scripts/translate_pending.py` automatically for all documents.

## Improving Translation Quality

NLLB-200's Latin is trained mostly on ecclesiastical/modern Latin. For Classical Latin, consider:

### Option 1: Use Multiple Translations

```python
def translate_with_fallback(text: str) -> dict:
    """Try multiple translation approaches."""
    results = {}
    
    # NLLB
    results["nllb"] = rag.translator.translate(text)
    
    # Could add: fine-tuned model, GPT-4, etc.
    
    return results
```

### Option 2: Fine-tune on Parallel Corpus

```python
# Download parallel corpus
from datasets import load_dataset
ds = load_dataset("grosenthal/latin_english_parallel")

# Fine-tune NLLB (simplified - see HuggingFace docs for full training)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./nllb-latin-finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=1000,
)

# ... tokenize data, create trainer, train
```

### Option 3: Use LLM for Translation

```python
import anthropic

def translate_with_claude(latin_text: str) -> str:
    """Use Claude for high-quality translation."""
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Translate this Latin text to English. Preserve the meaning accurately:\n\n{latin_text}"
        }]
    )
    
    return response.content[0].text
```

---

## File Structure

```
latin-rag/
├── latin_rag_pipeline.py   # Core RAG + translation logic
├── rag_ui.py               # Gradio web interface
├── iiif_downloader.py      # IIIF manuscript image downloader (CLI)
├── requirements.txt        # Dependencies
├── README.md               # This file
│
├── corpus/                 # Your Latin texts
│   ├── augustine.txt
│   ├── caesar.txt
│   └── ...
│
├── manuscripts/            # Downloaded manuscript images
│   └── vat_lat_3773/
│       ├── page_0001.jpg
│       ├── page_0002.jpg
│       └── transcription.txt
│
└── index/                  # Saved FAISS index
    ├── latin_corpus.index
    └── latin_corpus.passages.json
```

---

## API Reference

### LatinRAG

```python
rag = LatinRAG(
    embedder=None,      # Custom embedder, or uses default multilingual
    translator=None,    # Custom translator, or lazy-loads NLLB
    vector_db=None      # Custom vector DB, or creates FAISS
)

# Index texts
rag.index_texts([(text, source_name), ...], chunk_size=500, overlap=100)
rag.index_files([(file_path, source_name), ...])

# Query
results = rag.query(query, k=5, translate=True)
# Returns: List[RetrievalResult]
#   - passage: LatinPassage (text, source, chunk_id)
#   - score: float
#   - translation: str or None

# Persistence
rag.save("path/to/index")
rag.load("path/to/index")
```

### LatinTranslator

```python
translator = LatinTranslator(model_name="facebook/nllb-200-distilled-600M")
english = translator.translate("Gallia est omnis divisa in partes tres")
translations = translator.translate_batch(["text1", "text2"])
```

---

## Troubleshooting

### "CUDA out of memory"
- Use smaller batch sizes
- Use `facebook/nllb-200-distilled-600M` instead of larger variants
- Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`

### Poor translation quality
- NLLB is weak on Classical Latin; consider fine-tuning
- Try shorter chunks (200-300 chars)
- Use Claude/GPT-4 for critical translations

### HTR errors on manuscripts
- TrOCR expects line-level images; segment pages first
- Use Transkribus for complex layouts
- Pre-trained models need fine-tuning for specific scripts

### IIIF download failures
- Vatican limits request rate; add delays
- Some manifests require authentication
- Check if images are actually available (some are metadata-only)



## Credits

- Embedding: [Sentence Transformers](https://www.sbert.net/)
- Translation: [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)
- Vector Search: [FAISS](https://github.com/facebookresearch/faiss)
- HTR: [Transkribus](https://readcoop.eu/transkribus/), [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten)
## Working on more than one computer

`corpus.db` / `summaries.db` are local and gitignored. What *is* committed is a
small ledger of work already done, `data/ledger/*.jsonl`, keyed by
`documents.source` (not the local `id`):

```
git pull
python scripts/ledger.py sync      # import others' summaries, fold in this machine's progress
python scripts/ledger.py status --list 20
python scripts/translate_pending.py --skip-done-elsewhere ...
# ...work...
python scripts/ledger.py sync && git add data/ledger && git commit
```

The ledger records *that* a document is translated (and how much), and carries
summaries whole; it does not carry the English text itself.
