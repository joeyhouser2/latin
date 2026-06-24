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