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
pip install torch transformers sentence-transformers faiss-cpu gradio accelerate numpy
```

### Run the Web Interface

```bash
python latin_rag_ui.py
# Open http://localhost:7860
```

### Use as a Library

```python
from latin_rag_pipeline import LatinRAG

# Initialize
rag = LatinRAG()

# Index your texts
rag.index_texts([
    ("Gallia est omnis divisa in partes tres...", "Caesar, De Bello Gallico"),
    ("Confiteantur tibi, Domine...", "Psalms (Vulgate)"),
])

# Query (in English or Latin)
results = rag.query("What does Caesar say about Gaul?", k=3)

for r in results:
    print(f"Source: {r.passage.source}")
    print(f"Latin: {r.passage.text}")
    print(f"English: {r.translation}")
```

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

Vatican, Bodleian, and other libraries serve images via IIIF. Here's how to download and process them:

### Download Images from IIIF Manifest

```python
"""
Download manuscript images from IIIF manifests (Vatican, Bodleian, etc.)
"""

import json
import requests
from pathlib import Path
from urllib.parse import urljoin
import time

def download_iiif_manifest(manifest_url: str, output_dir: str, max_images: int = None):
    """
    Download all images from an IIIF manifest.
    
    Args:
        manifest_url: URL to the IIIF manifest JSON
        output_dir: Directory to save images
        max_images: Optional limit on number of images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Fetch manifest
    print(f"Fetching manifest: {manifest_url}")
    response = requests.get(manifest_url)
    manifest = response.json()
    
    # Extract canvases (pages)
    sequences = manifest.get("sequences", [])
    if not sequences:
        print("No sequences found in manifest")
        return
    
    canvases = sequences[0].get("canvases", [])
    print(f"Found {len(canvases)} pages")
    
    # Download each image
    for i, canvas in enumerate(canvases):
        if max_images and i >= max_images:
            break
        
        # Get image URL
        images = canvas.get("images", [])
        if not images:
            continue
        
        resource = images[0].get("resource", {})
        
        # Handle different IIIF versions
        if "@id" in resource:
            # IIIF 2.x
            base_url = resource["@id"]
            if "service" in resource:
                service = resource["service"]
                service_id = service.get("@id", service.get("id", ""))
                # Construct full-size image URL
                image_url = f"{service_id}/full/full/0/default.jpg"
            else:
                image_url = base_url
        else:
            # IIIF 3.x
            image_url = resource.get("id", "")
        
        # For Vatican specifically, construct the URL
        if "vatlib.it" in image_url and not image_url.endswith(".jpg"):
            image_url = f"{image_url}/full/1000,/0/default.jpg"
        
        # Download
        filename = output_path / f"page_{i:04d}.jpg"
        print(f"Downloading {i+1}/{len(canvases)}: {filename.name}")
        
        try:
            img_response = requests.get(image_url, timeout=30)
            if img_response.status_code == 200:
                filename.write_bytes(img_response.content)
            else:
                print(f"  Failed: HTTP {img_response.status_code}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # Be polite to servers
        time.sleep(0.5)
    
    print(f"Downloaded to {output_dir}")


def get_vatican_manifest_url(shelfmark: str) -> str:
    """
    Construct Vatican IIIF manifest URL from shelfmark.
    
    Example: "Vat.lat.3773" -> manifest URL
    """
    # Vatican format: MSS_Vat.lat.3773
    formatted = f"MSS_{shelfmark.replace(' ', '.')}"
    return f"https://digi.vatlib.it/iiif/{formatted}/manifest.json"


# Example usage:
if __name__ == "__main__":
    # Vatican manuscript
    manifest_url = get_vatican_manifest_url("Vat.lat.3773")
    download_iiif_manifest(manifest_url, "manuscript_images/vat_lat_3773", max_images=10)
    
    # Or use any IIIF manifest URL directly
    # download_iiif_manifest("https://example.com/iiif/manifest.json", "output_dir")
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
├── latin_rag_ui.py         # Gradio web interface
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