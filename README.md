# Practical Guide: Latin Corpus Downloads & Manuscript Access

## 1. BAMMAN'S CORPUS (11K Latin Texts - 1.38B tokens)

### Download Links
| Format | Size | Link |
|--------|------|------|
| **Plain text** | 3.9 GB | https://docs.google.com/uc?id=0B5pGKi0iCsnbZEdHZ3N6d216am8&export=download |
| **DJVU XML** (with formatting) | 9 GB | https://docs.google.com/uc?id=0B5pGKi0iCsnbczZhZTlGdkRNLVk&export=download |
| **Word embeddings** | 2.6 GB | https://docs.google.com/uc?id=0B5pGKi0iCsnbMm9Dd2hmb2UtbEk&export=download |
| **Metadata** | - | https://github.com/dbamman/latin-texts |

### What's in it
- 11,261 OCR'd texts from Internet Archive
- Spans two millennia of Latin
- Includes date of composition metadata (not just publication date)
- OCR quality varies from excellent to terrible

### Caveats
- OCR errors can be significant - may need cleaning
- Contains non-Latin content (use langid.py to filter)

---

## 2. VATICAN MANUSCRIPT IMAGES (DigiVatLib)

### Access
- **URL**: https://digi.vatlib.it/mss/
- **Current count**: 30,327+ digitized manuscripts
- **Format**: IIIF-compliant images
- **License**: NOT licensed for reuse - scholarly use only

### How to Download Images

The Vatican uses IIIF but URLs don't link directly to images. Here's the process:

**Step 1**: Find your manuscript at https://digi.vatlib.it/mss/

**Step 2**: Get the IIIF manifest
- Click "Bibliographic Information" (left sidebar)
- Click the IIIF manifest link at bottom

**Step 3**: Extract image URLs from manifest
The manifest URLs look like:
```
http://digi.vatlib.it/iiifimage/MSS_Vat.lat.3773/[identifier].jp2
```

You need to append IIIF parameters to get actual images:
```
/full/1047,/0/native.jpg
```
Where:
- `full` = entire image region
- `1047,` = width in pixels (height auto)
- `0` = rotation degrees
- `native.jpg` = quality/format

**Step 4**: Use DownThemAll (Firefox plugin) or wget to batch download

**Example full URL**:
```
http://digi.vatlib.it/iiifimage/MSS_Vat.lat.3773/[id].jp2/full/1047,/0/native.jpg
```

### Bulk Download Script Approach
```python
import json
import requests

# Load manifest
manifest = json.load(open('manifest.json'))

# Extract canvas image URLs
for canvas in manifest['sequences'][0]['canvases']:
    base_url = canvas['images'][0]['resource']['service']['@id']
    image_url = f"{base_url}/full/full/0/default.jpg"
    # Download image_url
```

---

## 3. BODLEIAN LIBRARY (Oxford)

### Access
- **URL**: https://digital.bodleian.ox.ac.uk/
- **Polonsky Project**: 1.5M pages from Vatican + Bodleian collaboration
- **Format**: IIIF-compliant

### Easier Downloads
Bodleian manifests often include direct image URLs, so the process is simpler:
1. Get IIIF manifest
2. Use DownThemAll to grab all URLs ending in `.jpg`

### Tools
- **Manifest Editor**: https://digital.bodleian.ox.ac.uk/manifest-editor/
- **Digital Manuscripts Toolkit**: https://dmt.bodleian.ox.ac.uk/

---

## 4. LATIN TRANSLATION MODELS

### Ready-to-Use Models

| Model | Direction | Notes |
|-------|-----------|-------|
| **BryanFalkowski/english-to-latin-v2** | Latin ↔ English | Trained on CCMatrix, handles Classical + Vulgar Latin |
| **facebook/nllb-200-distilled-600M** | Multilingual (200 langs) | Latin included as `lat_Latn` |
| **Helsinki-NLP multilingual** | Various | Check for `la` language pairs |

### NLLB-200 (Recommended Starting Point)

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Latin to English
text = "Gallia est omnis divisa in partes tres"
inputs = tokenizer(text, return_tensors="pt")

translated = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
    max_length=100
)
print(tokenizer.batch_decode(translated, skip_special_tokens=True)[0])
```

**Note**: NLLB includes Latin (`lat_Latn`) but it's trained primarily on modern/ecclesiastical Latin. For Classical Latin, fine-tuning recommended.

### Fine-tuning Approach

1. Start with NLLB-200 or mBART
2. Fine-tune on:
   - grosenthal/latin_english_parallel (101k pairs)
   - Perseus parallel texts
   - Vulgate + Douay-Rheims Bible
3. Use HuggingFace Trainer API

---

## 5. ADDITIONAL DATASETS ON HUGGINGFACE

| Dataset | Description |
|---------|-------------|
| `grosenthal/latin_english_parallel` | 101k translation pairs |
| `pstroe/cc100-latin` | 390M tokens, cleaned |
| `Fece228/latin-literature-dataset-170M` | Corpus Corporum dump |
| `LatinNLP/latin-summarizer-dataset` | 320k rows with translations |

---

## 6. HTR/OCR FOR MANUSCRIPTS

Once you have Vatican/Bodleian images, you need to transcribe them:

### Recommended Tools

| Tool | Type | Notes |
|------|------|-------|
| **Transkribus** | Platform | Best for Latin, has pre-trained models, freemium |
| **eScriptorium** | Open source | Self-hosted, uses Kraken engine |
| **TrOCR** | Model | Microsoft transformer-based, fine-tunable |

### Pre-trained Latin Models
- Transkribus has public Latin models for various scripts
- CREMMA Medii Aevi dataset for Medieval Latin (11th-16th c)
- Expected CER: ~6% out-of-box, ~1.5% with fine-tuning

### Pipeline
```
Manuscript images → HTR (Transkribus/TrOCR) → Raw Latin text → 
Lemmatization (LatinCy) → Translation model → English
```

---

## 7. QUICK START CHECKLIST

### For Translation Model Training
- [ ] Download grosenthal/latin_english_parallel from HuggingFace
- [ ] Download NLLB-200 as base model
- [ ] Fine-tune using HuggingFace Trainer

### For RAG over Untranslated Texts
- [ ] Download Bamman's 3.9GB plain text corpus
- [ ] Clean with langid.py to filter non-Latin
- [ ] Chunk and embed with multilingual model (mE5, LaBSE)
- [ ] Set up vector store (FAISS, Chroma, etc.)

### For Manuscript Processing
- [ ] Set up Transkribus account (or install eScriptorium)
- [ ] Write IIIF manifest scraper for Vatican
- [ ] Build HTR → Translation pipeline

---

## REFERENCES

- Bamman corpus: https://www.cs.cmu.edu/~dbamman/latin.html
- Vatican DigiVatLib: https://digi.vatlib.it/
- Bodleian Digital: https://digital.bodleian.ox.ac.uk/
- IIIF download tutorial: https://www.dotporterdigital.org/how-to-download-images-using-iiif-manifests-part-ii-hacking-the-vatican/
- NLLB-200: https://huggingface.co/facebook/nllb-200-distilled-600M
