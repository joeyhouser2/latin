# OCR / HTR: known gaps and manual-download sources

_Generated from `ingest/ocr_notes.py` -- edit that file, then run `python -m ingest.ocr_notes > docs/ocr-gaps.md`._

## Known gaps

### Uncial, rustic capitals and papyrus are unreadable  _(blocker)_

Every installed handwriting model scores ~30% known words on St Gall Cod. Sang. 226 (uncial on papyrus), against 75-85% on Carolingian and Gothic hands. Tesseract fails too. The pipeline refuses to ingest such output (under 50%).

**Next:** Fine-tune a model on 20-50 hand-corrected pages of the script (eScriptorium / `ketos train`), or transcribe by hand. Late-antique and early-medieval material (5th-8th c.) is mostly in this class.

### The translator invents content on damaged text  _(blocker)_

On the readable St Gall 195 transcription the Latin model produced fluent errors ("tulips", "second world war"). Better transcription helps, but the translator is the weaker link for HTR text.

**Next:** Evaluate a stronger translator on doc 13402 (St Gall 195); keep the original beside any translation.

### Abbreviation expansion is Latin-only and rule-based  _(limit)_

`ingest/abbrev.py` resolves marks against the Latin corpus vocabulary. Old French, German and Greek handwriting come out with abbreviations unexpanded, and some Latin words stay wrong ("deustuus", split fragments like "si onibus").

**Next:** Per-language expansion, or use an expanding model (Manicule, Frolat `expan`) where it scores well.

### Greek, German-handwriting and Old French models are downloaded but not wired in  _(todo)_

In `models/htr/`: greek_minuscule_s9-12_NFC, german_handwriting. Not yet benchmarked or selectable; Gallicorpora+ (Old French) not downloaded. Greek print goes only through Tesseract `grc`.

**Next:** Add test sets to `scripts/htr_benchmark.py`, then add them to `ingest/htr.py` candidate lists.

### Several major sources cannot be fetched by script  _(limit)_

HathiTrust (Cloudflare), ISTC/CERL and Biblissima (bot check), USTC (login), Gallica text (ALTCHA), SLUB Dresden and Halle (bot check). The project does not defeat these checks.

**Next:** Download by hand (see the manual-download list below) and ingest the PDF with `ocrimages`.

### VD16 is not searchable  _(limit)_

The K10plus SRU endpoint serves VD17 and VD18 only.

**Next:** Search vd16.de by hand and download the linked scan.

### Many VD17/VD18 copies sit on hosts we cannot read  _(limit)_

In a 2,665-record sample, ~30% of records had a copy our connectors read automatically, ~26% had no free link, and ~43% pointed at hosts listed below (some now routed: Tübingen, Rostock, Weimar, Berlin).

**Next:** Add per-host manifest patterns where a host exposes IIIF (check with a live request first).

### No real layout analysis  _(limit)_

Marginalia, interlinear and marginal glosses (glossed Bibles), multi-column pages and music notation are read in whatever order the segmenter yields, and neumes produce junk lines that are only filtered heuristically.

**Next:** Region-aware segmentation with a layout model; keep gloss and text separate.

### Mixed Latin/German books get one language label  _(limit)_

The pipeline labels a book by its dominant language. German passages inside Latin documents are routed per segment at translation time, but a book that is mostly German is simply filed as German.

**Next:** Per-segment language tags at ingest.

### Tesseract reads long s as f  _(limit)_

Where Tesseract wins over the Kraken print models it still misreads ſ as f. The existing long-s repair tools (`scripts/fix_long_s_ocr*.py`) are separate and not run automatically on new ingests.

**Next:** Run long-s correction in the connector when the engine is Tesseract.

### Older OCR-derived documents have not been quality-checked with the new metric  _(todo)_

The known-word rate (`abbrev.vocab_hit_rate`) has not been run over the existing library.

**Next:** Scan the library, list documents under ~60% known words, queue repair or re-OCR.

### No way to correct a transcription in the app  _(todo)_

HTR/OCR output can only be re-run, not edited; corrections are also the training data uncial needs.

**Next:** A page-image + transcription side-by-side editor that saves corrected lines.

## Sources to download by hand

Save the PDF (or images), then ingest it:

```
python scripts/ingest.py ocrimages "path/to/book.pdf#pages=1-60"
```

or use *OCR → Ingest a downloaded file* in the web app. `share` = records in a 2,665-record VD17/VD18 sample.

| Source | Why not automatic | How to get it | share |
|---|---|---|---|
| [HathiTrust](https://babel.hathitrust.org/) | Cloudflare challenge on catalogue, API and page text. | Open the volume, use its Download / PDF option (public-domain volumes only), save the PDF. |  |
| [Gallica (BnF)](https://gallica.bnf.fr/) | Every text and image-viewer endpoint is behind an ALTCHA bot check; the catalogue (SRU) works. | Find a work with the `gallica` connector, open its ark, use Télécharger -> PDF. |  |
| [ISTC (incunabula) / CERL](https://data.cerl.org/istc/) | Bot check. ISTC lists incunabula and links to digitized copies; it holds no text. | Search ISTC, follow the link to the holding library's scan, download the PDF. |  |
| [Biblissima portal](https://portail.biblissima.fr/) | Bot check. It aggregates manuscripts from many French and European libraries. | Search, follow the link to the holding library; if it publishes IIIF, paste the manifest URL into the `iiif` source. |  |
| [USTC (Universal Short Title Catalogue)](https://www.ustc.ac.uk/) | Login required; finding aid only. | Use it to identify editions, then fetch the digitized copy from the library it names. |  |
| [SLUB Dresden](https://digital.slub-dresden.de/) | Bot check on the viewer and its manifests. | Open the link from the VD17/18 record, download the work as PDF from the viewer. | 190 |
| [Halle (ULB Sachsen-Anhalt) / DOI links](https://opendata.uni-halle.de/) | Bot check at opendata.uni-halle.de; DOI links resolve there. | Follow the DOI, download the PDF. | 124 |
| [URN resolvers (nbn-resolving.de / .org)](https://nbn-resolving.org/) | They redirect to whichever library holds the copy (Göttingen, Leipzig, Halle, Bavarian libraries...); the target is not predictable from the URN. | Open the URN; you land on the holding library's viewer. Use its PDF download. If the viewer is one we route (Heidelberg, Göttingen, MDZ), ingest that URL instead. | 446 |
| [Herzog August Bibliothek Wolfenbüttel](http://diglib.hab.de/) | No IIIF manifest found at the URL patterns we tried. | Open the digital copy and use the viewer's download option. | 71 |
| [Universitätsbibliothek Freiburg](https://dl.ub.uni-freiburg.de/) | The Heidelberg-style manifest URL returns 404. | Download from the diglit viewer. | 10 |
| Hamburg, Darmstadt, Mainz, Stuttgart, Karlsruhe, Jena, BVB and others | Small hosts; no manifest pattern confirmed. | Open the record's link and use the viewer's PDF/image download. | 40 |
| [Archive.org items with no text layer](https://archive.org/) | About one item in five is image-only; the text connectors raise for them. | Download the PDF from the item page and ingest it with `ocrimages`. |  |
| [Universitätsbibliothek Tübingen (opendigi)](https://opendigi.ub.uni-tuebingen.de/) | Routed automatically since Oct 2026 (manifest pattern verified live). | Paste the viewer URL into Ingest by identifier (source: iiif). | 49 |
| Rostock (rosdok), Weimar (HAAB), Berlin (SBB) | Routed automatically since Oct 2026 (manifest patterns verified live). | Paste the link into Ingest by identifier (source: iiif); if page download fails, fall back to PDF. | 224 |

