# Papyri: what is digitized, and can our HTR read it?

_Written 2026-10-27 alongside `ingest/papyri.py`. The corpus counts are from
the full index (all 82,827 texts). The image counts in section 1 come from the
earlier partial index of eight folders. `scripts/papyri_index.py --stats`
shows the current numbers._

## Short answer

- **Published papyri are fully transcribed and openly licensed.** papyri.info
  (DDbDP documentary + DCLP literary texts, CC BY 3.0) holds about 83,000
  texts. About 3% of the documentary texts and 6% of the literary ones are
  Latin. The new `papyri` connector ingests them.
- **The unpublished Oxyrhynchus "rubbish dump" fragments are not available to
  us.** Of roughly 500,000 fragments in Oxford, about 5,000 are published. The
  rest exists only as photographs. Some were used for crowd transcription
  (Ancient Lives), but the public dataset (AL-PUB) explicitly *excludes* the
  unpublished material. The fragments are also almost all Greek.
- **No model we have reads papyrus hands.** On a clean, well-preserved Latin
  papyrus, both installed Kraken handwriting models produced no correct words
  (results below). Line finding half-works. A papyrus model would have to be
  trained.
- **Training a Latin papyrus model is feasible but a real project.** Images
  exist for most published Latin documents that have a catalogue record, and
  their editions give line-by-line ground truth. The work is in pairing image
  lines with edition lines.

## 1. What exists

### Texts: papyri.info (`papyri/idp.data` on GitHub)

| Corpus | Texts | Latin | Readable Latin (≥ 40 letters) | Notes |
|---|---|---|---|---|
| DDbDP (documentary) | 67,980 | 1,416 | **762** | ChLA 186, Vindolanda 130, Latin letters 79, military records 77, Bu Njem 77, Ravenna 36, Albertini tablets 34 |
| DCLP (literary) | 14,847 | 1,247 | **43** | most Latin entries are **catalogue stubs** for early Latin manuscripts (CLA) with no text; Herculaneum and other literary fragments make up the rest |

All 805 readable Latin texts were ingested on 2026-10-27 (5,530 segments).
Greek, for comparison: 46,126 readable documentary and 2,031 literary texts.
| HGV metadata | 68,022 | — | dates, places, German keywords, image links. HGV catalogues *Greek* documents, so about half of the Latin texts have no record |

The papyri.info website is behind an Anubis proof-of-work bot check, which we
do not try to pass. The GitHub repository is the project's own channel for bulk
use. It is licensed CC BY 3.0 and synced hourly.

In the partial index (8 of ~1,200 folders, chosen because they are dense in
Latin), 822 DDbDP texts are Latin and **415 have at least 40 surviving
letters**. Many entries are a few letters between gaps. The connector refuses
those (`MIN_LETTERS`). Of 1,071 DCLP Latin entries in the same folders, 4 have
readable text.

### Images

For the 415 readable Latin documents: 208 have an HGV record, and **172 of
those link to an image**. Hosts: Columbia/APIS (70, mostly Yale's Dura texts),
Berlin papyrus database (57), Michigan (13), Geneva (18), British Museum (6),
RIB online/Vindolanda (6), British Library (9), Leipzig (4).

| Source | Licence | Checked |
|---|---|---|
| Berlin, berlpap.smb.museum | CC BY 3.0, stated on each record page | yes |
| Yale (P.CtYBR, Dura, P.Ital.) via IIIF | manifest carries no rights statement | **not verified** |
| Oxyrhynchus (POxy, Oxford/EES) | credit line required; reuse requests go to the EES archivist | no open licence found |
| Vindolanda | old site superseded by RIB online; licence not stated | **not verified** |
| Geneva (P.Gen.Lat.) | image server returned 503 | — |
| AL-PUB (Ancient Lives, published papyri only) | dataset licence not stated | **not verified** |

**AL-PUB** has about 196,000–206,000 cropped *Greek character* images from
about 5,000 published Oxyrhynchus images, with crowd-sourced (noisy) labels. It
is at data.cs.mtsu.edu/al-pub, and a v2 is on Kaggle. It suits a Greek
character classifier, not line HTR, and contains nothing Latin.

## 2. Experiment: our Kraken models on a Latin papyrus

**Test page:** P.Ital. 1 28 (Yale P.CtYBR inv. 2125). It is a deed of gift from
Ravenna, 613–641, in New Roman Cursive: 9 clear lines on a clean,
well-preserved sheet, which is close to the best case. Ground truth is the
DDbDP edition (TM 114825), with the editor's restorations removed.

| Model | Lines found | Recognised words in the edition | Known-word rate | Char-trigram precision / recall |
|---|---|---|---|---|
| CATMuS Medieval 1.6.0 | 44 fragments | **0 / 34** | 0.41 | 0.25 / 0.08 |
| Manicule 2026 (Latin medieval) | 44 fragments | **0 / 17** | 0.24 | 0.24 / 0.04 |
| _edition text (control)_ | — | — | 0.86 | — |

Both models output noise ("oncsopemaliiumsnuocuriumsndqonom"). They read the
museum label and the ruler ("crros in. 2125", "cenlimetrs") better than the
papyrus. Line *segmentation* finds most of the writing, but it breaks long
lines into pieces, is thrown by the tall ascenders, and boxes the label and
scale bar. This matches the earlier St Gall 226 result (uncial on papyrus,
~0.30 known words, see `docs/ocr-gaps.md`): the medieval models do not
transfer to ancient hands.

A first attempt on BGU 2 628 (Berlin) was discarded. The image shows eight
columns of writing in what looks like Greek cursive, and the edition has 41
Latin lines, so the image and the Latin text could not be confidently matched.

## 3. What a papyrus model would take

1. **Full index** (`scripts/papyri_index.py`, no arguments). This reads all
   ~83,000 files once (~530 MB mirrored into `data/raw/papyri`, ~83,000 small
   requests to raw.githubusercontent.com). It gives exact counts and the
   complete list of image-linked Latin texts.
2. **Image–text pairs.** Download images only from hosts with a usable
   licence (Berlin CC BY first). Pair each with its EpiDoc edition.
3. **Line ground truth.** Editions mark line breaks (`<lb/>`) but not where
   lines sit on the image. Run Kraken segmentation, fix the lines in the scan
   viewer's line editor, and export with the existing Kraken training export
   (commit c3d1765). Transcribe each line with only the *visible* letters:
   drop `[restorations]`, mark gaps, and keep abbreviations unexpanded, since
   a model learns what is on the papyrus, not the editor's resolution.
4. **Fine-tune** with `ketos train` from CATMuS Medieval, which knows Latin
   letter shapes even if not these ones. Plan on the order of a thousand or
   more corrected lines for a usable cursive model. That is an estimate from
   general Kraken practice, not something measured here. Old Roman Cursive
   (Vindolanda, 1st–3rd c.) and New Roman Cursive (Ravenna, 5th–7th c.)
   differ enough that they may need separate models.
5. **Measure** with `scripts/htr_benchmark.py` on held-out pages against
   their editions, as in section 2.

**What it buys:** re-reading and checking published Latin documents from
their images, and a route for Latin papyri whose editions are not in
papyri.info. It does **not** open up the unpublished Oxyrhynchus material:
those images are not public, and they are Greek, where the active work (AL-PUB,
the ICDAR 2023 papyrus competition, Transkribus Greek models) is happening
elsewhere.

## Sources

- papyri.info data: https://github.com/papyri/idp.data (README: CC BY 3.0)
- AL-PUB / Ancient Lives: https://arxiv.org/html/2210.16380 ;
  https://arxiv.org/html/2401.12513 ; https://acwio.github.io/pubs/escience2021-AncientLivesModels.pdf
- Greek papyri character detection (ICDAR 2023): https://link.springer.com/article/10.1007/s10032-024-00504-8
- POxy / EES reproduction terms: https://oxyrhynchus.web.ox.ac.uk/images ; https://ees.ac.uk/reproduction-requests-and-fees
- Vindolanda Tablets Online (superseded by RIB online): https://csad.web.ox.ac.uk/node/4462121
- HTR for medieval Greek in Transkribus: https://aclanthology.org/2026.lt4hala-1.13.pdf
- Berlin papyrus database record (licence statement): https://berlpap.smb.museum/02270/
- Yale P.CtYBR inv. 2125 IIIF manifest: https://collections.library.yale.edu/manifests/33165728
