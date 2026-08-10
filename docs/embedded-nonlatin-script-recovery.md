# Runbook: recovering embedded non-Latin script from a garbled OCR source

Built 2026-08 for doc 376 (Salmasius, *De modo usurarum liber*, 1638), a
Google-Books/archive.org scan whose original OCR pass never recognized its
embedded (polytonic Greek) quotations as Greek at all — it force-fit the
Greek glyphs into Latin-alphabet lookalikes (`\%uu 'j 7i T? il/msis meei*
«« ciiaswjaj...`), producing text that isn't recoverable by any Latin-only
OCR-correction pass because there's no real Latin underneath it to recover.
Left alone, this is worse than it looks: a translation model doesn't fail
loudly on the noise, it produces fluent, confident, entirely fabricated
English — see [`ingest/garble_detect.py`](../ingest/garble_detect.py)'s
docstring. Use this runbook if another source in the corpus shows the same
symptom.

## 1. Recognize the symptom

Signs a document has this problem (not to be confused with ordinary
long-s/OCR corruption, which *is* recoverable — see
[`ingest/ocr_fix.py`](../ingest/ocr_fix.py)):

- Segments with runs of punctuation-heavy, vowel-poor "Latin" that doesn't
  parse as any real word no matter how you flip f/s: `il/msis`, `ciiaswjaj`,
  `lytlsyggoipiimt`.
- Zero real Greek/Cyrillic/etc. Unicode characters anywhere in the raw
  ingested text (`data/raw/*.txt`) despite the work being a classical
  philology/patristics text that would obviously quote the original
  languages. Check with a quick regex over the Unicode block, e.g.
  `[Ͱ-Ͽἀ-῿]` for polytonic Greek.
- The document's translations read as fluent but semantically bizarre or
  wildly inconsistent with the surrounding scholarly argument (a tell that
  the model translated noise, not text).

Run [`scripts/check_ocr_corruption.py`](../scripts/check_ocr_corruption.py)
first regardless — it's the general long-s corruption scanner and rules that
out. Then scan for the transliteration-noise signature specifically with
[`ingest/garble_detect.py`](../ingest/garble_detect.py)'s `garble_score` /
`is_garbled` (calibrate the threshold per-document the way doc 376's was —
sample scores across bands, spot-check precision, don't just reuse
threshold=10 blindly for a different scan/font/OCR-engine combination).

## 2. Find the original page images

The corpus almost always ingests *text*, not images, so you'll need to go
back to the source scan:

1. Find the archive.org identifier. It's usually in the raw source
   filename/path (`data/raw/<identifier>.txt` or similar) or discoverable by
   searching archive.org for the title.
2. Fetch `https://archive.org/metadata/<identifier>` to list available
   files. Look for a "Text PDF" derivative (`<id>.pdf`, rendered page
   images + embedded OCR text layer) — this is almost always the right
   choice over the raw JP2 zip or image tar, which are much larger and
   need extra unpacking for no OCR-quality benefit.
3. **Ask the user before downloading** (per the standing download-permission
   rule) — state the file, source, and size. These PDFs commonly run
   30-100+ MB.
4. Download to `data/raw/<identifier>_scan/`.

## 3. Set up language-aware OCR

The default local Tesseract install is usually English-only.

1. Check `tesseract --list-langs`. If the needed language(s) are missing,
   download `<lang>.traineddata` from
   `github.com/tesseract-ocr/tessdata_best` (small, a few MB to ~15MB each —
   ask before downloading, same as any file fetch) into a local directory,
   e.g. `models/tessdata/`. Don't assume write access to the system
   `tessdata` dir (it's typically under `Program Files` on Windows and
   needs admin rights) — point `TESSDATA_PREFIX` at your local copy instead.
2. Sanity-check on one page before committing to a full run: render a page
   you know contains the target script at ~400 DPI (PyMuPDF/`fitz`,
   `page.get_pixmap(dpi=400)`), OCR with `-l <lang1>+<lang2> --psm 6`, and
   manually verify real script characters come out where the original OCR
   had noise. This is cheap and catches a bad language pack or wrong PSM
   before you spend 20+ minutes OCR'ing hundreds of pages.

## 4. Align corpus segments to page numbers

[`ingest/page_align.py`](../ingest/page_align.py)'s `align_segments`
sequentially walks segments and pages together (both preserve the book's
linear reading order, so segment page numbers are monotonically
non-decreasing) with an adaptive forward window that widens the longer it
goes without a confident match — needed because front matter (title pages,
dedications) has short fragments that don't reliably match a narrow window,
and a fixed narrow window gets permanently stuck before reaching the first
real content page.

**Match against the pre-correction original text**, not whatever's
currently in the segments table — if you've already run the long-s/ct
correction tiers on the document, the corrected text (`Lector`) won't match
the PDF's still-uncorrected OCR layer (`Leftor`). Pull the original from
the earliest backup snapshot (`data/corpus.db.bak-*`) if the live table has
already been touched.

Validate before trusting it: check zero non-monotonic jumps in the aligned
page sequence, and spot-check several known segments' assigned pages by eye
(the sample page's extracted text should visibly contain that segment's
text, even in its garbled form).

## 5. Re-OCR the implicated pages

[`scripts/reocr_greek_pages.py`](../scripts/reocr_greek_pages.py) renders
and OCRs only the *unique* pages that garbled segments fall on (usually a
small fraction of the whole book — 271 of 1052 pages for doc 376),
checkpointing to a JSON cache every 10 pages so it's resumable. Runs in the
background; ~4-5 sec/page is typical at 400 DPI.

## 6. Splice recovered text back into segments

[`ingest/page_splice.py`](../ingest/page_splice.py)'s `recover_segment`
locates a segment's span within the page's *old* OCR text (normalized,
alnum-only substring search with a shrinking-prefix fallback for short
segments), then maps that span through to the corresponding span in the
*new* OCR text via a local `difflib.SequenceMatcher` alignment (a small
window around the span, not whole-page — cheaper and long-range alignment
drifts more than it helps).

This beat an earlier approach of anchoring off neighboring segments' exact
text, which was fragile: garbling often bleeds across a segment boundary (a
segment right before a "fully garbled" one may itself have a garbled tail),
so a short exact-text anchor frequently doesn't exist. Whole-region sequence
alignment is far more robust to local OCR disagreement in the surrounding
real text.

**Quality-gate before trusting a recovery**: require the mapped span to be
found at all, and be substantial relative to the original (`min_len`,
`min_ratio` in `recover_segment`) — very short segments (1-3 words) often
don't have enough surrounding context for the local alignment to anchor
confidently, and a truncated non-answer isn't better than the honest
untranslatable placeholder it would replace. Expect well under 100% recall
(doc 376: 506/837, 60%) — that's fine; leaving the rest as the placeholder
is the correct, honest fallback for the pages/segments the pipeline
genuinely can't recover.

Apply via [`scripts/apply_greek_recovery.py`](../scripts/apply_greek_recovery.py)
(dry-run by default — review samples before `--apply`).

## 7. ⚠️ Do NOT translate embedded non-Latin runs word-by-word

The first design of the mixed-language translator (split each segment into
script-homogeneous runs, route Greek runs to the Greek NLLB checkpoint and
Latin runs to the Latin one) caused **severe repetition-loop degeneration**:
even with good re-OCR, embedded quotations still come out fragmented into
many 1-4 word runs (residual OCR noise breaks up what should be one
continuous quotation), and translating an isolated word or two as its own
"sentence" sent the fine-tuned NLLB model into collapse — one segment's
English output repeated "the son of Aeschines" seventeen times. That's not
a minor quality hit, it's actively *worse* than the placeholder it replaced,
because it reads as real (if strange) content instead of admitting
uncertainty.

The fix, in [`ingest/mixed_lang_translate.py`](../ingest/mixed_lang_translate.py):
translate **only the Latin portions**, concatenated back into one coherent
string per segment (full sentence context, one model call, no fragment
degeneration), and preserve the recovered non-Latin text **verbatim,
untranslated**, appended as a bracketed note (`[Greek in source: ...]`).
This is also usually the semantically right call independent of the
degeneration bug — most of these short embedded quotations turn out to be
personal names and single technical/legal terms, which shouldn't be
machine-"translated" word by word anyway. A human reader who knows the
source language can interpret the preserved original directly; a genuinely
long, coherent non-Latin passage (a real multi-clause quotation, not
fragments) might be worth actually translating as its own unit if one shows
up, but test that in isolation before trusting it at scale — the same
degeneration risk applies to any input that's unusually short or
out-of-distribution for the fine-tuned checkpoint.

This is wired into [`scripts/translate_pending.py`](../scripts/translate_pending.py)
automatically: any clean (non-garbled), Latin-language segment whose text
contains Greek Unicode characters gets routed through
`translate_mixed_batch` instead of the plain translator.

## 8. ⚠️ Reference-vocabulary drift

The long-s/ct correction tiers (`ingest/ocr_fix.py`, `ingest/ocr_fix_ct.py`)
build their "is this a real word" reference vocabulary from every *other*
document in the corpus. If new documents have been ingested since a
correction pass was last validated on a given document, re-check before
re-running: doc 376's vocabulary grew from 371 to 411 source documents
mid-session here, and picked up enough noise from newly-added (imperfectly
OCR'd) documents that a previously-safe exact-match correction started
flipping a genuinely correct word (`refuta` → `resuta`, `refuta` being real
Latin for "refute"). Don't blindly re-run and `--apply` a correction tier
after the corpus has grown — dry-run and scan the change list again first.

## 9. Retranslate, restyle, re-export

Standard pipeline from here: clear stale `english_text`/`english_styled`
for changed segments (the correction/apply scripts already do this via
`Store.set_latin_texts`), run `scripts/translate_pending.py`, then
`scripts/stylize_library.py`, then `scripts/export_pdf.py` if a portable
export is wanted. See [`docs/../CLAUDE.md`](../README.md) generally and the
main long-s pipeline for the GPU-selection gotcha: `CUDA_VISIBLE_DEVICES`
indices do **not** necessarily match `nvidia-smi`'s own GPU numbering on
this machine (confirmed backwards for the 4060 Ti / 4070 Super pair) —
verify with `torch.cuda.get_device_name(0)` under each candidate index
before assuming which physical card a job landed on, especially before
routing a large model to a GPU with too little VRAM for it.
