# Planning brief: finding quotations of lost works

A handoff for a planning session. The goal of that session is a better plan, not code: what to search, against what baseline, with which methods, and how to tell a real find from noise. Everything below is the current state as of 2026-11-11, with measured numbers where we have them.

## The goal

Find passages in later texts that quote or paraphrase **lost works**, and that nobody has collected yet. The frontier, as the user framed it: classical authors' indirect tradition is heavily mined already; the open ground is under-edited material such as Byzantine scholia, catenae, florilegia, commentaries on Aristotle, and Arabic translations.

The repo is a reading and discovery library for understudied Latin and Greek (`corpus.db`, a FastAPI web app with a GPU job queue and local models through Ollama). Quotation mining is a new branch of it.

## What exists now

**Code** (all committed on `main`):

| Piece | File |
|---|---|
| Matcher, text readers, DFHG loader, citation finder, list of searched texts (`WITNESSES`) | `ingest/quotations.py` |
| CLI: `run`, `fetch`, `recall`, `reuse`, `cites`, `check` | `scripts/mine_quotations.py` |
| Local-model sorting of leads | `scripts/triage_leads.py`, `core/lead_triage.py` |
| Lead store: leads, model tags, user verdicts | `core/leads.py` → `data/leads.db` |
| Web app: Leads tab; job kinds `mine_quotes` (CPU) and `triage_leads` (GPU) | `web/static/app.js`, `web/server.py`, `web/jobs.py` |
| List of Greek works with no open text, by TLG ID | `scripts/tlg_gap_list.py` → `docs/tlg-gap-list.csv` |
| Self-test (no network) | `scripts/quotations_selftest.py` |
| Pilot report with all measurements | `docs/quotation-mining-pilot.md` |

**Method:**
- **Baseline:** DFHG, Müller's *Fragmenta Historicorum Graecorum*, pulled per author from its public API (608 of 629 authors; 8,200 records). Open, CC BY-SA, but historians only and a 19th-century edition. The bulk export is behind a CAPTCHA, so we don't use it.
- **Reuse:** a fragment's words, accent-stripped, matched as shared 3-word sequences; scored as the share of fragment words covered in the best window. A lead is a match in a text DFHG doesn't name as the source.
- **Citation:** a capitalized DFHG author name (Latin name transliterated to a Greek stem, kept only if the author's own fragments use it) within 6 words of a citing formula (φησί, ἱστορεῖ, ἐν τῷ/τῇ/τοῖς…). A lead is a citation whose surrounding words match no DFHG fragment. 29 first names that mostly mean someone else here (Aristophanes, Demosthenes, Dionysius…) are skipped by default.
- **Sorting:** a local model (qwen3:14b) translates the passage, says who the name refers to, then tags same person yes/no/unsure and quotation/paraphrase/mention. Tags only order the list.

**Texts searched** (`WITNESSES`): Athenaeus (Perseus TEI), Suda (First1KGreek), Pindar scholia (First1KGreek `tlg5034`), Procopius of Gaza's catena (PG 87 OCR), Stobaeus (archive.org OCR, 5 vols), Photius's *Lexicon* (Naber, archive.org OCR, 2 vols).

## Measured results

**Recall** on fragments DFHG says come from that text (share found at ≥50% coverage; a shuffled-word control finds 0%):

| Text | Known fragments | Found |
|---|---:|---:|
| Athenaeus, clean TEI | 965 | 96.9% |
| Photius, *Lexicon*, OCR | 47 | 81% |
| Stobaeus, OCR | 67 | 64% |

**Leads now** (224; 3 sorted, none reviewed): Suda 118 reuse + 11 citation; Photius 46 + 8; Athenaeus 3 + 22; Stobaeus 2 + 6; Pindar scholia 3 + 5; catena 0.

**Model sorting**, tested against a hand-reviewed set of 102 citations (7 real, 95 name collisions):

| Prompt | Real ones tagged "the historian" | Collisions tagged "someone else" |
|---|---:|---:|
| Yes/no asked first | 0 of 7 (all "unsure") | 4 of 95 |
| Translate and say who it is first | 7 of 7 | 71 of 95 |

**Novel finds: none confirmed.** Six leads survive hand checking and need comparing with Jacoby/BNJ: Aristodemus in the Pindar scholia (*Pyth.* 3.137, *Nem.* 7.56, *Ol.* 10.83), Arrian in the Suda (A 1218, A 1223), Priscus in the Suda (Π 687). From memory, unverified, the Aristodemus ones are probably Jacoby *FGrHist* 383 and the Arrian ones in Roos's edition.

## What we learned

1. **The matcher isn't the bottleneck.** Recall on clean text is near 97%. The limits are the baseline and the texts searched.
2. **"Missing from DFHG" doesn't mean uncollected.** DFHG is Müller only; Jacoby/BNJ (paywalled) has much more. In Athenaeus, the most-mined text there is, citations with no DFHG match include well-known fragments such as Aristobulus on Susa. Novelty can't be established from open data alone.
3. **Name collisions dominate citation leads.** Before filtering, most unmatched citations in the Suda and Pindar scholia meant a famous namesake. A local model helps only when made to identify the person before answering.
4. **DFHG's source field is messy.** Typos ("Athenoeus", "Atehn."), chained "Idem"/"Ibid."/"Adde"/bare "XLIV, 41", and sources given only in section headings ("ΕΚ ΤΟΥ ΣΤΟΒΑΙΟΥ"). Each misread turns a known fragment into a false "new" lead.
5. **OCR costs about a third of recall** and creates false citation leads (a garbled word breaks the match with a known fragment).
6. **The texts searched so far are the most-mined ones in Greek.** They were chosen because they have open text and a DFHG answer key, which makes them good for tuning and poor for discovery.
7. **The library's own corpus lacks the target genres.** A read-only probe of `corpus.db` (281 prose docs) found 3,544 citation-formula hits and nothing new; the corpus has almost no scholia, catenae, florilegia or Aristotle commentaries. It is mostly hymns, charters, inscriptions and early-modern Latin.
8. **Formulas alone don't work for Latin or Greek in medieval texts.** Latin hits were mostly "Philosophus", "Commentator" and place names; Greek φησίν hits were mostly scripture.

## Constraints

- **No scraping of walled or paywalled sources:** TLG (license forbids copying/downloading), BNJ, PHI's bulk text, LLT, HathiTrust, Gallica full text (ALTCHA), CAPTCHA'd exports. Manual lookups by the user are fine.
- **`corpus.db` safety:** never raw-copy it; long translation jobs may be writing. Quotation work keeps its own `data/leads.db`.
- **Hardware:** RTX 4060 Ti 16 GB and RTX 4070 SUPER 12 GB (the 4070 also runs the user's other projects). Installed Ollama models: qwen3:14b, gemma3:12b, phi4, gemma4:26b, qwen3.8:27b. (gemma4:12b, the summarizer's default, is no longer installed.)
- **Greek handwriting isn't readable yet.** HTR models in the repo read Latin hands; nothing reads Greek minuscule, and no model reads uncial. Monastery manuscript images (Sinai, Athos, Meteora) need that first.
- **Downloads** need the user's OK when they're large or new sources.

## Open questions for the plan

1. **Baseline.** How do we get beyond DFHG? Options to weigh: other open fragment collections (comic and tragic poets, philosophers: are any open and machine-readable?), Wikisource/Perseus editions of FGrHist-era fragments, building our own index of "already collected" from editions' indices of sources (*index locorum*), or accepting manual BNJ checks as the final step and designing the workflow around a small, high-quality list.
2. **Which texts.** Where are quotations of lost works most likely to be uncollected *and* available as open text? Candidates: Constantine Porphyrogenitus's *Excerpta*, Stephanus of Byzantium's *Ethnika* (needs the 1849 Meineke OCR), Eustathius's commentaries, the Etymologica, Byzantine scholia beyond Pindar, Aristotle commentators (CAG: Simplicius, Philoponus, Alexander are on First1KGreek), Cramer's *Anecdota* and *Catenae*, Latin compilers (Gellius, Macrobius, Servius, Nonius, Isidore). `docs/tlg-gap-list.csv` has 1,665 Greek works with no open TEI.
3. **Beyond historians.** Is the right target lost historians at all? Lost philosophy (via the Aristotle commentators), lost poetry (via lexica and scholia), or lost Christian works (via catenae: Origen, Papias, Hegesippus) may have better open baselines or more untouched material.
4. **Method.** Paraphrase detection beyond word overlap (multilingual embeddings? lemmatized matching?), Latin quotations of Greek works, and Arabic translations (out of scope today: nothing in the repo reads Arabic).
5. **Scoring novelty.** What evidence should a lead need before a human spends time on it? (For example: named author + formula + no DFHG overlap + model says same person + not in the author's extant works + not in a second witness already credited.)
6. **Evaluation.** We have answer keys for recall (DFHG's own source attributions). What answer key would measure precision of "new"?
7. **OCR.** Is it worth correcting Greek OCR (the repo has a long-s/OCR correction pipeline for Latin), or better to find texts with clean TEI?

## A good plan would say

- which baseline(s) to use and how to get them legally;
- which 5–10 texts to search next, in order, and why each is likely to hold uncollected material;
- what changes in method each step needs, and how to measure it;
- what a "lead worth a human's time" means, as a checklist;
- what to stop doing.

## How to run what exists

```bash
python scripts/quotations_selftest.py
```

```bash
python scripts/mine_quotations.py run --witness suda
```

```bash
python scripts/mine_quotations.py recall --witness photius_lexicon
```

```bash
python scripts/triage_leads.py --model qwen3:14b
```

```bash
python scripts/tlg_gap_list.py
```

Adding a text means adding a row to `WITNESSES` in `ingest/quotations.py` (name, label, reader kind `tei`/`plain`/`pg`, file → URL, and a pattern for that text in DFHG's source field), then checking it with `recall`.
