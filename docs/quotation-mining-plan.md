# Quotation mining: the next plan

Written 2026-11-11 in answer to [`quotation-mining-planning-brief.md`](quotation-mining-planning-brief.md), after reading the pilot report, the code in `ingest/quotations.py`, `scripts/mine_quotations.py`, `core/lead_triage.py`, `core/leads.py`, and [`tlg-gap-list.csv`](tlg-gap-list.csv). Availability claims were checked on the web on that date unless marked **(unverified)**. Nothing was downloaded beyond file listings and a few KB of OCR samples; no code or `corpus.db` was touched.

## The short version

1. **Change the question.** In open, printed Greek, nearly every passage that names a lost author has been read by an editor. With open data, the realistic finds are, in order of likelihood: (a) **new witnesses or parallels** to known fragments, (b) **fragments that one collection missed** and a later one has, which lets us *measure* the pipeline, and (c) occasionally an **uncollected item** in a little-read text. Plan for (a) and (b), and treat (c) as a bonus. The unedited material (catena manuscripts, Arabic) needs Greek HTR or Arabic first, and that is a separate decision (Phase 6).
2. **The best new baseline is a table of passages, not of texts.** Jacoby's *FGrHist* parts I–II (1923–30) are openly on archive.org (Vanderbilt scans). Their Greek OCR is unusable (it was read as Latin letters), but the **source lines are readable**, for example `97 (—) SCHOL.GEN.HOM. D. 8 472`. Parse those lines into a "known loci" table: (witness, passage) → collection and fragment number. A lead at a known locus is known, whatever its wording. Add DFHG's source fields and the source lines of the public-domain poetry and philosophy collections.
3. **That table gives us an answer key for precision of "new".** Fragments in Jacoby I–II that Müller (DFHG) lacks are, from DFHG's point of view, "uncollected" fragments that have since been collected. Run the pipeline with DFHG as the only baseline and count how many top-ranked "new" leads are Jacoby additions. That is the first real measure of whether the pipeline can discover anything.
4. **Search next, in order:** the Euripides, Sophocles and Homer scholia (calibration), the less-read Aristotle commentators, Cramer's New Testament catenae (all clean TEI on First1KGreek), then Cramer's *Anecdota*, Eustathius, and Stephanus of Byzantium (archive.org OCR). Details below.
5. **Move the main target from historians to lost philosophy**, through the commentators, where the open baselines (Diels, Rose, von Arnim) are better than anything we have for historians.
6. **Stop** using the most-mined texts for discovery, stop calling a lead "new" when it means "not in DFHG", and stop expanding citation formulas for medieval texts.

---

## Answers to the seven open questions

### 1. Baseline: how to get beyond DFHG

Use three tiers. Each tier is legal to obtain and does a different job.

| Tier | What | Used for | How to get it | Status |
|---|---|---|---|---|
| **A. Fragment text** | DFHG (have); Kock *CAF* (1880–88), Nauck *TrGF*² (1889), Diels *Vorsokratiker* (1912, 1922), von Arnim *SVF* (1903–05), Kinkel *EGF* (1877), Kern *Orphicorum fragmenta* (1922), Rose *Aristoteles pseudepigraphus* (1863) | Reuse matching (shingles), as now | **Lace** high-quality Greek OCR of these exact volumes (Bruce Robertson, Mount Allison) | Lace's host `heml.mta.ca` did not answer today; the Digital Classicist wiki points to a Wayback snapshot of its index (2025-01-15). Whether the run files themselves are retrievable from the Wayback Machine is **(unverified)**. Fallback: the archive.org `_djvu.txt` of the same scans (all listed above have one), quality unknown until sampled. |
| **B. Known loci** | Source lines from *FGrHist* I, IIA, IIB and commentaries (1923–30); DFHG source fields (have); source lines from every Tier A collection | Deciding "already collected" by passage, independent of wording or OCR quality | Vanderbilt scans on archive.org, e.g. `30081009156641`, `30081002444507`, `30081009156740` (each with a 1.1–1.7 MB `_djvu.txt`); three more Vanderbilt items have no text file | Checked: open, not lending-restricted. **Do not** use the 1954–63 reprints (`diefragmentederg0000jaco` etc.): those are controlled-lending items. |
| **C. Manual check** | BNJ, TLG, Kassel–Austin *PCG*, Snell–Kannicht *TrGF*, Staab, Reuss, CAG indices | Final novelty check on a short list | The user, by hand, through a library login | Allowed by the brief's constraints; budget ≤10 leads per batch. |

**Copyright note (not legal advice).** *FGrHist* I–II and every Tier A edition listed were published before 1931, so they are public domain in the US as of 2026. *FGrHist* III (1940–58, local history and ethnography, numbers 262–856) is not, and Jacoby (d. 1959) is protected in the EU until 2030. So Tier B covers Jacoby nos. 1–261 only. Aristodemus (*FGrHist* 383, if the pilot's guess is right) falls outside it.

**Why passages, not texts.** The pilot's false "new" leads came from three sources: an incomplete baseline, a messy source field, and OCR that breaks the wording. A locus table fixes the second and third: it doesn't care whether our OCR or theirs garbled the Greek. It also reads cheaply. The source lines are in Latin script, which the bad OCR got right.

**Building it** (a method sketch, not code to write now):
- Parse `<witness abbreviation> <book/line/page>` from each fragment heading, using a hand-made table of the abbreviations Jacoby and Müller use (`SCHOL. GEN. HOM.`, `STEPH. BYZ. s. v.`, `ATHEN.`, `HARPOKR.`, `SCHOL. APOLL. RHOD.` …), probably 150–300 entries.
- Map each witness's citation scheme onto the scheme of our text (Dindorf page vs. book and line, Meineke page vs. lemma). This mapping is the hard, slow part; do it only for witnesses we search.
- **Measure:** parse precision on 100 random source lines, checked by hand (target ≥90%), and coverage: the share of the pilot's DFHG-sourced reuse hits whose locus the table contains (target ≥80%).

### 2. Which texts, in order

Selection rule: open and clean (or OCR that measures well), cites many lost authors, and *less* read by editors than Athenaeus, the Suda, Stobaeus or Photius. Each step also states its job: **calibrate** (we expect to find only known material, and we use that to measure), or **discover**.

| # | Text | Open text? | Job | Why | Likely to hold uncollected material? |
|---|---|---|---|---|---|
| 1 | **Scholia on Euripides** (`tlg5023`, 8 files), **Sophocles** (`tlg5037.tlg004`), **Homer** (`tlg5026.tlg001` Iliad, `.tlg007` Odyssey) | **Yes**, First1KGreek TEI. Which editions (Schwartz? Dindorf?) **(unverified)**; check the TEI headers. | Calibrate | The genealogists and mythographers of *FGrHist* I are cited mostly in exactly these scholia, so Tier B covers them well. This is where we measure "rediscover Jacoby's additions to Müller" (Q6). | Low for new fragments; moderate for new parallels. |
| 2 | **The less-read Aristotle commentators**: Michael of Ephesus (`tlg4034`, Parva Naturalia, Parts and Movement of Animals, EN V), Philoponus on *Meteorology* I and *Generation of Animals* (`tlg4015.tlg005`, `.tlg007`), Olympiodorus on *Meteorology* (`tlg4019.tlg003`), Asclepius on *Metaphysics* (`tlg4018.tlg001`), Eustratius on EN I (`tlg4031.tlg002`), Themistius's paraphrases (`tlg2001`). Simplicius and Alexander (`tlg4013`, `tlg0732`) as calibration. | **Yes**, First1KGreek TEI (titles checked from each work's `__cts__.xml`). | Discover | Commentators quote lost Aristotle (exoteric works, *De philosophia*), Theophrastus, Eudemus, Strato, Alexander's lost commentaries, Porphyry, Stoics. Simplicius on *Physics* has been combed for the Presocratics; Michael of Ephesus and Philoponus on biology and meteorology much less. | **Best bet in open text.** Modern collections (FHS&G Theophrastus, Wehrli, Gigon's Aristotle) are copyrighted, so final checks are manual, and some of these will turn out to be collected. |
| 3 | **Cramer, *Catenae Graecorum Patrum in Novum Testamentum*** (`tlg4102`, 32 works: Matthew to Jude) | **Yes**, First1KGreek TEI. Scans also on archive.org (`catenaegraecorum01cramuoft` etc.). | Discover | Catena items are labelled with their author (Ὠριγένους, Ἀπολιναρίου, Θεοδώρου, Διδύμου, Σευήρου …). Many of those authors' commentaries are lost. The author lemma *is* the citation, so the name-collision problem mostly disappears. | Moderate-low. Paul's catenae were collected by Staab (1933), Matthew and John by Reuss (1957, 1966); all copyrighted, so checks are manual. Good for lost Christian works (Q3). |
| 4 | **Cramer, *Anecdota Graeca* Oxoniensia (4 vols, 1835–37) and Parisiensia (4 vols, 1839–41)** | archive.org OCR (e.g. `anecdotagraecae01cram`, `anecdotagraecaec01cram`–`04cram`); quality **(unverified)** | Discover | First and often only editions of Byzantine lexica, epimerisms, scholia and chronicles. Some items were never re-edited. | Moderate, and the least predictable. Needs an OCR quality check first (Q7). |
| 5 | **Eustathius, Commentaries on the Iliad and Odyssey** (Leipzig 1825–30) | archive.org OCR (`commentariiadho01eust`, `commentariiadhom01eust` and companions); quality **(unverified)**. Van der Valk's edition is copyrighted. | Discover + method test | Eustathius had an unabridged Athenaeus and the Atticist lexica (Aelius Dionysius, Pausanias). Their fragments are collected by Erbse (1950, copyrighted) and Schwabe (1890, **Lace has it**). Also the ideal paraphrase test, because his source Athenaeus is clean TEI (see Q4). | Moderate for lexicographic fragments. |
| 6 | **Stephanus of Byzantium, *Ethnika*** (Meineke 1849) | archive.org OCR, three scans (`stephanibyzanti00meingoog`, `bub_gb_0NIPAAAAQAAJ`, `bub_gb_O70UAAAAQAAJ`). No open digital text of Billerbeck. | Calibrate | Cites hundreds of lost historians and geographers. Jacoby III mined it, so it is not a discovery target, but it is a large precision test for the citation finder. | Low. |
| 7 | **Tzetzes**: *Chiliades* and the Lycophron scholia (Scheer 1881/1908, `lycophronisalexa02lycouoft`) | archive.org OCR | Discover | Late, cites lost works (often second-hand) and was less systematically excerpted. | Low-moderate; many citations will be unreliable or second-hand. |
| 8 | **Latin compilers**: Servius (including Servius Danielis), Macrobius, Gellius, Nonius | Latin is this repo's home ground; Perseus and Latin Library texts. Exact open TEI per author **(unverified)** | Discover, cross-lingual | Quote lost Latin works directly and lost Greek works in translation. Open Latin baselines exist: Peter *HRR* (1906–14), Funaioli *GRF* (1907), Ribbeck (1897–98), Baehrens *FPR* (1886), all pre-1931 **(archive.org availability unverified)**. | Low-moderate; heavily mined, but the cross-lingual (Greek quoted in Latin) layer is less so. |

**Considered and not chosen now**
- **Constantinian *Excerpta*** (de Boor 1903–05, Büttner-Wobst and Roos 1906–10, Boissevain 1906; partly on archive.org): they *are* the source of most fragments of Priscus, Malchus, Dexippus and others, so everything in them is collected. Use them only as a recall test.
- **Photius, *Bibliotheca*** (PG 103–104 OCR): fully studied.
- **Etymologica**: Gaisford's *Etymologicum Magnum* (1848) was not found on archive.org by title and author search **(unverified)**; the *Genuinum* is only partly edited (Lasserre–Livadaras, copyrighted).
- **PTA** (`PatristicTextArchive/pta_data`, branch `public`, CC BY/BY-SA): 23 text groups and about 1,000 XML files, including five diplomatic transcriptions of catena manuscripts (`pta0001.pta001.pta-MsCat*`). Small, but it is open manuscript catena text and worth one look once step 3 works.

### 3. Beyond historians

Yes: change the main target, and keep historians as the calibration domain because we already have their answer key.

| Target | Where quoted | Open baseline | Verdict |
|---|---|---|---|
| Lost philosophy (Aristotle's lost works, Theophrastus, Eudemus, Stoics, Presocratics) | Aristotle commentators | Diels 1912/1922, Rose 1863/1886, von Arnim, Mullach, Diels *Doxographi* (1879): all PD, most on Lace or archive.org | **Primary target.** Best ratio of open baseline to under-read text. |
| Lost Christian works (Origen, Apollinaris, Didymus, Theodore of Mopsuestia, Papias, Hegesippus) | Catenae, Eusebius | Weak: Routh's *Reliquiae Sacrae* (1846–48) is open but old; the real collections are copyrighted | **Second target.** The novelty test leans on the "not in extant works" filter (Q4). |
| Lost poetry (comedy, tragedy, epic) | Lexica, scholia, Eustathius | Kock, Nauck, Kinkel: open but superseded by *PCG* and *TrGF* (copyrighted) | Same problem as DFHG versus BNJ: an old baseline overstates novelty. Run it, but expect most "new" leads to be in Kassel–Austin. |
| Lost historians | Everywhere | DFHG + *FGrHist* I–II loci | Calibration and evaluation, not the main discovery target. |

### 4. Method changes

Keep the shingle matcher; recall is not the problem. Add the following, each with its own measurement.

| Change | Why | How to measure |
|---|---|---|
| **Known-loci check** (Tier B) on every lead | Removes known fragments whose wording our OCR or the baseline OCR broke | Share of pilot leads it removes that a human agrees were known (target: most of the pilot's 224 non-new leads, false removals ≈0 on a hand-checked 50) |
| **Extant-works filter**: index the cited author's surviving works (First1KGreek and Perseus TEI; PG OCR for Fathers) and drop citations that match them | Commentators mostly quote *extant* Aristotle and Plato; catenae mostly quote extant Chrysostom and Cyril. Without this, steps 2 and 3 drown. | Share of citation leads removed; spot-check 50 removed ones for false removals |
| **General author-name list**: build it from the Perseus catalog (TLG author IDs and names, open), not DFHG, plus work titles (ἐν τῷ Περὶ …, ἐν τοῖς Μετεωρολογικοῖς) | DFHG names cover historians only. A work title beside a name is much stronger evidence than φησί. | Precision of citation leads by formula type (verb vs. book vs. title), on step 1 |
| **Catena lemma reader**: treat the catena's author label as the citation | The label is the attribution; no formula search needed | Hand-check 50 labels for correct segmentation |
| **Lemma shingles** as a second index: the same 3-gram matcher over lemmas (CLTK or OdyCy; CLTK is already in the repo for scansion) | Catches quotations with changed inflection and light paraphrase | **Paraphrase test set:** Eustathius against Athenaeus (Eustathius rewords Athenaeus constantly, and both texts are available). Compare surface vs. lemma recall on passages Eustathius credits to Athenaeus. Shuffled control must stay ≈0. |
| **Cross-lingual rerank**, optional: SPhilBERTa (Riemenschneider and Frank, 2023: Greek–Latin–English sentence embeddings built for this task; model size and licence **(unverified)**) | Latin quoting Greek in translation (step 8) | Known Latin translations of Greek fragments (e.g. Cicero's versions of Aratus or Plato) as positives; precision@20 |
| **Second-witness clustering**: group leads that are the same text in several witnesses | A passage already in the Suda and Athenaeus is a parallel, not a find | Number of leads collapsed |
| **Model triage**: keep "translate, then say who it is", add "quotation / paraphrase / report / mention" and "is the claim substantive" | The pilot showed the order of the prompt matters (0/7 vs. 7/7) | Re-run the 102-item hand set after each prompt change; add 50 commentator and 50 catena items to it |
| **Stats for each run**: leads, after each filter, model-ranked top 20 hand-checked | So every step is a measured change, not an impression | One table per text in this doc |

**Arabic**: still out of scope. Note for later: much of the Arabic philosophy tradition (lost Alexander, Galen, Themistius) has open-ish editions only in print, and needs Arabic OCR plus a cross-lingual matcher. Do not start it before step 2 shows the pipeline produces reviewable leads at all.

### 5. Scoring novelty: a lead worth a human's time

A lead goes to the user only if it passes every item below. Items 1–8 are automatic, 9–10 are quick human checks before the expensive BNJ/TLG check.

1. **Identity**: the cited name is resolved to one author: model says "same person: yes" *and* the name is not on the ambiguous list, *or* an epithet, ethnic or work title disambiguates it.
2. **Formula**: a citing formula is present; a work title or book number counts more than a bare verb. (Catena lemmas count as a title.)
3. **Locus is not known**: the passage is not in the known-loci table (DFHG + *FGrHist* I–II + Tier A collections).
4. **Wording is not known**: the window shares less than about 30% of its words, as 3-grams (surface and lemma), with any baseline fragment.
5. **Not extant**: the window does not match the cited author's surviving works.
6. **Not a parallel**: no second witness in our index already carries the same passage with a credited source.
7. **Clean text**: at least 90% of the window's words are valid Greek words (dictionary built from First1KGreek and Perseus), or the passage has been checked against the page image.
8. **Substance**: at least about 12 words of reported content beyond the name and formula, and the model tags it "quotation", "paraphrase" or "report", not "mention".
9. **Editor's note**: a human looks at the printed page (scan) for a footnote or margin note already identifying it ("fr. 12 M.", "= FGrHist …").
10. **Intermediary**: a human notes whether the citation could come via a known intermediary (Didymus, Athenaeus, Porphyry). That doesn't disqualify it, but it changes where to look.

Then the user checks BNJ, TLG or the relevant modern collection, and records the result as a verdict in the Leads tab with the reference. **Budget:** at most 10 such leads per batch; if a batch of 10 yields none that are even uncertain, stop and fix the filters before the next batch.

### 6. Evaluation: an answer key for precision of "new"

We now have three, from strongest to weakest:

1. **Rediscover Jacoby's additions to Müller.** Baseline = DFHG only. Answer key = *FGrHist* I–II loci that DFHG lacks for the same authors. Run on the scholia (step 1). Report: of the top-k "new" leads, how many are Jacoby additions (true "new to this baseline"), how many are known to DFHG under a misparsed source (pipeline error), and how many are junk (collisions, OCR, mentions). This is a real precision@k for discovery, on real data. **Gate:** if precision@20 is below about 25%, fix filters before searching discovery texts.
2. **Leave-one-collection-out.** Remove one Tier A collection (e.g. Kock) from the baseline, run on a text that quotes it (Athenaeus, which also lets us reuse the 97% recall), and count how many held-out fragments come back as top-ranked "new" leads and how much junk ranks above them. This measures recall of novelty and ranking quality.
3. **Hand-reviewed set**: the existing 102 citations, grown with commentator and catena items, for the model-triage step only.

Report all three in this file after each phase.

### 7. OCR: correct it or find clean text?

**Prefer clean TEI wherever it exists**, and order the work so it comes first: steps 1–3 are all clean First1KGreek TEI. For OCR-only texts (steps 4–7):

1. **Measure before correcting.** Score each archive.org `_djvu.txt` by the share of tokens that are valid Greek words. The Vanderbilt *FGrHist* sample shows the failure to look for: Greek read as Latin letters (`onaodtzei xal To dLaLgedEv`), which scores near 0 and cannot be corrected.
2. **If the Greek was read as Latin**, re-OCR the page images instead (Tesseract `grc`, or a Kraken polytonic-print model in the repo's Kraken venv **(model availability unverified)**). Page-image downloads are large (often hundreds of MB per volume), so **ask the user first**.
3. **If the Greek OCR is merely noisy** (like the pilot's Stobaeus and Photius, 64–81% recall at 50% coverage): lower the threshold to 30% (82–85% in the pilot), rely on the known-loci check (which ignores wording), and add lemma shingles. Build a Greek correction pass only if recall at 30% stays below about 80%. The Latin long-s pipeline does not transfer directly; a Greek version would be dictionary-plus-confusion-pairs (ο/σ, ι/l, ε/c …).

---

## Phased plan

Every phase ends with numbers in this document. A phase that fails its gate stops the next one.

| Phase | Work | Downloads (ask first if large) | Measure | Gate to continue |
|---|---|---|---|---|
| **0. Close the pilot** | User checks the six pilot leads in BNJ and records verdicts. Fetch the 21 DFHG authors that failed (Polybius, Memnon, Heraclides Ponticus, Aristodemus …). | DFHG API, small | Verdicts recorded | none |
| **1. Known-loci table** | Parse *FGrHist* I–II source lines (Vanderbilt `_djvu.txt`) and DFHG source fields into one table; abbreviation map for the witnesses we search | About 6–10 MB of text; ask before fetching | Parse precision on 100 lines (≥90%); coverage of pilot's DFHG reuse hits (≥80%) | Both targets met |
| **2. Calibrate on scholia** | Run Euripides, Sophocles and Homer scholia (TEI) with the general name list and the loci check; evaluation 1 (Jacoby additions) | First1KGreek TEI, a few MB | Precision@20 of "new" against Jacoby additions; leads removed per filter | Precision@20 ≥ about 25% |
| **3. Philosophy** | Add Tier A philosophy baselines (Lace or archive.org OCR); extant-works index for Aristotle, Plato and Theophrastus; run the commentators, Simplicius and Alexander first as calibration | Lace/OCR text ~10–30 MB; ask | Leave-one-collection-out with Diels or Rose; leads passing the checklist per 100k words | At least some leads pass items 1–8 and survive item 9 |
| **4. Catenae** | Catena lemma reader; extant-works index for Chrysostom, Cyril, Theodoret (First1KGreek, PG OCR); run `tlg4102` | First1KGreek TEI | Share removed by extant filter; checklist pass rate | Same as above |
| **5. OCR'd Byzantine reference works** | OCR quality score; Eustathius vs. Athenaeus paraphrase test of lemma shingles; then *Anecdota*, Eustathius, Stephanus, Tzetzes | archive.org text, ~5–20 MB each; page images only with the user's OK | Surface vs. lemma recall on the paraphrase set; valid-word ratio per volume | Lemma matching beats surface without breaking the shuffled control |
| **5b. Latin compilers** (optional, can run in parallel) | Latin baselines (Peter, Funaioli, Ribbeck); Servius, Macrobius, Gellius; cross-lingual rerank as an experiment | Small | Same checklist | — |
| **6. Decision: manuscripts and Arabic** | Not work, a decision. Only if phases 3–5 produce leads that survive the checklist does it make sense to build Greek minuscule HTR for unedited catena manuscripts (PTA's transcriptions could seed training), or Arabic | — | — | User decides |

## What to stop doing

- **Searching the most-mined texts for discovery.** Athenaeus, the Suda, Stobaeus and Photius's *Lexicon* stay as calibration and recall tests only. Don't add more texts of that kind (Diogenes Laertius, Plutarch, Clement, Eusebius's *Praeparatio*) for discovery.
- **Calling a lead "new" because DFHG lacks it.** Label it "not in baseline", and name the baseline.
- **Adding citation formulas for medieval Latin and Greek in general.** The `corpus.db` probe showed the formulas pick up "Philosophus", place names and scripture. Formulas work only with a resolved author list and a work title.
- **Mining `corpus.db` for this.** It holds almost none of the target genres. Keep quotation work in `data/leads.db`.
- **Model-triaging every lead.** Run the automatic filters (checklist items 3–7) first and spend the GPU on what is left.
- **Treating OCR texts like clean ones.** Lower the threshold, use the loci check, and score OCR quality before a run, not after.
- **Using lending-library copies** (archive.org `inlibrary`/`printdisabled` items, such as the 1954 *FGrHist* reprints): they are not open, even though their text files show up in search.

## Uncertainties, stated plainly

- I could not confirm that Lace's OCR files are retrievable from the Wayback Machine, which First1KGreek editions the scholia files use, whether Gaisford's *Etymologicum Magnum* is on archive.org, or the size and licence of SPhilBERTa.
- I did not sample the OCR quality of Cramer's *Anecdota*, Eustathius, Stephanus or Tzetzes. Phase 5 starts by doing that.
- Which Vanderbilt items correspond to which *FGrHist* volumes is inferred from three page samples (Pherecydes in vol. I, "Universal- und Zeitgeschichte" in II A, Apollodorus no. 244 and the Zeittafeln in II B). Three items have no text file. Phase 1 should list all six with page counts first.
- How much of the commentators and catenae is already in copyrighted collections is the main unknown, and only the manual checks can answer it. The plan is built so that we learn this from a few batches of ten rather than from a large build.
- Public-domain status is a US reading of publication dates, not legal advice.

## Sources checked

- First1KGreek and Perseus `canonical-greekLit` file trees (GitHub API), and each candidate work's `__cts__.xml` title
- [PTA data repository](https://github.com/PatristicTextArchive/pta_data) (branch `public`), and the [PTA project page](https://pta.bbaw.de/project/about)
- archive.org search and metadata API for the editions named above, and the [Vanderbilt *FGrHist* item](https://archive.org/details/30081009156641)
- [Lace: Greek OCR](https://wiki.digitalclassicist.org/Lace:_Greek_OCR) (Digital Classicist wiki), and [AWOL's list of Lace fragment-collection OCR](https://ancientworldonline.blogspot.com/2015/11/fragmentary-texts-ocr-from-the-leipzig.html)
- [Trismegistos entry for Stephanus's *Ethnika*](https://trismegistos.org/authorwork/369) (editions; no open digital text of Billerbeck found)

---

## Phase 1 results (2026-11-11)

Built: `ingest/known_loci.py` (parser and passage mapping), `scripts/build_known_loci.py` (download, build, `--measure`), checks in `scripts/quotations_selftest.py`. The table is `data/quotations/known_loci.jsonl`: 13,708 rows, 5,688 from *FGrHist* and 8,020 from DFHG. The scan text is in `data/raw/quotations/fgrhist/` with a private-use notice. Both are gitignored and not to be published.

**Sources used.** These are archive.org scans that anyone may download, not lending-only copies:

- the Vanderbilt text volumes I (1923), II A (1926) and II B (1929);
- from Claremont, III B (1950; authors on single cities, nos. 297–607) and the 1957 reprint of I, whose OCR reads more source lines.

III B and the 1957 reprint are in copyright and are used privately under the no-publication notice. III A and III C exist only as lending-only copies, so they aren't used, and neither are the "vols 1–8" uploads, which look unauthorized. Claremont's other open items are commentary volumes with no source lines.

**Parse precision: target met.**

- *Parts I–II:* 95 of 100 randomly sampled lines had the right source line and author. The 5 errors were Greek book titles that the OCR rendered in Latin capitals.
- *Part III B:* all 30 sampled lines that carry a passage were right. Over all lines, 43 of 50 were right; the misses are critical-apparatus lines, which carry no passage.

Changes for part III:

- Author heads can have a date after the name.
- Running heads that span several authors are skipped.
- A numbered line counts as a new author only if its number is in sequence. This stops book titles and source lines with a dotted number from being taken as authors.
- An Athenaeus passage is taken only from a line that begins "ATHEN. book, chapter p. page letter", never from apparatus.

**Coverage: met only with DFHG included.** Share of fragments DFHG credits to a text, found by the matcher, whose passage is in the table:

| Text | *FGrHist* I, II, III B alone (its own authors only) | DFHG + *FGrHist* |
|---|---:|---:|
| Athenaeus (Casaubon page, ±1 section) | 75% of 501 | 98% of 935 |
| Pindar scholia (ode, poem, ±2 lines) | 61% of 93 | 89% of 118 |
| Suda (headword) | 11% of 66 | 21% of 189 |

- *FGrHist* alone falls short mainly because the OCR loses source lines. Volume I yields only about a third of Hekataios's ~380 fragments.
- Suda headwords are Greek, so *FGrHist*'s OCR garbles most of them. DFHG often gives just "Suidas:" with no headword.
- The Pindar numbers are approximate: FHG cites Boeckh's lines, our text uses Drachmann's scholion numbers.

**Effect on current leads.** A passage counts as known only *for the same author*. Six Athenaeus citation leads sit at a passage collected for *another* author on the same page, so "any author" would be wrong. Of the current leads, 1 Pindar citation and 1 Suda reuse lead are at a passage collected for their author. Part III changed neither number. The Pindar one is a real catch: the scholion on *Nem.* 10.114, which DFHG's name matching gave to Apollodorus of Artemita, is Jacoby's Apollodorus of Athens, *FGrHist* 244 F 148.

**The main finding: Jacoby adds little beyond Müller for these texts, even with III B.** Passages in *FGrHist* but not in DFHG:

| Text | *FGrHist* I–II | *FGrHist* I, II, III B |
|---|---:|---:|
| Athenaeus | 20 | 40 |
| Pindar scholia | 7 | 23 |
| Suda | 0 | 21 |

That's about 85 passages of "collected after Müller", still small for phase 2's planned top-20 precision measure, and lopsided towards Athenaeus. The rest of Jacoby's additions are in III A, III C and the continuation (IV), none of which is openly available.

**Gate.** Precision is met. Coverage is met for Athenaeus and the Pindar scholia only with DFHG included, and not for the Suda. Recommendation for phase 2:

- Use the ~85 passages as a small answer key, and report precision with its confidence interval rather than as a single number.
- Add Suda headwords from the fragment texts themselves (most Suda fragments begin with the headword), which needs no new source.
