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

## Phase 2 results (2026-11-12)

**Gate met: precision@20 is 55% (11 of 20) against the 25% target**, but only with a ranking the plan didn't specify (below), and on a small answer key.

**What ran.** These are all First1KGreek TEI, now registered as witnesses:

- Euripides scholia: Schwartz 1887, nine plays.
- Sophocles scholia: Papageorgius 1888, seven plays.
- Iliad scholia (Dindorf and Maass) and Odyssey scholia (Dindorf).

Dindorf's separate Euripides files repeat plays Schwartz covers and are not used. Passages are mapped to play and line, or to poem, book and line. The Homer verse is taken from each scholion's lemma ("307. …"), because the TEI's section numbers are running counts, not lines.

**Known-passages table** for these texts: Jacoby's Homer books don't survive OCR, so they match any book. Share of fragments Müller credits to each text whose passage is in the table (DFHG + *FGrHist*):

| Text | Coverage |
|---|---:|
| Euripides scholia | 91% |
| Sophocles scholia | 100% |
| Homer scholia | 77% |

Line tolerance is ±10 for the tragedies, because Müller cites older line numbers 5–20 off Schwartz's, and ±3 for Homer. Most of Homer's misses are wrongly divided books, in Müller or in the TEI.

**Evaluation 1** (`scripts/evaluate_novelty.py`): the baseline is Müller only. A lead counts as a Jacoby addition if Jacoby prints a fragment of that author at that passage and Müller doesn't. "Judged" leads are those whose author is in our *FGrHist* volumes (I, II, III B).

**Answer key.** Jacoby has 76 additions in these scholia by authors the finder can name. How many each version of the finder recovers:

| Step | Named | Survive as leads |
|---|---:|---:|
| Pilot finder (first run) | 28 | 3 |
| + 4-letter and "-er" name stems (Ephoros, Istros, Timaios); formulas "ἡ ἱστορία παρὰ X", "ὡς X", "X δέ", "ἐν τρίτῳ", γράφων/γενεαλογεῖ/ἱστοροῦσι | 45 | 3 |
| + wording checked against the cited author's fragments only | 45 | **30** |

The 31 still unnamed are mostly scholia not in our editions at the passage Jacoby cites. These are probably the Homeric D-scholia, which Dindorf doesn't print, or differently numbered notes.

**Each filter, on the 1,131 citations found:**

| Filter | Removed | Jacoby additions among them |
|---|---:|---:|
| Nearby wording in Müller for the *cited author* (≥10%) | 175 | 20 (Müller has the author's wording here under a passage cited differently, so arguably known) |
| Name is a character of the commented work (Menelaus, Antenor, Idomeneus…), from the Perseus texts of the Iliad, Odyssey and plays | 126 | 0 |
| Passage in Müller for this author (known-passages table) | 19 | 0 |
| Left: new to Müller | 811 (720 judged, 32 of them Jacoby additions, 4.4%) | |

Checking wording against *all* of Müller, as the pilot did, leaves only 3 of the 76 additions. A scholion citing two historians counts as "known" if Müller has either one's fragment from it.

**Ranking** (precision among judged leads):

| Ranking | P@20 | P@50 | P@100 |
|---|---:|---:|---:|
| Local model (qwen3:14b, "same person?" then quotation type) | 0/20 | 1/50 | 2/100 |
| Citing formula ("in book N" > "the story is in" > "says" > "as" > "X but") | 1/20 | 2/50 | 3/100 |
| **Müller-only prior**: fragments Müller credits to this text for the author, per lead of that author | **11/20** | **22/50** | **29/100** |

- **The model can't tell homonyms apart here.** It answers "same person: yes" for Zenodotus the Homeric editor as the historian Zenodotus of Troezen, and for Porphyry's *Homeric Questions*. Its "yes" leads are Jacoby additions 4.8% of the time (25 of 519), against 3.4% (5 of 148) for its "no". Sorting all 830 scholia citations took about 80 minutes on the RTX 4060 Ti, for no gain.
- **Formula type doesn't predict a real addition either.** "In book N" gives 4 of 71, "says" 21 of 464.
- **Why the prior works:** a historian a text really quotes already has fragments from it in Müller. Pherecydes has 33 from the Homer scholia and 10 leads. A homonym cited constantly for something else has none: Zenodotus has 0 credited and about 200 leads.

**Caveats.**

- The answer key is small (32 additions among the judged leads), so 55% has a wide margin, roughly 30–75%.
- The top 20 is nearly all Pherecydes and Hellanicus. The prior ranks authors, not passages, and Jacoby's additions in these scholia are concentrated on exactly those genealogists.
- Some "misses" in the top 20 are probably real: Pherecydes at Il. 14.323/324 is one scholion counted twice, and Jacoby's OCR loses source lines. So the true precision is likely somewhat higher.
- In discovery texts, where Müller credits few fragments, the prior has little to go on. It should be combined with something else there, not relied on alone.

**What changed in mining** (`scripts/mine_quotations.py`, all texts): wording is checked against the cited author's fragments, character names of the commented work are dropped, and there are new formulas and short stems. Re-mining the older texts will produce more leads than before.

**Next.**

1. ~~Put the prior into the Leads tab's sort.~~ Done 2026-11-12. Mining stores the prior with each citation lead (`ingest.quotations.author_priors`). "Most promising" now orders citations by it, and the card says how many fragments Müller credits to the text for that author. The model's ordering is still available as "model's rank". The prior is per text: Athenaeus's values are much larger than the scholia's, so compare leads within one text.
2. Mark leads at a known passage in the Leads tab rather than dropping them.
3. Refetch the 21 DFHG authors that failed (phase 0). Castor and Eratosthenes are among Jacoby's additions the finder can't name.
4. For phase 3, the identity problem gets harder: commentators cite Aristotle, Plato and Theophrastus constantly. Expect to need the extant-works filter, not just a prior.

## Phase 0 note (2026-11-12)

The 21 DFHG authors that "failed" can't be fetched: the API's author list (613 names) doesn't include them. They are Polybius, Memnon, Diodorus, Dionysius of Halicarnassus, Heraclides Ponticus, Priscus, the Marmor Parium and others. The one listed name not yet downloaded, Leroubna, returns an empty response. The six pilot leads still need your BNJ check.

## Phase 3 results (2026-11-12)

**Gate: not shown to be met.** Items 1–8 are automatic and 76 discovery leads pass them. Item 9, the printed page, is a human check. A skim of the strongest candidates finds commentators retelling Aristotle's own reports and other known testimonia, not uncollected fragments. By the plan's rule, phase 4 waits on that check.

**Baselines.** Only collections with usable Greek OCR can serve. Valid-word share, measured against the clean TEI already downloaded:

| Collection | Source | Valid Greek words |
|---|---|---:|
| Wimmer, Theophrastus, *Fragmenta varia* (1866) | First1KGreek TEI | clean |
| von Arnim, *SVF* I–II | archive.org OCR | 88–90% |
| Diels, *Doxographi Graeci* | archive.org OCR | 79% |
| Rose's Aristotle fragments, in the Berlin Aristotle vol. V (1870), lines 100–16,590; the rest of the volume is Rose's indexes and Bonitz's *Index* | archive.org OCR | 55% |

Rose 1886, Diels's *Vorsokratiker* (every edition), *SVF* III and Usener's *Epicurea* have no Greek in archive.org's OCR: it was read as Latin letters. Lace's re-OCR (heml.mta.ca) is offline, and the Wayback Machine kept only training samples. Getting those collections means re-OCRing page images, which needs your OK for the downloads.

**Texts** (First1KGreek TEI, now witnesses):

- Calibration: Simplicius (Physics, De caelo, Categories, De anima) and Alexander (Metaphysics, Topics, Prior Analytics, De sensu), 1.72 million words.
- Discovery: Michael of Ephesus, Philoponus on *Meteorology* I and *Generation of Animals*, Olympiodorus on the *Meteorology*, Asclepius, and Eustratius, 0.83 million words.
- Surviving works for the extant filter: Aristotle (41 works), Theophrastus (14), Plato (36).

Code: `ingest/philosophy.py` (baselines, the 20 philosophers' Greek name stems, lost and extant titles) and `scripts/mine_philosophy.py` (`evaluate`, `run`).

**Extant filter.** "Nearby wording shares ≥15% of 3-grams with the surviving works" is useless on commentaries, because they share phrases with Aristotle everywhere. It kept 0 of the 9 Aristotle citations at passages Rose prints. Requiring an *unbroken run* of 8+ words from a surviving work kept 8 of 9 and removed 76 quotations of extant Aristotle. For Theophrastus it kept all 20 known passages.

**Evaluation 2: hide a collection.** Hide the collection, mine the calibration texts, and count leads at passages the hidden collection prints:

| Hidden | Citations | Leads | At hidden passages | P@20 | P@50 |
|---|---:|---:|---:|---:|---:|
| Wimmer only (Theophrastus) | 62 | — | — | — | — |
| Wimmer + *Doxographi* (no Theophrastus collection at all) | 62 | 60 | 20 (33%) | 6/20 | 17/50 |
| Rose (Aristotle) | 463 | 308 | 7 (2.3%) | 4/20 | 5/50 |

- **Wimmer only:** Diels's *Doxographi* still catches 13 of the 19 hidden passages. The collections overlap heavily for Theophrastus in Simplicius.
- **Theophrastus:** a third of the leads are passages Wimmer or Diels print. Many of the rest are probably in the modern edition (FHS&G, copyrighted), e.g. Alexander on Theophrastus's modal logic. So 33% is a lower bound. Ranking adds nothing beyond that.
- **Aristotle:** the only strong signal is a lost-work title beside the name. 4 of the 9 such leads are Rose passages, against a 2.3% base rate. Caveat: I added *On Opposites* (Περὶ ἀντικειμένων) and the crasis form of *On the Good* (περὶ τἀγαθοῦ) to the title list after seeing the hits. Both are in Rose's catalogue, but P@20 = 4/20 is partly tuned on its own test. Formula type barely helps: "in book…" gives 6 of 78, "says" 1 of 194.

**Discovery run** (0.83 million words; leads are in the Leads tab):

| Checklist step | Leads | Per 100k words |
|---|---:|---:|
| Leads after the baseline and extant filters (items 3–5) | 255 | 30.8 |
| Item 1: name unambiguous (not Zeno, Alexander, Strato, Heraclides, Boethus, Andronicus) | 210 | 25.4 |
| …and the author's works aren't all extant (not Plato) | 80 | 9.7 |
| Item 2 formula, item 7 clean TEI text, item 8 full window | 76 | 9.2 |

- By author: Aristotle 65, Porphyry 6, Speusippus 3, Aristoxenus 1, Xenocrates 1.
- 17 of them name a work.
- A skim of those 17 and of every Speusippus and Xenocrates lead finds:
  - Porphyry's *Isagoge*, which survives but isn't in the extant index yet.
  - An example sentence ("τὸ μουσικὸν ἐν τῷ Ἀριστοξένῳ").
  - Reworded references to surviving Aristotle ("ἐν τῷ περὶ βροντῆς … λόγῳ" is *Meteorology* II).
  - Testimonia the commentators take from Aristotle himself.
- No lost-title lead turned up in the discovery texts.

**What this says.**

- The machinery transfers: the baseline filter, the extant filter and lost titles each measurably help.
- But the open baselines are thin. Rose is at 55% OCR, and there is no Diels-Kranz, Wehrli, FHS&G or Smith.
- The less-read commentators cite few lost works by title. Most of what remains are paraphrases of extant works, which only a reader (or a much better model) can tell from a fragment.

**Before phase 4, in order of value:**

1. Your check of the 17 work-naming leads (item 9). They're in the Leads tab under these texts.
2. Re-OCR Rose 1886 and Diels's *Vorsokratiker* from page images (Tesseract `grc` or Kraken). This needs a few hundred MB of images, so it needs your OK.
3. Add Porphyry, Alexander and Plotinus to the extant index.
4. Have the model tag "reference: quotation/paraphrase/mention" on these leads, measured against the evaluation above before trusting it.

### Phase 3b: Rose and Diels re-OCR'd (2026-11-12)

`scripts/reocr_scans.py` re-read the page images of three archive.org scans with Tesseract 5.3 and the `tessdata_best` Ancient Greek model:

- Rose's Teubner *Aristotelis qui ferebantur librorum fragmenta* (`aristotelisquif00arisgoog`), with the Latin model;
- Diels, *Fragmente der Vorsokratiker*, 4th edition 1922, vols. I–II (`diefragmentederv01diel_0`, `…02diel_0`), with the German model.

That's 1,381 pages, about 40 minutes on 12 CPU workers. Images were fetched one page at a time and discarded; only the text is kept, in the gitignored `data/raw/quotations/philosophy/reocr/`. Models are in `models/tessdata/`.

**Greek now usable.** Valid words: Rose 88%, Diels 86–87%. archive.org's own OCR of the same scans had 0%, and the Berlin printing of Rose had 55%. Both are now baselines (`rose1886`, `diels`). The philosopher list gained 18 Presocratics and sophists.

**Hiding a collection, again** (calibration texts: Simplicius and Alexander):

| Hidden | Citations | Leads | At hidden passages | P@20 | P@50 | P@100 |
|---|---:|---:|---:|---:|---:|---:|
| Diels (19 Presocratics) | 508 | 500 | 69 (13.8%) | 6/20 | 9/50 | 22/100 |
| Rose 1886 + Rose 1870 (Aristotle) | 463 | 308 | 10 (3.2%) | 5/20 | 7/50 | 8/100 |

- **Presocratics:** Simplicius is the main source of their fragments, so this is the strongest answer key so far. The citing formula ranks real passages at about twice the base rate in the top 20.
- **Aristotle:** with the better Rose text, 5 of the 9 lost-title leads are Rose passages, so precision@20 is 25% against a 3.2% base. The finder still names only 13 of the 89 Rose passages it can locate in these texts. Most Rose fragments from commentators don't put "Aristotle says" next to the words.

**Discovery, re-run with Rose 1886, Diels and Porphyry's surviving works** (Isagoge, Categories commentary, De abstinentia, Ad Marcellam, Vita Pythagorae):

| Checklist step | Leads | Per 100k words |
|---|---:|---:|
| Leads after the baseline and extant filters (items 3–5) | 246 | 29.7 |
| Item 1: unambiguous name | 202 | 24.4 |
| …and the author's works aren't all extant | 72 | 8.7 |
| Item 2: book or "says" formula | 66 | 8.0 |

The new collections mark many more citations as known, 126 in Asclepius alone. By author: Aristotle 44, Anaxagoras 4, Democritus 3, Heraclitus 3, Speusippus 3, Empedocles 2, Porphyry 2, Alcmaeon 2, others 1 each.

**Strong candidates** (a lost title, or a named work of a lost author): 3, and none looks new:

- Olympiodorus retelling Aristotle on Anaxagoras's hail (*Meteorology* I.12; Diels prints Aristotle's version as a testimonium).
- Porphyry on accidents in individuals in Asclepius. Probably from his lost *Categories* commentary, but that needs a check.
- The Aristoxenus example sentence.

**Gate, again: not met on present evidence.** The calibration texts now clear 25% at the top for Aristotle and the Presocratics. The open-text discovery commentators, though, produce paraphrases and known testimonia rather than uncollected fragments. The 66 checklist leads are in the Leads tab for item 9.

## Phase 4 results: catenae (2026-11-12)

**Text.** Cramer, *Catenae Graecorum Patrum in Novum Testamentum*, First1KGreek TEI `tlg4102`. It has 24 catenae, not 32: Matthew, Mark, Luke, John, Acts, the Pauline letters and James. 15,745 paragraphs, about 1 million words. Registered as the witness `cramer_catenae`.

**Catena reader** (`ingest/catenae.py`). An item starts at a label: up to 8 words ending in a full stop, at the start of a paragraph or after a full stop, containing an author's name. Accepted around the name:

- titles and sees ("Τοῦ ἁγίου Τίτου Βοστρῶν.");
- a work after ἐκ/ἐν ("Κυρίλλου ἐκ τῶν Θησαυρῶν.");
- nominatives in brackets ("(Θεόδωρος.)");
- for a genitive name, a short garbled tail ("Ἰσιδώρου Πηαοτσιώρου.").

Names and titles match with one letter of slack, for Cramer's misprints and OCR (Κυτίλλου, Ἀποδιναρίου, μονάκου). Inline reports ("Θεόδωρος δέ φησι") are not labels, and unlabelled paragraphs (usually the biblical lemma) get no author.

Result: 3,853 labelled items (354,000 words); 3,704 have 12+ words.

**Label check** (the plan asked for 50): 125 random labels checked in three samples. 124 named the right author. The miss was "Ἰωάννου τοῦ Δαμασκέως", taken as Chrysostom and now fixed. About 200 labels are still missed, mostly with badly garbled sees.

**Authors labelled:**

| Author | Items |
|---|---:|
| Chrysostom | 1,031 |
| Cyril | 437 |
| Theodoret | 373 |
| Theodore (Mopsuestia / Heraclea) | 328 |
| Ammonius | 272 |
| Origen | 222 |
| Severian | 197 |
| Gennadius | 123 |
| Photius | 90 |
| Apollinaris | 76 |
| Anonymous ("ἐξ ἀνεπιγράφου") | 71 |
| Didymus | 67 |
| Severus | 57 |
| Oecumenius | 57 |
| about 15 others | fewer |

**Open texts to check items against** (`scripts/mine_catenae.py`):

- Origen (46 files): the surviving John and Matthew commentaries, plus the editions of his catena fragments on John (Preuschen) and Luke (Rauer).
- Eusebius, Athanasius, Gregory of Nazianzus, Irenaeus and Hippolytus.
- Cyril (*Minor Prophets* only) and Theodoret (the *Histories* only).

**No open clean text:** Chrysostom (not in First1KGreek; the Patrologia Graeca OCR repository has 33 volumes, none of them his), Theodoret on Paul, Cyril on the Gospels and Paul, and every other main catena author. The editions of their catena fragments (Reuss 1957/1966, Staab 1933) are in copyright.

**Calibration: Origen.** Where an open edition covers him, nearly all his items are found (≥50% of the item's words matched together):

| Catena | Origen items | Found |
|---|---:|---:|
| Luke (Rauer's fragments, *Homilies*) | 17 | 16 |
| John (Preuschen's fragments, *Commentary*) | 15 | 11 |
| Matthew | 46 | 0 |
| Romans | 58 | 2 |
| 1 Corinthians | 75 | 0 |

For Luke and John that's 27 of 32 (84%). Matthew, Romans and 1 Corinthians have no open edition of his fragments (Klostermann's GCS; Ramsbotham and Jenkins in *JTS*), so 0 found there says nothing. Across all 3,704 items, 49 are found in an open text, and only Origen's Luke and John items can be called missing.

**Leads: 8, now in the Leads tab.**

- **4 Origen items in the John catena that Preuschen's fragments don't contain** in this wording (coverage 0.15–0.33):
  - ch. 3, on the lifting up of the Son;
  - ch. 5, on "salvation is of the Jews";
  - ch. 16 (two items), on "you shall never wash my feet" and Judas going out into the night.

  They could be passages Preuschen rejected or assigned elsewhere, or wrong labels. Preuschen (GCS 10, 1903) is public domain, so this is a quick check.
- **1 Origen item in the Luke catena** at 0.47, just under the threshold: probably a known fragment that Cramer abridges.
- **3 anonymous items identified:** two are Origen on Luke (in Rauer's collection), and one is Eusebius quoting Josephus from the *Church History*.

**The Patristic Text Archive's manuscript catenae.** Its five `MsCat*` transcriptions are one Severian excerpt (*De fide et lege naturae*) as copied in the *Catena Andreae* on Acts. Cramer prints it: coverage is 0.36–0.72, with the manuscripts' spelling. The comparison works on diplomatic transcriptions, but this sample adds nothing.

**Gate and what it means.**

- Items 1, 2, 7 and 8 are given by the label, the TEI and the item.
- Items 3–5 can only be checked for Origen on Luke and John, and that check gives the 5 leads above.
- Most of what makes catenae valuable (Apollinaris, Theodore, Didymus, Severian, Gennadius, Photius, Ammonius) is printed in Cramer already, and collected by Reuss and Staab, so it isn't new.

A printed catena is itself the edition of its fragments. What's unedited is in catena *manuscripts* Cramer didn't print. That is phase 6's decision (manuscript HTR), and the Origen result shows how such text would be checked: against the open editions, item by item.

## Phase 5 results: OCR'd Byzantine reference works (2026-11-12)

**OCR.** Valid Greek words (3+ letters found in a vocabulary from Athenaeus and the three scholia corpora):

| Text | Source | Valid words |
|---|---|---:|
| Eustathius on the *Iliad* I, III (Leipzig 1827–29) | Tesseract re-OCR (archive.org's own text was 0%) | 83%, 85% |
| Eustathius on the *Iliad* II, IV; *Odyssey* I–II | archive.org OCR | 82–84% |
| Cramer, *Anecdota Graeca* (Oxford I–IV, Paris III and two others) | archive.org OCR | 81–90% |
| Tzetzes on Lycophron (Müller 1811, vol. II) | Tesseract re-OCR | 72% |
| Stephanus, *Ethnica* (Meineke 1849) | Tesseract re-OCR | 63% |

Stephanus scores low mainly because most of its words are place names the vocabulary lacks; samples read cleanly. *Anecdota Oxoniensia* II (55%) is left out. The re-OCR'd text is in the gitignored `data/raw/quotations/philosophy/reocr/` with the other scans, for private use only.

**A tokenizer bug, fixed.** OCR puts free-standing breathings (᾿, ῾, ᾽; Unicode "Sk") before capitals: "᾿Αθήναιος". `norm()` stripped only combining accents, so such a word matched no name stem and no shingle. `tokenize()` now drops them. Effect: the matcher finds 278 of the Stephanus fragments DFHG credits to it (was 220) and 41 in Eustathius (was 34). Phase 2's numbers are unchanged (P@20 11/20, recall 30 of 76). Phase 3b's Diels test moves slightly (P@20 6/20 as before; P@50 7/50 and P@100 19/100, were 9 and 22); Aristotle is unchanged.

A second fix: a name may add at most 3 letters to its stem (was 4), because Callias's stem "καλλι" took Καλλιμάχῳ. Phase 2's numbers are unchanged by this too.

**The paraphrase test (the gate): met, by a small margin.** Code: `ingest/lemmas.py`, `scripts/build_lemmas.py` (Stanza's Ancient Greek lemmatizer, each distinct word once, out of context; the table has 310,591 words), `scripts/paraphrase_test.py`. Test passages: 267 places where Eustathius names Athenaeus (Ἀθήναιος with the accent on the eta, or ὁ Δειπνοσοφιστής); window 10 words before, 60 after; matched against the Athenaeus TEI.

| Coverage ≥ | Surface | Lemma | Shuffled surface | Shuffled lemma |
|---:|---:|---:|---:|---:|
| 0.10 | 38% | 47% | 0% | 1% |
| 0.15 | 23% | 27% | 0% | 0% |
| 0.20 | 14% | 17% | 0% | 0% |
| 0.25 | 8% | 10% | 0% | 0% |
| 0.30 | 4% | 5% | 0% | 0% |

Lemmas find more at every threshold and the shuffled control stays at 0–1%. The gain is small, about a fifth more passages: Eustathius's rewording changes word order and vocabulary, not just inflection. The 7 passages only lemmas find at 0.25 are all real Athenaeus (Anacreon's κελέβη, Archestratus on the sea-boar, the Peloria festival). Lemma matching isn't yet wired into the miner's "wording is known" check, because DFHG's fragments would need lemmatizing first (one more `build_lemmas.py` run).

**Mining** (`scripts/mine_quotations.py run`, leads now in the Leads tab):

| Text | Greek words | DFHG fragments found | Unmatched citations | Per 100k words | By an author Müller credits to this text |
|---|---:|---:|---:|---:|---:|
| Eustathius (with Homer's characters filtered, as for the Homer scholia) | 1.55 M | 41 | 266 | 17 | 18% |
| Stephanus | 156 k | 278 | 294 | 188 | 89% |
| Tzetzes on Lycophron | 99 k | 8 | 48 | 48 | 33% |
| Cramer, *Anecdota* | 825 k | 14 | 175 | 21 | 19% |

Plus 337 "reuse" leads (a DFHG fragment's wording in a text DFHG doesn't credit for it), 148 of them in the *Anecdota*: second witnesses, not new fragments.

What the top of each list is:

- **Stephanus**: real fragments, all collected. Androtion's *Atthis* book 3 (*FGrHist* 324), Philistus book 9 (556), Uranius's *Arabica* (675). Jacoby's Stephanus loci can't be matched passage by passage: they cite the headword ("s. Ἀκεσαμεναί"), which his OCR turns into Latin letters ("s. Yupavai"), and DFHG gives no headword. 556 and 675 are in III C, which we don't have anyway.
- **Eustathius**: Ister, Theopompus, Critias, and many name collisions. His "ἐν τῷ …" introduces a phrase, not a book, so the book formula misfires; the names are often poets and grammarians (Euripides' *Telephus*, Hermippus the comic poet, Timon, Zenodotus, Callimachus), or extant authors (Strabo, Arrian, Porphyry).
- **Tzetzes**: Timaeus, Hellanicus, Philostephanus, the kind of mythographic reports Jacoby collects.
- ***Anecdota***: Pherecydes and Hellanicus from the Homeric epimerisms; collisions such as Peter the Fuller and Hegesippus the church historian.

**What phase 5 shows.** The reference works are where Müller and Jacoby looked first, and the leads from them are known fragments or collisions. Lemma matching works but adds little for discovery. Without Jacoby's III A and III C, these leads can't be scored as finely as phase 2's scholia; the Müller-only prior is the ranking in the Leads tab.

**Next, if wanted.**

1. Lemmatize DFHG and use the lemma share in the "wording is known" check (item 4).
2. Phase 5b (Latin compilers) is optional.
3. Phase 6 is the user's decision. Phases 3–5 have produced checked leads but no strong uncollected fragment from printed open text. The remaining unedited Greek is in manuscripts (catenae, scholia). The repo already has a Greek minuscule model (`greek_minuscule_s9-12`, in the scan pipeline since October); the test below measures it on catena hands.

## Phase 6 test: reading a catena manuscript with the local model (2026-11-12)

Phase 6 is still the user's decision; this tests whether its main premise holds. Code: `scripts/catena_htr_test.py`; results in `data/htr_bench_catena.json`.

**Ground truth.** The PTA's diplomatic transcriptions of the Severian excerpt in five *Catena on Acts* manuscripts mark line breaks, and four link each page to a IIIF crop. Gallica refused the two Paris manuscripts (HTTP 403, its bot check, not bypassed). That leaves Jerusalem, Taphou 25, from the Library of Congress: 4 page crops and 97 transcribed lines. Oxford New College 58 and Barberini gr. 582 have no image links.

**Scores.** Each transcribed line is matched to the model line closest to it; accents and case are removed before scoring. Coverage is the quotation matcher's share of the transcription's words found in the model's text.

| Model | Line CER | Coverage |
|---|---:|---:|
| `greek_minuscule_s9-12_NFC` (Kraken) | 9% | 0.72 |
| CATMuS Medieval 1.6.0 | 85% | 0.00 |

Per page, the Greek model's CER is 6–11% and coverage 0.59–0.90. The errors are mostly missing accents and words run together ("σοιῥήματα", "ταεργασω"), plus a few misread letters ("ὄρτα" for ἔργα). Coverage of 0.72 is above the ~0.5 the matcher needs to find a known passage, so this hand can be mined as is, without fine-tuning.

**Limits.** This is one manuscript, one hand and 97 lines. The Pal. gr. 23 benchmark gave only 40% known words, so hands differ a lot. The crops are tidy single-column pages; frame catenae, with the text surrounded by commentary, will segment worse.

**What it means for phase 6.** The expensive part, building Greek minuscule HTR, turned out to exist already. The next step would be one whole catena manuscript from an open IIIF source, such as the LoC's Jerusalem and Sinai collections. Each item would be matched against Cramer and the open Fathers (phase 4's `mine_catenae.py` checks) to list the items that are in no printed edition. Labelling pages in the scan viewer is only needed if a hand reads much worse than this one. The PTA transcriptions can then serve as ready-made training lines.

## Phase 6 run: one whole catena manuscript (2026-11-12)

Code: `scripts/catena_manuscript.py` (`fetch`, `read`, `map`). The images, the HTR text and the maps are gitignored: `data/raw/quotations/manuscripts/taphou25/` and `data/quotations/catena_ms_taphou25*.json`.

**Manuscript.** Jerusalem, Hagios Stavros (Taphou) 25: a catena on Acts and the Catholic Epistles, 9th/10th century, 347 folios. The source is the Library of Congress film (public domain). It has 360 IIIF frames at half size, each cut at the gutter into two pages, so 720 pages and 298 MB. Kraken `greek_minuscule_s9-12` read the 720 pages on the RTX 4060 Ti, giving 154,955 words. About 80% of words are in the lemma table on most pages; one sampled page reached only 52%.

**Mapping.** A 50-word window (stride 25) is scored against everything printed and open:
- Cramer's catenae (tlg4102);
- the phase 4 Fathers;
- the SBLGNT text of Acts and the Catholic Epistles.

Each window gets the nearest author label before it.

The first pass covered 47% of windows at coverage ≥ 0.25. Two gaps in the baseline explained much of the rest:
- **Missing Cramer volumes.** The witness list stopped at James (tlg040). First1KGreek also has 1–2 Peter, 1–3 John and Jude (tlg041–046), plus the Apocalypse commentaries (X01–X03), and they are now in `_CRAMER`. Phase 4's `mine_catenae.py` will read them too on its next run.
- **Nomina sacra.** The scribe writes θυ, κυ, πρς, πνα, ανος and so on. Expanding them before matching (`NOMINA_SACRA`) gave every sentence about God its shingles back.

After both fixes, 58% of windows are covered. The share is about the same for every labelled author (Ammonius 59%, Severus 61%, Chrysostom 57%, Didymus 62%, Cyril 58%), so the remaining gaps are mostly misreading, not missing authors.

**Chance level.** Shuffling each window's words gives a coverage of 0.08 or less in 99% of windows. The real windows spread from 0 to 0.7 (median about 0.3). A stretch counts as a candidate only if all of the following hold:
- at least four consecutive windows (125+ words) are at or below the shuffled 99th percentile;
- the stretch is legible, with 75% or more of its words in the lemma table.

That leaves 82 chance-level runs (15,475 words), of which 27 are legible (about 4,500 words, 3% of the manuscript). They are listed in `data/quotations/catena_ms_taphou25_candidates.json` with page, line, label and text. Of the 27:
- Most cluster on the first 20 frames (Acts 1–2) and in the Pauline speeches later in Acts.
- Some are coherent text, such as the Chrysostom-labelled passage on f0022 (against those who deny the Ascension) and the Didymus passage on f0029 (the senses of τόπος).
- Others have truncated words (f0232, f0331), which looks like a segmentation failure, with half lines or marginal blocks read on their own.

**Checks so far.** Cramer's Acts in First1KGreek is complete: 118,578 words, with the same authors in the same proportions (Chrysostom 441 labels, Ammonius 143, Didymus 59). So the candidates are not in Cramer. A web search for two of the coherent passages found neither. That proves little, because the editions that could hold them are not open as clean text:
- Chrysostom's homilies on Acts (PG 60);
- Didymus's fragments on Acts (PG 39);
- the Cyril and Athanasius volumes of PG.

**Verdict.** The pipeline works end to end on a real manuscript. It reads the hand, maps 58% of it onto printed text without any labelling, and narrows 155,000 words to 27 passages worth a human look. Nothing is confirmed unpublished. Settling them takes either a check against PG 60 and PG 39, or open PG OCR added to the baseline (archive.org scans of Migne, re-OCR'd with `scripts/reocr_scans.py` if needed).

**PG 39 and PG 60 added (same day).** Neither volume is in First1KGreek, Perseus or Calfa's open PG, so I used archive.org OCR: two public-domain scans of each, 1857–62.
- **Quality.** In the Google scans, 83–88% of Greek words are real. The other two OCR'd Migne's Latin columns as Greek letters, which adds noise but cannot create 3-word matches.
- **κ read as χ.** Migne's OCR reads κ as χ throughout (χοινήν, πεντήχοντα). The map now merges the two letters on both sides (`KAPPA`) and joins words hyphenated across line ends.
- **What PG adds.** Against the Acts part of the manuscript:
  - PG 60 alone covers 21% of the windows labelled Chrysostom; Cramer's Acts covers 60%.
  - PG 60 adds only 6 windows that Cramer lacks.
  - PG 39 adds 1 window for Didymus. Migne's Didymus on Acts is largely taken from Cramer.
  - Overall coverage goes from 58% to 59%.
- **Candidates.** The strict list drops from 27 to 24 (about 4,075 words). The three that dropped were just at the threshold.
- **Weak evidence.** PG 60 finds only a fifth of even the known Chrysostom extracts. The catena abridges and rewords him, and the OCR is noisy. So a candidate's absence from PG 60 is weak evidence, not proof.
- **Who they are.** The 24 survivors are labelled Ammonius (8), Didymus (4), Cyril (4), Chrysostom (3), Severus (2), Theodoret (2), Athanasius (1) and Severian (1).
- **The best two.** The two coherent Chrysostom passages on f0022 (Acts 1, the Ascension) remain. Their content is anti-heretical: against those who say Christ neither came from heaven nor was taken up into it in the flesh. That suggests the label belongs to an earlier item and the text is by someone else.

**Next, if wanted.**
1. Read the f0011–f0032 candidates (Acts 1–2) against the page images. That is 10 passages, the most coherent ones.
2. Fix the segmentation on two-block pages: re-segment the pages whose candidates have truncated words (f0232, f0331).
3. Run the same pipeline on a second LoC catena manuscript of a different type.
