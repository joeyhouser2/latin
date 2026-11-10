# Quotation-mining pilot (Greek, November 2026)

Goal: test whether we can find quotations of lost Greek historians automatically, using open data only. This was a read-only experiment: nothing was added to `corpus.db`.

**Result: the matcher works, and the run found no confirmed new fragments.** Six citations are worth checking by hand against Jacoby/BNJ. They are leads, not discoveries.

## Method

- **Baseline:** DFHG, Müller's *Fragmenta Historicorum Graecorum*, pulled through its public API. This gave 8,200 records, 6,389 of them with at least 12 Greek words. DFHG's bulk export page was not used because it carries a CAPTCHA.
- **Matching:** shared three-word sequences after removing accents, scored as the share of a fragment's words covered within the best-matching passage.
- **Texts searched:** Athenaeus (Perseus), the Suda (First1KGreek), the Pindar scholia (`tlg5034`), and Procopius of Gaza's catena (`PG087_1`, OCR).

## Results

### Recall on a heavily mined text

DFHG names Athenaeus as the source of 951 fragments, once Müller's typos ("Athenoeus", "Atehn.") and "same as above" notes are resolved.

| Coverage at least | Fragments found | Shuffled-word control |
|---:|---:|---:|
| 30% | 99.1% | 0.1% |
| 50% | 98.3% | 0.0% |
| 70% | 96.1% | 0.0% |
| 90% | 77.2% | 0.0% |

This is close to a best case: clean text, and fragments printed from the same work.

### Recall on OCR

The same test on archive.org OCR of Stobaeus (three volumes, 27 known fragments) found 89% at 30% coverage and 70% at 50% coverage, but only 4% at 90%. The sample is small, and the three volumes may not contain every fragment. OCR is usable with a lower threshold, but it clearly loses wording.

### Other texts

| Text | DFHG fragments matched, source listed | Matched, source not listed | Citations of DFHG authors | Citations with no DFHG match |
|---|---:|---:|---:|---:|
| Athenaeus | n/a | n/a | 713 | 22 |
| Suda | 185 | 120 | 151 | 11 |
| Pindar scholia | 114 | 3 | 101 | 5 |
| Procopius catena | 0 | 0 | 2 | 0 |

- **"Source not listed" is an upper bound.** DFHG's source field is irregular, so some are parsing misses. The rest look like real parallels, which could help an editor, but they aren't new fragments.
- **The catena is the wrong test for this baseline.** It excerpts biblical commentators, not historians.
- **Name collisions dominate without a filter.** An earlier run without one gave 110 unmatched Suda citations and 25 Pindar ones, and I read them all. Most were Aristophanes the playwright read as the Boeotian historian, Demosthenes the orator, *Olympia* read as Olympus, and Patroclus read as Patrocles. The tool now skips 29 such first names by default, which also hides real historians named Dionysius, Demetrius or Apollonius (`--include-ambiguous` shows them).

### Absence from DFHG doesn't mean uncollected

Athenaeus works as the control. Its unmatched citations include real ones such as Aristobulus on Susa, which is well known but absent from DFHG's text. Any lead must therefore be checked against Jacoby/BNJ.

## Leads to check by hand

Each passage names the author, and its key wording was not found in DFHG's fragment records. Arrian's wording was also checked against the *Anabasis* and *Indica* only.

| Passage | Cited as | Note |
|---|---|---|
| Schol. Pind. Pyth. 3.137 (`b.3.137`) | Aristodemus | Olympichus the piper taught by Pindar |
| Schol. Pind. Nem. 7.56 (`c.7.56`) | Aristodemus | Neoptolemus as trainer |
| Schol. Pind. Ol. 10.83 (`a.10.83.b`) | Aristodemus | Chronology of Halirrothius |
| Suda A 1218 (`1.A.1218`) | Arrian | Fishermen |
| Suda A 1223 (`1.A.1223`) | Arrian | Stoneless plain |
| Suda Π 687 (`4.Π.687`) | Priscus | Envoys denied passage |

My expectation, from memory and unverified, is that the Aristodemus passages are already in Jacoby (*FGrHist* 383) and the Arrian glosses in Roos's edition. They may also be a different Aristodemus. The Suda's own numbering wasn't confirmed: the labels come from the First1KGreek file.

## Limits

- 21 of 629 DFHG author names failed to download, including Polybius, Memnon, Heraclides Ponticus and "Aristodemus", so those fragments aren't in the baseline.
- Only 185 of 608 authors are in the citation search: those whose name could be matched to Greek and isn't on the ambiguous list.
- DFHG covers historians only, from a 19th-century edition.
- The remaining names still collide. Of the 16 unmatched Suda and Pindar citations in the final run, most were still not real leads, such as Theopompus the comic poet.

## Gap list

[`tlg-gap-list.csv`](tlg-gap-list.csv) lists 1,665 Greek works in the Perseus catalog with no open TEI in Perseus or First1KGreek. 1,356 have an archive.org link, and another 176 have only a Google Books link.

- **Not a full view of TLG:** the catalog covers about 2,800 Greek works, against TLG's 10,000 online.
- **Not checked against PTA, PG Corpus or `corpus.db`:** some listed works may already be available.
- **Genre labels are rough:** they come from title keywords, and only 36 works got one, so `other` still contains relevant texts.
- **OCR column:** it was only checked for works with a genre label and an archive.org link. All 18 of those have OCR text.
- **Edition links:** each row shows one edition, and for multi-volume sets that may not be the right volume.

## Running it

### In the web app

The **Leads** tab does the whole loop:

1. **Find quotations** queues a CPU job (`mine_quotes`) for the ticked texts. The first run downloads DFHG, which takes about 10 minutes.
2. **Sort … with the local model** queues a GPU job (`triage_leads`). A private Ollama reads each untagged lead and tags it: for a citation, whether the name means the DFHG historian, and whether it's a quotation, paraphrase or passing mention; for a reuse lead, how close the wording is. It also gives a rough English translation. The tags only order and filter the list.
3. **Review**: filter by text, kind, tag and verdict, and mark each lead *new*, *already collected*, *not a quotation* or *unsure*, with a note such as a BNJ number.

Leads are kept in `data/leads.db`, apart from `corpus.db`. Re-running a search keeps the model's tags on leads it finds again and every lead you've reviewed; unreviewed leads it no longer finds are dropped.

### How good is the model's sorting?

Measured on 138 Suda and Pindar-scholia citations, mined with ambiguous names included. The pilot's hand review gives the answer for 102 of them: 7 that really mean the historian (the six leads plus Arrian, Suda Ρ 217), and 95 name collisions.

| Model and prompt | Known leads tagged "the historian" | Collisions tagged "someone else" |
|---|---:|---:|
| qwen3:14b, yes/no asked first | 0 of 7 (all "unsure") | 4 of 95 |
| qwen3:14b, translation and "who is this?" first | 7 of 7 | 71 of 95 |

With the second prompt, all 7 known leads ranked in the top 33 of 138. Most of the 22 collisions it still got wrong were Aristophanes the playwright and Demosthenes the orator, which the default name filter already removes. About 6 leads a minute on the RTX 4060 Ti. The tags are a sorting aid only: one in four "the historian" tags in this test was wrong.

### From the command line

```bash
python scripts/mine_quotations.py run --witness suda
```

```bash
python scripts/triage_leads.py --model qwen3:14b
```

```bash
python scripts/mine_quotations.py recall --witness athenaeus
```

```bash
python scripts/tlg_gap_list.py
```

```bash
python scripts/quotations_selftest.py
```

Downloads go to `data/raw/quotations`, and the measuring commands write JSON to `data/quotations`, both gitignored. A new text is added as a row in `WITNESSES` in `ingest/quotations.py`.

## Sources

- [DFHG](https://www.dfhg-project.org/) (CC BY-SA 4.0)
- [Perseus catalog data](https://github.com/PerseusDL/catalog_data)
- [Perseus canonical-greekLit](https://github.com/PerseusDL/canonical-greekLit)
- [First1KGreek](https://github.com/OpenGreekAndLatin/First1KGreek)
- [Patrologia Graeca corpus](https://github.com/calfa-co/Patrologia-Graeca)
