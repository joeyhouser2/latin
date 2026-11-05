# Quotation-mining pilot (Greek, November 2026)

Goal: test whether we can find quotations of lost Greek historians automatically, using open data only. This was a read-only experiment: nothing was added to `corpus.db`.

**Result: the matcher works, and the run found no confirmed new fragments.** Six citations are worth checking by hand against Jacoby/BNJ. They are leads, not discoveries.

## Method

- **Baseline:** DFHG, Müller's *Fragmenta Historicorum Graecorum*, pulled through its public API. This gave 8,200 records, 6,389 of them with at least 12 Greek words. DFHG's bulk export page was not used because it carries a CAPTCHA.
- **Matching:** shared three-word sequences after removing accents, scored as the share of a fragment's words covered within the best-matching passage.
- **Texts searched:** Athenaeus (Perseus), the Suda (First1KGreek), the Pindar scholia (`tlg5034`), and Procopius of Gaza's catena (`PG087_1`, OCR).

## Results

### Recall on a heavily mined text

DFHG names Athenaeus as the source of 933 fragments.

| Coverage at least | Fragments found | Shuffled-word control |
|---:|---:|---:|
| 30% | 99.6% | 0.0% |
| 50% | 98.8% | 0.0% |
| 70% | 96.6% | 0.0% |
| 90% | 77.8% | 0.0% |

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
- **OCR column:** it was only checked for works with a genre label and an archive.org link. Two of those items returned no file list (`missing`).
- **Edition links:** each row shows one edition, and for multi-volume sets that may not be the right volume.

## Running it

```bash
python scripts/mine_quotations.py fetch
```

```bash
python scripts/mine_quotations.py recall --witness athenaeus
```

```bash
python scripts/mine_quotations.py cites --witness suda
```

```bash
python scripts/tlg_gap_list.py
```

```bash
python scripts/quotations_selftest.py
```

Downloads go to `data/raw/quotations` and results to `data/quotations`, both gitignored. New witness texts are added to `WITNESSES` in `scripts/mine_quotations.py`.

## Sources

- [DFHG](https://www.dfhg-project.org/) (CC BY-SA 4.0)
- [Perseus catalog data](https://github.com/PerseusDL/catalog_data)
- [Perseus canonical-greekLit](https://github.com/PerseusDL/canonical-greekLit)
- [First1KGreek](https://github.com/OpenGreekAndLatin/First1KGreek)
- [Patrologia Graeca corpus](https://github.com/calfa-co/Patrologia-Graeca)
