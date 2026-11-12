"""Lost philosophy quoted in the Aristotle commentators: baselines, names and surviving works.

Phase 3 of docs/quotation-mining-plan.md. The machinery is the historians' (ingest.quotations:
find_citations, the shingle matcher); what changes is what counts as "already known":

  baseline   the open fragment collections, as chunks of their Greek text. Only collections
             whose Greek OCR is usable are here (valid-word share in brackets, measured against
             the clean TEI already downloaded):
               rose        Rose's Aristotle fragments in the Berlin Aristotle vol. V (1870) [55%]
               wimmer      Wimmer's Theophrastus, Fragmenta varia (1866), First1KGreek TEI [clean]
               svf1, svf2  von Arnim, Stoicorum veterum fragmenta I-II (1903-05) [88-90%]
               doxographi  Diels, Doxographi Graeci (1879) [79%]
               rose1886    Rose's Teubner Aristotle fragments (1886), re-OCR'd here
               diels       Diels, Vorsokratiker I-II (1922), re-OCR'd here
             archive.org's OCR of Rose 1886 and Diels read the Greek as Latin letters, and Lace's
             re-OCR is offline, so scripts/reocr_scans.py re-reads their page images with
             Tesseract's Ancient Greek model. Usener's Epicurea is still missing.
  extant     the cited author's surviving works. A commentator mostly quotes extant Aristotle,
             Plato and Theophrastus; a citation whose nearby wording matches those is not a lead.

The texts searched (the commentators) are registered in ingest.quotations.WITNESSES.

PRIVATE RESEARCH USE for the OCR text, as for the FGrHist scans: von Arnim is public domain
in the US, but none of this is redistributed.
"""
from ingest.quotations import K, RAW, read_plain, read_tei, tokenize

DIR = RAW / "philosophy"
_GH = "https://raw.githubusercontent.com"
_FIRST1K = f"{_GH}/OpenGreekAndLatin/First1KGreek/master/data"
_PERSEUS = f"{_GH}/PerseusDL/canonical-greekLit/master/data"
_IA = "https://archive.org/download/{0}/{0}_djvu.txt"


def _first1k(work, edition="1st1K-grc1"):
    group, text = work.split(".", 1)
    return f"{_FIRST1K}/{group}/{text}/{work}.{edition}.xml"


def _perseus(work, edition="perseus-grc2"):
    group, text = work.split(".", 1)
    return f"{_PERSEUS}/{group}/{text}/{work}.{edition}.xml"


# name -> (label, kind, {file: url}, line range for plain text or None, authors it covers)
BASELINES = {
    "rose": ("Rose, Aristotle fragments (Berlin Aristotle V, 1870; OCR)", "plain",
             {"bub_gb_LUf4CctFZ50C.txt": _IA.format("bub_gb_LUf4CctFZ50C")},
             (100, 16590),           # the fragments; after them come Rose's indexes and Bonitz's Index
             ("ARISTOTLE",)),
    "wimmer": ("Wimmer, Theophrastus, Fragmenta varia (1866)", "tei",
               {"tlg0093.tlg010x09.xml": _first1k("tlg0093.tlg010x09")}, None, ("THEOPHRASTUS",)),
    "svf1": ("von Arnim, SVF I (Zeno and his pupils; OCR)", "plain",
             {"stoicorumveterum0001arni.txt": _IA.format("stoicorumveterum0001arni")}, None,
             ("ZENO", "CLEANTHES")),
    "svf2": ("von Arnim, SVF II (Chrysippus; OCR)", "plain",
             {"stoicorumveterum0002arni.txt": _IA.format("stoicorumveterum0002arni")}, None, ("CHRYSIPPUS",)),
    "doxographi": ("Diels, Doxographi Graeci (1879; OCR)", "plain",
                   {"doxographigraec00dielgoog.txt": _IA.format("doxographigraec00dielgoog")}, None,
                   ("THEOPHRASTUS",)),
    # Re-OCR'd here with Tesseract's Ancient Greek model (scripts/reocr_scans.py): archive.org's own
    # OCR of these read the Greek as Latin letters. No URL: the text is made, not downloaded.
    "rose1886": ("Rose, Aristotelis qui ferebantur librorum fragmenta (Teubner 1886; re-OCR)", "plain",
                 {"reocr/aristotelisquif00arisgoog.txt": None}, None, ("ARISTOTLE",)),
    "diels": ("Diels, Die Fragmente der Vorsokratiker I-II (4th ed. 1922; re-OCR)", "plain",
              {"reocr/diefragmentederv01diel_0.txt": None, "reocr/diefragmentederv02diel_0.txt": None}, None,
              ("EMPEDOCLES", "ANAXAGORAS", "DEMOCRITUS", "HERACLITUS", "PARMENIDES", "MELISSUS", "ANAXIMANDER",
               "ANAXIMENES", "PHILOLAUS", "ARCHYTAS", "LEUCIPPUS", "XENOPHANES", "THALES", "PROTAGORAS",
               "GORGIAS", "ALCMAEON", "ANTIPHON", "CRITIAS", "ZENO")),
}

# Surviving works, by author: First1KGreek and Perseus TEI.
_ARISTOTLE_FIRST1K = ("001 002 003 004 005 006 007 008 011 012 013 014 015 016 017 018 019 020 021 022 023 024 "
                      "026 027 030 031 032 034 036 037 039 040 041 042 043 044 046 047").split()
_ARISTOTLE_PERSEUS = "009 010 025 029 035 038 045".split()
EXTANT = {
    "ARISTOTLE": {**{f"tlg0086.tlg{n}.xml": _first1k(f"tlg0086.tlg{n}") for n in _ARISTOTLE_FIRST1K},
                  **{f"tlg0086.tlg{n}.xml": _perseus(f"tlg0086.tlg{n}", "perseus-grc1") for n in _ARISTOTLE_PERSEUS}},
    "THEOPHRASTUS": {f"tlg0093.{w}.xml": _first1k(f"tlg0093.{w}") for w in
                     ("tlg001", "tlg002", "tlg003", "tlg004", "tlg005", "tlg006",
                      *(f"tlg010x0{i}" for i in range(1, 9)))},
    "PLATO": {f"tlg0059.tlg0{n:02d}.xml": _perseus(f"tlg0059.tlg0{n:02d}", "perseus-grc1") for n in range(1, 37)},
    # Life of Pythagoras, On abstinence, To Marcella, Isagoge, the short Categories commentary.
    "PORPHYRY": {f"tlg2034.tlg{n}.xml": _first1k(f"tlg2034.tlg{n}", "opp-grc1" if n in ("006", "007") else "1st1K-grc1")
                 for n in ("002", "003", "005", "006", "007")},
}

# The philosophers looked for, by the accent-free stem of their Greek name. Each is matched in any
# inflection (stem + up to 4 letters). Zeno also names Zeno of Elea, Alexander the Great and
# Alexander of Aphrodisias, Strato the physicist and others: the "same person?" problem again.
PHILOSOPHERS = {
    "ARISTOTLE": "αριστοτελ", "THEOPHRASTUS": "θεοφραστ", "PLATO": "πλατων", "EUDEMUS": "ευδημ",
    "CHRYSIPPUS": "χρυσιππ", "ZENO": "ζηνων", "CLEANTHES": "κλεανθ", "STRATO": "στρατων",
    "PORPHYRY": "πορφυρι", "ALEXANDER": "αλεξανδρ", "SPEUSIPPUS": "σπευσιππ", "XENOCRATES": "ξενοκρατ",
    "ARISTOXENUS": "αριστοξεν", "DICAEARCHUS": "δικαιαρχ", "HERACLIDES": "ηρακλειδ", "POSIDONIUS": "ποσειδωνι",
    "ANDRONICUS": "ανδρονικ", "BOETHUS": "βοηθ", "ASPASIUS": "ασπασι", "HERMINUS": "ερμιν",
    # The Presocratics and sophists of Diels's Vorsokratiker. "Zeno" is shared with the Stoic.
    "EMPEDOCLES": "εμπεδοκλ", "ANAXAGORAS": "αναξαγορ", "DEMOCRITUS": "δημοκριτ", "HERACLITUS": "ηρακλειτ",
    "PARMENIDES": "παρμενιδ", "MELISSUS": "μελισσ", "ANAXIMANDER": "αναξιμανδρ", "ANAXIMENES": "αναξιμεν",
    "PHILOLAUS": "φιλολα", "ARCHYTAS": "αρχυτ", "LEUCIPPUS": "λευκιππ", "XENOPHANES": "ξενοφαν", "THALES": "θαλη",
    "PROTAGORAS": "πρωταγορ", "GORGIAS": "γοργι", "ALCMAEON": "αλκμαιων", "ANTIPHON": "αντιφων", "CRITIAS": "κριτι",
}

# Work titles beside a name ("ἐν τῷ Περὶ φιλοσοφίας"). Aristotle's lost works (Rose's titles) and
# his surviving ones; a lost title is the strongest single sign of a fragment.
# From Rose's catalogue, as the commentators write them (accent-free, crasis included: "περὶ
# τἀγαθοῦ"). Not bare "Εὔδημος" or "Σοφιστής": those are the man and Plato's dialogue.
LOST_TITLES = ("περι φιλοσοφιασ", "εν τω ευδημω", "προτρεπτικ", "περι ιδεων", "περι ταγαθου", "περι του αγαθου",
               "περι αγαθου", "περι ποιητων", "περι ευγενειασ", "περι δικαιοσυνησ", "περι ευχησ", "πολιτειαι",
               "πολιτειων", "περι αντικειμενων", "περι των αντικειμενων", "περι εναντιων", "περι των εναντιων",
               "εναντιοτητ", "αρχυτ", "πυθαγορει", "περι δημοκριτου", "ερωτικ", "γρυλλ", "μαγικ", "περι πλουτου",
               "περι ηδονησ", "περι παιδειασ", "νηρινθ", "αποικιων", "απορηματα ομηρικ", "ομηρικα απορηματα",
               "περι νειλου", "νομιμα βαρβαρικ", "διδασκαλιαι", "πυθιονικ", "ολυμπιονικ", "ανατομαι", "ανατομων",
               "περι φυτων", "περι τροφησ", "παροιμι", "εξωτερικοισ λογοισ", "εξωτερικων λογων", "τοισ διαλογοισ",
               "εν τοισ διαλογοισ")
EXTANT_TITLES = ("περι ψυχησ", "μετεωρολογικ", "φυσικ", "κατηγορι", "αναλυτικ", "τοπικ", "μετα τα φυσικα",
                 "ηθικ", "πολιτικ", "περι ουρανου", "περι γενεσεωσ", "ιστοριαι", "περι ζωων", "ρητορικ",
                 "ποιητικ", "περι ερμηνειασ", "σοφιστικ", "περι αισθησεωσ", "περι υπνου", "μορι")


def baseline_paths(name):
    return [DIR / f for f in BASELINES[name][2]]


def extant_paths(author):
    return [DIR / "extant" / f for f in EXTANT.get(author, {})]


def _text_parts(name):
    _, kind, _, lines, _ = BASELINES[name]
    if kind == "tei":
        return read_tei(baseline_paths(name))
    parts = read_plain(baseline_paths(name))
    if lines:
        parts = [p for p in parts if lines[0] <= int(p[1].rsplit(" ", 1)[1]) <= lines[1]]
    return parts


def baseline_fragments(names=None, chunk=60):
    """The baselines as fragment-like records (the shape load_dfhg returns), one per chunk of
    ``chunk`` Greek words, once for each author the collection covers."""
    out = []
    for name in names or BASELINES:
        if not all(p.exists() for p in baseline_paths(name)):
            continue
        words = [n for text, _ in _text_parts(name) for _, n in tokenize(text)]
        for author in BASELINES[name][4]:
            for i in range(0, len(words), chunk):
                toks = words[i:i + chunk]
                out.append({"author": author, "n": f"{name}:{i // chunk + 1}", "witness": "", "work": name,
                            "text": " ".join(toks), "toks": toks, "blob": " ".join(toks), "collection": name})
    return out


def extant_grams(author):
    """3-grams of an author's surviving works (empty if none registered or downloaded)."""
    grams = set()
    for path in extant_paths(author):
        if path.exists():
            toks = [n for text, _ in read_tei([path]) for _, n in tokenize(text)]
            grams.update(tuple(toks[i:i + K]) for i in range(len(toks) - K + 1))
    return grams


# Titles by author: (lost, extant). Only these authors' titles are checked: "Θεόφραστος ἐν τοῖς
# Φυσικοῖς" is Theophrastus's lost Physics, not Aristotle's.
TITLES = {
    "ARISTOTLE": (LOST_TITLES, EXTANT_TITLES),
    "PORPHYRY": ((), ("εισαγωγ", "προσ μαρκελλ", "περι αποχησ", "πυθαγορου βι")),
}


def title_near(witness, pos, author="ARISTOTLE", span=8):
    """'lost' or 'extant' if one of the author's work titles stands within ``span`` words of the name."""
    lost, extant = TITLES.get(author, ((), ()))
    near = " ".join(witness.tok[max(0, pos - span):pos + span])
    if any(t in near for t in lost):
        return "lost"
    if any(t in near for t in extant):
        return "extant"
    return None


def valid_word_share(text, vocabulary):
    """Share of Greek words (3+ letters) found in a vocabulary: a quick OCR quality score."""
    toks = [n for _, n in tokenize(text) if len(n) >= 3]
    return sum(t in vocabulary for t in toks) / max(1, len(toks))


def downloads():
    """Every file this module downloads: (path, url). Re-OCR'd baselines have no URL."""
    files = [(DIR / f, url) for spec in BASELINES.values() for f, url in spec[2].items() if url]
    files += [(DIR / "extant" / f, url) for works in EXTANT.values() for f, url in works.items()]
    return files
