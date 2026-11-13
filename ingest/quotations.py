"""Find quotations of lost Greek authors in later texts.

The baseline is DFHG (Mueller's Fragmenta Historicorum Graecorum). A witness is
any later text flattened to a token stream with a location per token. Two
independent signals come out of comparing them:

  reuse      a DFHG fragment's wording appears in the witness (`score`)
  citation   the witness names a DFHG author beside a citing formula, and the
             surrounding words match no DFHG fragment (`find_citations`)

Neither proves anything new. DFHG is a 19th-century edition of the historians
only, so a passage missing from it may well be in Jacoby/BNJ: treat every result
as a lead to check by hand. See docs/quotation-mining-pilot.md for measured
recall and for what the false positives look like.
"""
import collections
import json
import pathlib
import re
import unicodedata
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "quotations"      # downloads (gitignored)
_GH = "https://raw.githubusercontent.com"
_PERSEUS = f"{_GH}/PerseusDL/canonical-greekLit/master/data"
_FIRST1K = f"{_GH}/OpenGreekAndLatin/First1KGreek/master/data"
_IA = "https://archive.org/download/{0}/{0}_djvu.txt"

# The texts searched for quotations. name -> label, reader kind, {file: url}, and a
# pattern for this text in DFHG's source field (used to tell a known source from
# a new one). Adding a text means adding a row here.
WITNESSES = {
    "athenaeus": ("Athenaeus, Deipnosophists", "tei",
                  {"athenaeus.xml": f"{_PERSEUS}/tlg0008/tlg001/tlg0008.tlg001.perseus-grc4.xml"},
                  # DFHG's typos include Athcnaeus, Atheuaes, Atehn.; not Athenagoras or Atheniensis.
                  r"\bAt(?:h[ce][nu](?!ag|i)|ehn|heiu)"),
    "suda": ("Suda", "tei", {"suda.xml": f"{_FIRST1K}/tlg9010/tlg001/tlg9010.tlg001.1st1K-grc1.xml"}, r"Suid|Suda"),
    "pindar_scholia": ("Scholia on Pindar (Drachmann)", "tei",
                       {f"tlg5034{p}.xml": f"{_FIRST1K}/tlg5034/tlg001{p}/tlg5034.tlg001{p}.perseus-grc1.xml" for p in "abcd"},
                       r"Schol[^|]*Pind|Pind[^|]*[Ss]chol"),
    "procopius_catena": ("Procopius of Gaza, catena on the Octateuch (PG 87, OCR)", "pg",
                         {"pg087_1.txt": f"{_GH}/calfa-co/Patrologia-Graeca/main/PG087_1/PG087_1_text.txt"}, r"Procop[^|]*Gaz"),
    # New volumes go at the end: a lead's key is its word position, so the order must not change.
    # Hense vol. II and Wachsmuth's 1884 vols. I-II have no usable Greek OCR on archive.org.
    "stobaeus_ocr": ("Stobaeus, Anthology (archive.org OCR, 5 vols)", "plain",
                     {f"{i}.txt": _IA.format(i) for i in ("ioannisstobaeiec00stobuoft", "joannisstobaeian03stovuoft",
                                                          "joannisstobaeian04stovuoft",
                                                          "ioannisstobaeie01meingoog",      # Meineke, Eclogae book I
                                                          "adw8682.0005.001.umich.edu")},   # Hense, vol. V
                     r"Stob|ΣΤΟΒΑ"),
    # Naber's edition (1864-65); Porson's 1822 text adds only 1 of 47 known fragments.
    "photius_lexicon": ("Photius, Lexicon (Naber, archive.org OCR, 2 vols)", "plain",
                        {f"{i}.txt": _IA.format(i) for i in ("photiipatriarch00nabegoog", "photiipatriarcha02phot")},
                        r"Phot[^|]*Lex|Lex[^|]*Phot"),
    # Calibration texts (phase 2 of docs/quotation-mining-plan.md): the scholia where Jacoby found
    # most of what he added to Mueller for the genealogists and mythographers.
    "euripides_scholia": ("Scholia on Euripides (Schwartz)", "tei",
                          {"euripides_scholia.xml": f"{_FIRST1K}/tlg5023/tlg001/tlg5023.tlg001.1st1K-grc1.xml"},
                          r"Schol[^|]*Eur|Eur[^|]*[Ss]chol"),
    "sophocles_scholia": ("Scholia on Sophocles (Papageorgius)", "tei",
                          {"sophocles_scholia.xml": f"{_FIRST1K}/tlg5037/tlg004/tlg5037.tlg004.1st1K-grc1.xml"},
                          r"Schol[^|]*Soph|Soph[^|]*[Ss]chol"),
    # The Odyssey scholia give the verse only in each note's lemma ("5. τοῖσι δʼ Ἀθηναίη] ..."), hence tei_verse.
    "homer_scholia": ("Scholia on the Iliad (Dindorf, Maass) and Odyssey (Dindorf)", "tei_verse",
                      {"homer_scholia_iliad.xml": f"{_FIRST1K}/tlg5026/tlg001/tlg5026.tlg001.1st1K-grc1.xml",
                       "homer_scholia_odyssey.xml": f"{_FIRST1K}/tlg5026/tlg007/tlg5026.tlg007.1st1K-grc1.xml"},
                      r"Schol[^|]*(?:Hom|Il\.|Od)|(?:Hom|Il\.|Odyss)[^|]*[Ss]chol"),
    # The Aristotle commentators (phase 3; their lost philosophers are in ingest/philosophy.py).
    # Simplicius and Alexander are the calibration texts: the most combed for fragments.
    **{name: (label, "tei", {f"{w}.xml": f"{_FIRST1K}/{w.split('.')[0]}/{w.split('.')[1]}/{w}.1st1K-grc1.xml"
                             for w in works}, r"Simpl" if name == "simplicius" else r"Alex[^|]*Aphr" if name == "alexander"
                    else r"(?!x)x")
       for name, label, works in (
           ("simplicius", "Simplicius, on the Physics, De caelo, Categories, De anima",
            ("tlg4013.tlg004", "tlg4013.tlg001", "tlg4013.tlg003", "tlg4013.tlg005")),
           ("alexander", "Alexander of Aphrodisias, on the Metaphysics, Topics, Prior Analytics, De sensu",
            ("tlg0732.tlg004", "tlg0732.tlg006", "tlg0732.tlg005", "tlg0732.tlg007")),
           ("michael_ephesus", "Michael of Ephesus, on the Parva naturalia, Parts, Movement and Gait of animals, EN V",
            ("tlg4034.tlg002", "tlg4034.tlg003", "tlg4034.tlg004", "tlg4034.tlg005", "tlg4034.tlg006")),
           ("philoponus_bio", "Philoponus, on Meteorology I and Generation of animals",
            ("tlg4015.tlg005", "tlg4015.tlg007")),
           ("olympiodorus_meteor", "Olympiodorus, on the Meteorology", ("tlg4019.tlg003",)),
           ("asclepius_metaph", "Asclepius, on the Metaphysics A-Z", ("tlg4018.tlg001",)),
           ("eustratius_en", "Eustratius, on the Nicomachean Ethics I", ("tlg4031.tlg002",)),
       )},
}
# Cramer's catenae on the New Testament (phase 4): items labelled by author, read by ingest/catenae.py.
# 041-046: the Petrine, Johannine and Jude catenae; X01-X03: the Apocalypse commentaries.
_CRAMER = ("001 002 003 004 005 006 007 008 010 011 012 013 019 020 021 022 023 024 034 035 036 037 038 040 "
           "041 042 043 044 045 046 X01 X02 X03").split()
WITNESSES["cramer_catenae"] = ("Cramer, Catenae Graecorum Patrum in Novum Testamentum (33 catenae)", "tei",
                               {f"tlg4102.tlg{n}.xml": f"{_FIRST1K}/tlg4102/tlg{n}/tlg4102.tlg{n}.1st1K-grc1.xml"
                                for n in _CRAMER}, r"Cramer[^|]*Caten|Caten[^|]*Cramer")
# OCR'd Byzantine reference works (phase 5). archive.org's OCR where its Greek is usable (84-91%
# valid words); re-OCR'd text ("reocr/...", scripts/reocr_scans.py) where it read the Greek as Latin
# letters (Eustathius Iliad I and III) or poorly (Stephanus, Tzetzes: 66%).
_EUSTATHIUS = {"reocr/commentariiadho01eust.txt": None, "commentariiadho02eust.txt": _IA.format("commentariiadho02eust"),
               "reocr/commentariiadhom03eust.txt": None, "commentariiadhom04eust.txt": _IA.format("commentariiadhom04eust"),
               "commentariiadhom01eust.txt": _IA.format("commentariiadhom01eust"),
               "commentariiadhom02eust.txt": _IA.format("commentariiadhom02eust")}
WITNESSES.update({
    "eustathius": ("Eustathius, on the Iliad (I-IV) and Odyssey (I-II) (Stallbaum 1825-30; OCR)", "plain",
                   _EUSTATHIUS, r"Eustath"),
    "stephanus_ocr": ("Stephanus of Byzantium, Ethnica (Meineke 1849; re-OCR)", "plain",
                      {"reocr/bub_gb_0NIPAAAAQAAJ.txt": None}, r"Steph[^|]*Byz"),
    "tzetzes_lycophron": ("Tzetzes on Lycophron (Scheer 1908, vol. II; re-OCR)", "plain",
                          {"reocr/lycophronisalexa02lycouoft.txt": None}, r"Tzetz[^|]*Lyc|Lyc[^|]*Tzetz"),
    "cramer_anecdota": ("Cramer, Anecdota Graeca Oxoniensia I, III, IV and Parisiensia I-IV (OCR)", "plain",
                        {f"{i}.txt": _IA.format(i) for i in (
                            "anecdotagraecae01cram", "anecdotagraecaec0003cram", "anecdotagraecae02unkngoog",
                            "anecdotagraecaec01cram", "anecdotagraecaec02cram", "anecdotagraecaec03cram",
                            "anecdotagraecaec04cram")}, r"Cram[^|]*An|Anecd"),
})
# Texts whose citations are of philosophers (ingest/philosophy.py), not DFHG's historians.
PHILOSOPHY_WITNESSES = ("simplicius", "alexander", "michael_ephesus", "philoponus_bio", "olympiodorus_meteor",
                        "asclepius_metaph", "eustratius_en")

# The works a commentary explains (Perseus TEI). Their characters' names are the commonest false
# citation in scholia: "Menelaus says" is the hero, not Menelaus of Aegae. See character_names.
COMMENTED = {
    "homer_scholia": ("tlg0012.tlg001", "tlg0012.tlg002"),
    "eustathius": ("tlg0012.tlg001", "tlg0012.tlg002"),
    "euripides_scholia": ("tlg0006.tlg002", "tlg0006.tlg003", "tlg0006.tlg005", "tlg0006.tlg006", "tlg0006.tlg007",
                          "tlg0006.tlg011", "tlg0006.tlg015", "tlg0006.tlg016", "tlg0006.tlg019"),
    "sophocles_scholia": tuple(f"tlg0011.tlg00{i}" for i in range(1, 8)),
}


def commented_url(work):
    group, text = work.split(".")
    return f"{_PERSEUS}/{group}/{text}/{work}.perseus-grc2.xml"


def character_names(witness_name):
    """Names in the work(s) a commentary explains: words capitalized in at least 80% of their uses."""
    capital, total = collections.Counter(), collections.Counter()
    for work in COMMENTED.get(witness_name, ()):
        path = RAW / "commented" / f"{work}.xml"
        if not path.exists():
            continue
        text = " ".join(t for t, _ in read_tei([path]))
        for original, normalized in tokenize(text):
            total[normalized] += 1
            capital[normalized] += original[0].isupper()
    return {w for w, n in capital.items() if n >= 2 and n >= .8 * total[w]}


def is_character(name, stem, characters):
    """Does a cited name belong to a character of the commented work (any inflection)?"""
    return norm(name) in characters or any(c.startswith(stem) and len(c) <= len(stem) + 4 for c in characters)


def name_key(name):
    """A loose key for an author's first name across Latin and German spellings:
    "Pherecydes", "PHEREKYDES" -> "ferecid"."""
    s = (name.lower().split() or [""])[0]
    for a, b in (("k", "c"), ("ai", "ae"), ("oi", "oe"), ("ou", "u"), ("y", "i"), ("ph", "f"), ("th", "t")):
        s = s.replace(a, b)
    return re.sub(r"(os|us|es|as|is|on|um|e)$", "", s)


def author_priors(leads, fragments, source_pattern):
    """Give each citation lead from one text a prior from Mueller alone: how many fragments of
    the cited author DFHG already credits to this text, per citation lead of that author here.

    A historian the text really quotes scores high (Pherecydes in the Homer scholia: 33 credited,
    10 leads); a namesake cited constantly for something else scores near 0 (Zenodotus the Homeric
    editor: none credited, ~200 leads). On the scholia it put 11 of Jacoby's additions to Mueller
    in the top 20, where the local model put none (docs/quotation-mining-plan.md, phase 2). It
    knows nothing about texts Mueller barely used, where every author scores low."""
    credited = collections.Counter(name_key(f["author"]) for f in fragments
                                   if f["author"] and re.search(source_pattern, f["witness"]))
    cited = collections.Counter(name_key(c["author"]) for c in leads)
    for c in leads:
        key = name_key(c["author"])
        c["credited"] = credited[key]
        c["prior"] = round((credited[key] + .5) / (cited[key] + 1), 3)
    return leads


K = 3  # words per shingle; 3 keeps the shuffled-word control at about 0
TEI = "{http://www.tei-c.org/ns/1.0}"
WORD = re.compile(r"[Ͱ-Ͽἀ-῿]+")
# Editorial matter inside a TEI edition, not the witness's own text.
TEI_SKIP = {"note", "bibl", "teiHeader"}

# Mueller's "same source as the previous fragment", which the API returns as is:
# "Idem XII, p. 518", "[Id. v. X", "— Ibid.", "Adde XV, p. 677", or a bare "XI, p. 483" / "XLIV, 41".
SAME_WITNESS = re.compile(r"^[\s\[(—–-]*(?:(?:idem|ibid|ibidem|eadem|eodem|adde)\b|id\.|ibid\.|ead\."
                          r"|[IVXLC]+\s*[,.]\s*(?:p\.\s*)?\d)", re.I)
# Some DFHG witness fields use Greek capitals inside Latin words ("Αthenaeus").
HOMOGLYPHS = str.maketrans("ΑΒΕΖΗΙΚΜΝΟΡΤΥΧ", "ABEZHIKMNOPTYX")

# Formula evidence beside a name.
CITING_VERBS = {"φησι", "φησιν", "ιστορει", "ιστορησε", "ιστορησεν", "γραφει", "αναγραφει", "μεμνηται",
                "ειρηκε", "ειρηκεν", "λεγει", "γεγραφε", "γεγραφεν", "μνημονευει",
                # The scholia's own: "ὁ Τίμαιος οὕτως γράφων", "Φερεκύδης οὕτω γενεαλογεῖ", "διείλεκται Ἔφορος".
                "γραφων", "γενεαλογει", "ιστορουσι", "ιστορουσιν", "αναγραφουσι", "αναγραφουσιν", "διειλεκται",
                "φασι", "φασιν", "λεγουσι", "λεγουσιν"}
BOOK_ARTICLES = {"τω", "τη", "τοισ", "ταισ"}
# "Ἀκουσίλαος ἐν τρίτῳ Γενεαλογιῶν": a book number with no article.
ORDINALS = {o + e for o in ("πρωτ", "δευτερ", "τριτ", "τεταρτ", "πεμπτ", "εκτ", "εβδομ", "ογδο", "ενατ", "δεκατ")
            for e in ("ω", "η")}
# First names of DFHG authors that mostly mean someone else (or a place) in these
# texts. Reading all 135 unmatched Suda and Pindar-scholia citations in the pilot
# found these were nearly all collisions: Aristophanes the playwright for the
# Boeotian historian, Demosthenes the orator, Olympia for Olympus, Patroclus for
# Patrocles. Dionysius and Demetrius lose real historians too, but each covers too
# many people to separate. Pass ambiguous=frozenset() to see them anyway.
AMBIGUOUS_NAMES = frozenset({
    "ALCIMUS", "ANTIOCHUS", "APOLLONIUS", "ARISTAGORAS", "ARISTIDES", "ARISTOCRATES", "ARISTOPHANES", "ARISTOTELES",
    "ARTEMON", "CAPITO", "DEMETRIUS", "DEMOSTHENES", "DINARCHUS", "DIOGENES", "DIONYSIUS", "GORGIAS", "HERACLIDES",
    "MACAREUS", "MELANTHIUS", "NICANOR", "OLYMPUS", "PAMPHILA", "PAMPHILUS", "PATROCLES", "PHILIPPUS", "SIMONIDES",
    "SOCRATES", "THEODORUS", "ZOPYRUS",
})


def norm(token):
    """Accent- and case-free Greek, with lunate and final sigma folded."""
    decomposed = unicodedata.normalize("NFD", token)
    # Mn: combining accents. Sk: free-standing breathings and the koronis (᾿, ῾, ᾽), which OCR puts
    # before capitals ("᾿Αθήναιος") and which NFD leaves alone.
    bare = "".join(c for c in decomposed if unicodedata.category(c) not in ("Mn", "Sk")).lower()
    return bare.replace("ϲ", "σ").replace("ς", "σ").replace("ϐ", "β").replace("ϑ", "θ")


def tokenize(text):
    """[(original, normalized)] for each Greek word in text."""
    out = []
    for m in WORD.finditer(text):
        original = _SPACING_MARKS.sub("", m.group(0))
        if original:
            out.append((original, norm(original)))
    return out


_SPACING_MARKS = re.compile(r"^[᾽᾿῾΄῍῎῏῝῞῟΅῭`´]+|[᾽᾿῾΄῍῎῏῝῞῟΅῭`´]+$")


class Witness:
    """A text as parallel lists of original tokens, normalized tokens and locations."""

    def __init__(self, name, parts):
        self.name = name
        self.orig, self.tok, self.loc = [], [], []
        for text, loc in parts:
            for original, normalized in tokenize(text):
                self.orig.append(original)
                self.tok.append(normalized)
                self.loc.append(loc)
        self.index = collections.defaultdict(list)
        for i in range(len(self.tok) - K + 1):
            self.index[tuple(self.tok[i:i + K])].append(i)

    def context(self, start, end):
        return " ".join(self.orig[max(0, start):min(len(self.orig), end)])


_LEMMA_VERSE = re.compile(r"^\s*(\d{1,4})\.\s")


def _walk_tei(el, loc, out, page, verse=None):
    tag = el.tag.replace(TEI, "")
    if tag in TEI_SKIP:
        return
    if tag == "div" and el.get("type") == "textpart":
        loc = loc + [el.get("n") or "?"]
        if verse is not None and el.get("subtype") == "book":
            verse[0] = None                     # a new book starts without a verse
    # Editions cited by a standard page (Casaubon's for Athenaeus) mark it with a
    # milestone; carry it in the location so a lead can be checked against
    # collections that cite by that page. See ingest/known_loci.py.
    if tag == "milestone" and el.get("unit") == "casaubonpage" and el.get("n"):
        page[0] = el.get("n")
    # Scholia that number a note only in its lemma ("5. τοῖσι δʼ Ἀθηναίη] ...").
    if verse is not None and tag == "p" and el.text and _LEMMA_VERSE.match(el.text):
        verse[0] = _LEMMA_VERSE.match(el.text).group(1)
    where = lambda: (".".join(loc) + (f" p.{page[0]}" if page[0] else "")      # noqa: E731
                     + (f" v.{verse[0]}" if verse and verse[0] else ""))
    if el.text:
        out.append((el.text, where()))
    for child in el:
        _walk_tei(child, loc, out, page, verse)
        if child.tail:
            out.append((child.tail, where()))


def read_tei(paths, verses=False):
    """Text parts of one or more TEI editions. Several files get a part prefix: 1, 2, ...

    ``verses``: also track the verse number that starts a scholion's lemma, as " v.N"."""
    parts = []
    for i, path in enumerate(paths, 1):
        body = ET.parse(path).getroot().find(f"{TEI}text/{TEI}body")
        _walk_tei(body, [str(i)] if len(paths) > 1 else [], parts, [None], [None] if verses else None)
    return parts


def read_pg(path):
    """A Patrologia Graeca OCR volume, located by its $8=page $9=column markers."""
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    parts, loc = [], "?"
    for line in re.sub(r"[-‐]\n", "", text).split("\n"):
        marker = re.match(r"\$0=\S+ \$8=(\S+) \$9=(\S+)", line)
        if marker:
            loc = f"p.{marker.group(1)} col.{marker.group(2)}"
        else:
            parts.append((line, loc))
    return parts


def read_plain(paths):
    """Plain OCR text such as archive.org's _djvu.txt, located by file and line.

    archive.org's text files have no page markers, so a line number (counted after
    hyphenated line-ends are joined) is the most precise place we can give."""
    parts = []
    for path in paths:
        path = pathlib.Path(path)
        text = re.sub(r"-[ \t]*\n[ \t]*", "", path.read_text(encoding="utf-8", errors="replace"))
        for n, line in enumerate(text.split("\n"), 1):
            parts.append((line + "\n", f"{path.stem} line {n}"))
    return parts


REOCR = RAW / "philosophy" / "reocr"     # scripts/reocr_scans.py writes here


def witness_paths(name):
    """A witness's files. "reocr/<id>.txt" is text made by scripts/reocr_scans.py, not downloaded."""
    return [REOCR / f.split("/", 1)[1] if f.startswith("reocr/") else RAW / "witnesses" / f
            for f in WITNESSES[name][2]]


def witness_ready(name):
    return all(p.exists() for p in witness_paths(name))


def load_witness(name):
    """Read a registered witness from its downloaded files."""
    _, kind, _, _ = WITNESSES[name]
    paths = witness_paths(name)
    if kind in ("tei", "tei_verse"):
        parts = read_tei(paths, verses=kind == "tei_verse")
    else:
        parts = read_plain(paths) if kind == "plain" else read_pg(paths[0])
    return Witness(name, parts)


def resolve_witnesses(records):
    """Give each record its real witness, carrying "Idem ..." forward within an author."""
    last = ""
    for record in records:
        witness = (record.get("witness") or "").strip().translate(HOMOGLYPHS)
        if SAME_WITNESS.match(witness):
            witness = f"{last} | {witness}"
        elif witness:
            last = witness
        record["witness"] = witness
    return records


def load_dfhg(directory):
    """All fragments from a folder of per-author API responses."""
    fragments = []
    for path in sorted(pathlib.Path(directory).glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            continue
        for r in resolve_witnesses(records):
            # Some fragment groups name their source once, in a heading such as
            # "ΕΚ ΤΟΥ ΣΤΟΒΑΙΟΥ ΑΝΘΟΛΟΓΙΟΥ", rather than in each source field.
            heading = " ".join(n for n in (r.get("work_note"), r.get("section_note"), r.get("sub_volume_note"))
                               if n and re.search(r"\b(?:ΕΚ|EX|APUD|AP\.)\b", n, re.I))
            fragments.append({
                "author": r.get("author"),
                "n": f"{r.get('fragment_number')}{r.get('fragment_letter') or ''}",
                "witness": r["witness"] + (f" | heading: {heading[:120]}" if heading else ""),
                "work": (r.get("work") or "").strip(),
                "text": r.get("text") or "",
                "toks": [n for _, n in tokenize(r.get("text") or "")],
                "blob": norm(" ".join(str(r.get(k) or "") for k in ("text", "translation", "commentary", "note", "fragment_note"))),
            })
    return fragments


def score(fragment_tokens, witness, max_frequency=300):
    """(coverage, position): the share of a fragment's words found together in the best passage."""
    hits = []
    for gi in range(len(fragment_tokens) - K + 1):
        positions = witness.index.get(tuple(fragment_tokens[gi:gi + K]))
        if positions and len(positions) <= max_frequency:
            hits.extend((p, gi) for p in positions)
    if not hits:
        return 0.0, None
    hits.sort()
    span = int(len(fragment_tokens) * 1.6) + 10
    best, where, left = 0.0, None, 0
    for right in range(len(hits)):
        while hits[right][0] - hits[left][0] > span:
            left += 1
        covered = set()
        for _, gi in hits[left:right + 1]:
            covered.update(range(gi, gi + K))
        coverage = len(covered) / len(fragment_tokens)
        if coverage > best:
            best, where = coverage, hits[left][0]
    return best, where


def _translit(latin_name):
    s = latin_name.lower().split()[0]
    if s.startswith("h"):  # Latin h- is a breathing in Greek, which norm() removes
        s = s[1:]
    for a, b in (("ae", "e"), ("oe", "i"), ("ph", "f"), ("th", "0"), ("ch", "x"), ("rh", "r"), ("y", "i"), ("c", "k")):
        s = s.replace(a, b)
    s = re.sub(r"(?<=[^aeiou])er$", "r", s)   # Latin Ister, Alexander = Greek Ἴστρος, Ἀλέξανδρος
    return re.sub(r"(.)\1", r"\1", re.sub(r"(us|um|es|as|is|on|a|e|o|ius)$", "", s))


def _skeleton(greek):
    table = str.maketrans({"α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "e", "θ": "0", "ι": "i",
                           "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p", "ρ": "r", "σ": "s",
                           "τ": "t", "υ": "i", "φ": "f", "χ": "x", "ψ": "ps", "ω": "o"})
    s = greek.replace("αι", "ε").replace("ει", "ι").replace("οι", "ι").replace("ου", "u").replace("γγ", "ng").replace("γκ", "nk")
    return re.sub(r"(.)\1", r"\1", s.translate(table))


def author_stems(fragments, ambiguous=AMBIGUOUS_NAMES):
    """{DFHG author: Greek stem}, kept only if the author's own fragments use that name twice or more."""
    by_author = collections.defaultdict(list)
    for fragment in fragments:
        if fragment["author"] and fragment["author"].split()[0] not in ambiguous:
            by_author[fragment["author"]].append(fragment)
    stems = {}
    for author, own in by_author.items():
        # The stem kept is the Greek form attested in the author's own fragments, not this
        # transliteration, so a 4-letter target (Ephorus "efor", Andron "andr") is safe.
        target = _translit(author)
        if len(target) < 4:
            continue
        counts = collections.Counter()
        for fragment in own:
            for original, normalized in tokenize(fragment["text"]):
                if original[0].isupper() and _skeleton(normalized).startswith(target):
                    counts[normalized[:-2] if len(normalized) > 6 else normalized] += 1
        if counts and counts.most_common(1)[0][1] >= 2:
            stems[author] = counts.most_common(1)[0][0]
    return stems


def find_citations(witness, stems, fragments, before=15, after=45):
    """Capitalized DFHG names beside a citing formula, with the share of nearby words DFHG also has.

    Two shares: ``dfhg_share`` against all of DFHG, ``author_share`` against the cited author's own
    fragments only. A scholion citing two historians is in DFHG once for either one; for the
    other, only ``author_share`` says whether that author's fragment was collected."""
    baseline, own = set(), collections.defaultdict(set)
    for fragment in fragments:
        t = fragment["toks"]
        grams = {tuple(t[i:i + K]) for i in range(len(t) - K + 1)}
        baseline |= grams
        own[fragment["author"]] |= grams
    by_prefix = collections.defaultdict(list)
    author_of = {}
    for author, stem in stems.items():
        by_prefix[stem[:3]].append(stem)
        author_of.setdefault(stem, author)
    for prefix in by_prefix:
        by_prefix[prefix].sort(key=len, reverse=True)

    results, seen = [], set()
    for i, (original, normalized) in enumerate(zip(witness.orig, witness.tok)):
        if not original[0].isupper():
            continue
        stem = next((s for s in by_prefix.get(normalized[:3], ()) if normalized.startswith(s)), None)
        # A case ending adds at most 3 letters (-ου, -ει, -ους); 4 let Callias's "καλλι" take Καλλιμάχῳ.
        if not stem or len(normalized) > len(stem) + 3:
            continue
        low, high = max(0, i - 6), i + 7
        near = witness.tok[low:high]
        formula = "verb" if CITING_VERBS & set(near) else None
        # The scholia's subscription "ἡ ἱστορία παρὰ Φερεκύδει": "the story is in Pherecydes".
        if not formula and ("ιστορια" in near or (i and witness.tok[i - 1] == "παρα")):
            formula = "story"
        # "ὡς Φερεκύδης", and the scholia's "Ἔφορος δέ ..." setting one authority against another.
        if not formula and i and witness.tok[i - 1] == "ωσ":
            formula = "as"
        if not formula and witness.tok[i + 1:i + 2] == ["δε"]:
            formula = "but"
        if any(a == "εν" and (b in BOOK_ARTICLES or b in ORDINALS) for a, b in zip(near, near[1:])):
            formula = "book"
        if not formula or (stem, i // 30) in seen:
            continue
        seen.add((stem, i // 30))
        start, end = max(0, i - before), min(len(witness.tok), i + after)
        window = witness.tok[start:end]
        author = author_of[stem]
        results.append({"witness": witness.name, "pos": i, "loc": witness.loc[i], "author": author, "name": original,
                        "formula": formula, "dfhg_share": _share(window, baseline),
                        "author_share": _share(window, own[author]), "context": witness.context(start, end)})
    return results


def _share(window, grams):
    """Share of a window's words covered by 3-grams from ``grams``."""
    covered = set()
    for k in range(len(window) - K + 1):
        if tuple(window[k:k + K]) in grams:
            covered.update(range(k, k + K))
    return round(len(covered) / max(1, len(window)), 2)
