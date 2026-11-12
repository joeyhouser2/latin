"""Which passages are already collected as fragments: a table of (text, passage) pairs.

A lead found at a passage some collection already prints is known, however its
wording came out in our OCR or theirs. This module builds that table from two
sources and maps our own texts' locations onto the same scheme.

  FHG       DFHG's source field for each fragment ("Athen. XII, p. 527, F").
  FGrHist   the source line printed above each fragment in Jacoby's *Die
            Fragmente der griechischen Historiker* I, II and III B, read from the
            open archive.org scans (see scripts/build_known_loci.py). Their Greek
            OCR is poor (the Vanderbilt scans read it as Latin letters), but these
            lines are in Latin script and read well, e.g. "4 (8) ATHEN. XII 59 p.
            542 A: ...". The bracket is the matching FHG reference, so the line
            also gives a concordance.

The 1950s volumes are in copyright: what is built from them is for private
research use and is not to be published.

Passages are normalised per text, in the scheme that text's editors cite:

  athenaeus          Casaubon page and letter      "542a"
  suda               headword, accent-free          "ομηροσ"
  pindar_scholia     ode, poem, line                "O.7.24"
  euripides_scholia  play, line                     "Ph.1116"
  sophocles_scholia  play, line                     "OC.1053"
  homer_scholia      poem, book, line               "Il.14.119" ("?" for an unreadable book)

Other texts are kept with their raw citation (``locus`` None) so the table
still says which sources a collection used. Citation schemes differ between
editions (Pindar scholia: Boeckh's line numbers in FHG, Drachmann's scholion
numbers in our text), so matching takes a tolerance; see ``near``.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from ingest.quotations import name_key, norm   # noqa: F401  (name_key is used here and re-exported)

# OCR of "SCHOL." alone comes out as ScHoL., Scuor., Schon., Scnor., Scuou.
_SCHOL = r"S[cC][hHuUnN][oO0][lLrRuUnN]\.?"
WITNESS_PATTERNS = {
    "athenaeus": rf"^(?:A(?:TH|th|rn|ru|rh)[EeA]?[NnI]\w*\.?|Athen\w*\.?)(?!\s*(?:ag|ien))",
    "suda": r"^(?:PHOT\.\s*)?(?:SUID|Suid|SUDA|Suda|SUD\.|Sud\.)\w*\.?",
    "pindar_scholia": rf"^{_SCHOL}\s*(?:[A-Z]{{1,3}}\s+)?PIND\w*\.?|^Sch(?:ol)?\.\s*Pind",
    "photius_lexicon": r"^PHOT\.(?!\s*(?:Bibl|SUID))|^Phot\w*\.?\s*Lex",
    "stobaeus_ocr": r"^STOB\w*\.?|^Stob\w*\.?",
    "homer_scholia": rf"^{_SCHOL}\s*(?:[A-Z]{{1,4}}\s+)?H[oO]M\w*\.?|^Schol\.\s*(?:\w+\s+)?Hom",
    "euripides_scholia": rf"^{_SCHOL}\s*(?:[A-Z]{{1,3}}\s+)?EUR\w*\.?|^Schol\.\s*Eur",
    "sophocles_scholia": rf"^{_SCHOL}\s*(?:[A-Z]{{1,3}}\s+)?SOPH\w*\.?|^Schol\.\s*Soph",
    "stephanus": r"^STEPH\.?\s*BYZ|^Steph\.?\s*Byz",
    "harpocration": r"^HARPO[KR]R\w*\.?|^Harpocr",
    "eustathius": r"^EUSTATH\w*\.?|^Eustath",
}
_ODES = {"o": "O", "ol": "O", "olymp": "O", "p": "P", "py": "P", "pyth": "P", "n": "N", "nem": "N",
         "i": "I", "is": "I", "isthm": "I", "j": "I"}
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


@dataclass
class Locus:
    collection: str             # FHG | FGrHist
    author: str                 # FHG author name, or FGrHist number + name
    fragment: str
    witness: Optional[str]      # registry name if recognised (ingest.quotations.WITNESSES and a few more)
    locus: Optional[str]        # normalised passage in that text's scheme, if parsed
    raw: str                    # the source as printed
    concordance: str = ""       # FGrHist only: the FHG reference in brackets


def roman(s: str) -> Optional[int]:
    s = s.upper()
    if not s or not re.fullmatch(r"[IVXLC]+", s):
        return None
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = _ROMAN[a]
        total += -v if b != " " and _ROMAN.get(b, 0) > v else v
    return total


def canon_witness(raw: str) -> Optional[str]:
    """Which text a printed source names, if it is one we know."""
    text = raw.strip().lstrip("—–-[( ")
    for name, pattern in WITNESS_PATTERNS.items():
        if re.match(pattern, text):
            return name
    return None


_LATIN_CAPS = str.maketrans("ΑΒΓΔΕΖΗΙΚΜΝΟΡΤΥΧΆÀÁÂÉÈ", "ABΓDEZHIKMNOPTYXAAAAEE")


def parse_locus(witness: Optional[str], raw: str) -> Optional[str]:
    """Normalise a printed passage reference for the texts we can map."""
    if witness == "athenaeus":
        # "XII 59 p. 542 A", "Athen. XII, p. 527, F", "II 70 A", "XIV, 619, D", "II 35 AB": a page
        # number followed by a section letter is Casaubon's page. After "p." the first one is where the
        # passage starts ("p. 148 F—149 C"); without "p." the last one (the book number comes first).
        # Part III's OCR prints some section letters as Greek capitals ("660 ΑΒ") or with accents ("567 À").
        raw = raw.translate(_LATIN_CAPS)
        after_p = raw.split("p.", 1)[1] if "p." in raw else ""
        first = re.search(r"(\d{1,3})\s*[,.]?\s*([A-F])[A-F]?\b", after_p)
        if first:
            return f"{int(first.group(1))}{first.group(2).lower()}"
        pairs = re.findall(r"(\d{1,3})\s*[,.]?\s*([A-F])[A-F]?\b", raw)
        return f"{int(pairs[-1][0])}{pairs[-1][1].lower()}" if pairs else None
    if witness == "pindar_scholia":
        m = re.search(r"\b(Olymp|Ol|O|Pyth|Py|P|Nem|N|Isthm|Is|I|J)\b\.?\s*([IVXL]+|\d{1,2})\b"
                      r"(?:\s*\([IVXL]+\))?\s*[,.]?\s*(?:vs\.\s*)?(\d{1,4})", raw)
        if not m:
            return None
        poem = roman(m.group(2)) if not m.group(2).isdigit() else int(m.group(2))
        return f"{_ODES[m.group(1).lower()]}.{poem}.{int(m.group(3))}" if poem else None
    if witness == "suda":
        m = re.search(r"\b(?:s\.\s*[vV]\.|s\.|v\.)\s*[᾽᾿’']?([Ͱ-Ͽἀ-῿][^\s:,;.*]*)", raw)
        return norm(m.group(1)) if m else None
    if witness == "euripides_scholia":
        m = re.search(r"EUR\w*[.,]?\s*(?:ad\s+)?([A-Za-z]+)\.?\s*(?:v\.\s*)?(\d{1,4})", raw, re.I)
        play = m and next((code for prefix, code in _EURIPIDES if m.group(1).lower().startswith(prefix)), None)
        return f"{play}.{int(m.group(2))}" if play else None
    if witness == "sophocles_scholia":
        m = re.search(r"SOPH\w*\.?\s*(?:ad\s+)?([A-Za-z. ]{1,20}?)\s*(?:v\.\s*)?(\d{1,4})", raw, re.I)
        play = m and _sophocles_play(m.group(1))
        return f"{play}.{int(m.group(2))}" if play else None
    if witness == "homer_scholia":
        m = re.search(r"\b(Il|II|Tl|Od|Odyss)\w*\.?\s*(?:([^\s,.]{1,5})\s*[,.]?\s*)?(\d{1,4})\b", raw, re.I)
        # Jacoby's OCR garbles "Il." after HOM. ("I!.", "1].", "11.", "N."); "Od." comes through.
        m = m or re.search(r"H[oO]M\w*\.?\s*([^\sOo]\S{0,2})\s+([^\s,.]{1,5})\s*[,.]?\s*(\d{1,4})\b", raw)
        if not m:
            return None
        poem ="Od" if m.group(1).lower().startswith("od") else "Il"
        return f"{poem}.{_homer_book(m.group(2) or '')}.{int(m.group(3))}"
    return None


# Euripides plays as Jacoby and Mueller abbreviate them -> the code in our edition's
# "sch_Hec" divisions. Longer prefixes first.
_EURIPIDES = (("alk", "Alc"), ("alc", "Alc"), ("andr", "Andr"), ("hek", "Hec"), ("hec", "Hec"), ("hipp", "Hipp"),
              ("med", "Med"), ("or", "Or"), ("phoe", "Ph"), ("ph", "Ph"), ("rh", "Rh"), ("tro", "Tr"), ("tr", "Tr"))
# Our Sophocles scholia (Papageorgius) number the plays 1-7 in this order.
SOPHOCLES_ORDER = ("Aj", "El", "OT", "Ant", "Tr", "Ph", "OC")


def _sophocles_play(raw: str) -> Optional[str]:
    """ "OK", "O. K.", "Oed. Col.", "OEdip. Colon." -> "OC"; "OT", "Oed. Tyr." -> "OT"; "Trach." -> "Tr" ..."""
    s = re.sub(r"[^a-z]", "", raw.lower())
    if s in ("ok", "oc") or re.match(r"oe?d(?:ip(?:us)?)?(?:c|k)", s):
        return "OC"
    if s in ("ot", "or") or re.match(r"oe?d(?:ip(?:us)?)?(?:t|r)", s):
        return "OT"
    for prefix, code in (("tr", "Tr"), ("el", "El"), ("ai", "Aj"), ("aj", "Aj"), ("ant", "Ant"), ("ph", "Ph")):
        if s.startswith(prefix):
            return code
    return None


# Homer's books are cited by Greek letter (capital for the Iliad, small for the Odyssey, both
# α-ω = 1-24), or roman numerals. Jacoby's OCR prints capitals as their Latin look-alikes; any
# other reading ("Il. 2 343": Σ? Ζ?) is unknown, "?", and matches any book.
_GREEK_BOOKS = "αβγδεζηθικλμνξοπρστυφχψω"
_LATIN_LOOKALIKE = {"a": 1, "b": 2, "e": 5, "z": 6, "h": 7, "i": 9, "k": 10, "m": 12, "n": 13, "o": 15, "p": 17,
                    "t": 19, "y": 20, "u": 20, "x": 22}


def _homer_book(raw: str) -> str:
    raw = "".join(c for c in unicodedata.normalize("NFD", raw.strip()) if not unicodedata.combining(c))
    if len(raw) == 1 and raw.lower() in _GREEK_BOOKS:
        return str(_GREEK_BOOKS.index(raw.lower()) + 1)
    if len(raw) == 1 and raw.lower() in _LATIN_LOOKALIKE:
        return str(_LATIN_LOOKALIKE[raw.lower()])
    n = roman(raw)
    return str(n) if n and n <= 24 else "?"


def _last_segment(witness_field: str) -> str:
    """DFHG's resolved source keeps the chain ("Athen. XII | Idem XIV, p. 649, F"); the last part is this fragment's."""
    parts = [p.strip() for p in witness_field.split("|") if p.strip() and not p.strip().startswith("heading:")]
    return parts[-1] if parts else ""


def from_dfhg(fragments: Iterable[Dict]) -> List[Locus]:
    out = []
    for f in fragments:
        chain = [p.strip() for p in f["witness"].split("|") if p.strip() and not p.strip().startswith("heading:")]
        if not chain:
            continue
        name = None
        for part in chain:                       # the text is named once; "Idem ..." parts inherit it
            name = canon_witness(part) or name
        out.append(Locus("FHG", f["author"] or "", f["n"], name, parse_locus(name, chain[-1]), " | ".join(chain)))
    return out


# A fragment's source line: optional printer's line number, fragment number (+letter),
# optional FHG reference in brackets, then the source and its passage up to the colon.
# The source runs to the colon, or for lexicon entries ("STEPH. BYZ. s. Alneıa' nölıs") to the
# quote mark after the headword, or at most 90 characters.
_SRC = r"(?P<src>(?:[A-ZÄÖÜ][A-Za-zÄÖÜäöü]{1,}\.?|—|-)[^:'‘’·]{0,90})"
_SOURCE_LINE = re.compile(
    r"^\s*(?:\d{1,2}\s+)?(?P<n>\d{1,3})\s*(?P<letter>[a-e])?\)?\s*"
    r"(?:\((?P<conc>[^)]{1,18})\)\s*)?" + _SRC)
# A further source for the same fragment, lettered and often mid-line: "... b) SCHOL. PINDAR. P III 96:"
_SUB_SOURCE = re.compile(r"(?:^|\s)(?P<letter>[a-h])\)\s*(?:\((?P<conc>[^)]{1,18})\)\s*)?" + _SRC)
# "3. PHEREKYDES VON ATHEN", and in part III with a date or note after the name:
# "298. AUTESION s.III/II s ?", "304. DEMETRIOS (DAMEN) <VON ARGOS> 5. IVa". The name stops at the
# first lowercase word or number.
_AUTHOR_HEAD = re.compile(r"^\s*(?:\d{1,4}\s+)?(\d{1,3})\.\s+([A-ZÄÖÜ]{3,}(?:[ .\-]+[A-ZÄÖÜ()<>][A-ZÄÖÜ()<>\-]*)*)"
                          r"\.?(?:\s+\S.*)?\s*$")
# A running head over a page that spans several authors: "299. PYTHAINETOS -- 300. THEOGENES 3".
_RUNNING_HEAD = re.compile(r"\s(?:—|--|–)\s*\d{1,3}\.\s")
# Section headings inside an author ("2. LÄNDER UND VÖLKER") look like author heads.
_NOT_AUTHOR = re.compile(r"\b(?:UND|ÜBER|BUCH|BÜCHER|FRAGMENTE|ZEUGNISSE|ANHANG|NACHTRÄGE)\b")
# Apparatus lines start with a line number too, and name the editors whose readings they report.
_APPARATUS = re.compile(r"\b(?:Dindorf|Meineke|Kramer|Muenzer|Münzer|Bekker|Nauck|Wilamowitz|Schwartz|Jacobs|"
                        r"Casaubonus|Kaibel|Diels|Reitzenstein|Wachsmuth|Hercher|Sintenis|Valck|Lobeck|Leopardi|"
                        r"Gregoire|Jac)\b|\b(?:codd?|om)\.")
# An Athenaeus source in FGrHist starts with book, chapter and page: "ATHEN. XII 59 p. 542 A",
# "ATHEN. 11,2 p. 460 D", "ATHENAI. II 70 A". An apparatus line that mentions Athenaeus does not
# ("ATHEN. 7 Athen. 1, 5 p. 4 A 6 αὐτὸς ..."), and its page is not this fragment's.
_ATHENAEUS_SOURCE = re.compile(r"^\S+\.?\s*[IVXLlOo\d]{1,5}\s*[,.]?\s*(?:[\dIlOo]{1,3}\s*[,.]?\s*)?(?:p\.\s*)?"
                               r"\d{1,3}\s*[,.]?\s*[A-FΑΒΕΔÀ]")


def from_jacoby(text: str, source: str) -> List[Locus]:
    """Source lines from one FGrHist volume's OCR text."""
    out, author, last_src, last_n = [], "", "", ""
    number, names = 0, set()

    def add(n: str, letter: str, conc: str, src: str) -> None:
        nonlocal last_src
        src = src.strip()
        repeat = src.startswith(("—", "-"))
        if repeat:                                # "— XIII 60,5:" repeats the previous source
            src = (last_src + " " + src.lstrip("—- ")).strip()
        name = canon_witness(src)
        if not name and not re.match(r"[A-ZÄÖÜ]{3,}", src):
            return                                # a numbered line of Greek or German, not a source line
        if not re.search(r"\d|\bs\.(?:\s|$)|\bs\.v\.", src):
            return                                # a real citation has a number or a lexicon headword ("s. Alneia")
        if _APPARATUS.search(src):
            return                                # a line of the critical apparatus, which names editors
        if not repeat:                            # only a line that passed the checks can be repeated
            last_src = re.split(r"\s+(?=[IVXL]+\b|\d)", src, maxsplit=1)[0]
        locus = parse_locus(name, src)
        if name == "athenaeus" and not _ATHENAEUS_SOURCE.match(src):
            locus = None
        if name == "homer_scholia" and locus:      # Jacoby's Greek book letters don't survive OCR (δ -> ὃ, Γ -> T)
            poem, _, line = locus.split(".")
            locus = f"{poem}.?.{line}"
        out.append(Locus("FGrHist", author, f"{n}{letter or ''}", name, locus, src[:120],
                         (conc or "").strip()))

    for line in text.split("\n"):
        if _RUNNING_HEAD.search(line):
            continue
        head = _AUTHOR_HEAD.match(line)
        if head and _is_author_head(head, line, number, names):
            n, name = int(head.group(1)), head.group(2).strip(" .-")
            author, last_n = f"{n} {name}", ""
            if name not in names:
                number = n
                names.add(name)
            continue
        if not author:
            continue
        m = _SOURCE_LINE.match(line)
        if m:
            last_n = m.group("n")
            add(last_n, m.group("letter"), m.group("conc"), m.group("src"))
            rest = line[m.end():]
        else:
            rest = line
        if last_n:
            for sub in _SUB_SOURCE.finditer(rest):
                if canon_witness(sub.group("src")) or re.match(r"[A-ZÄÖÜ]{3,}\.", sub.group("src")):
                    add(last_n, sub.group("letter"), sub.group("conc"), sub.group("src"))
    return _fix_author_numbers(out)


def _is_author_head(head, line: str, number: int, names: set) -> bool:
    """Whether a line shaped like "N. NAME ..." really starts (or repeats) an author.

    Three things look the same: a source line whose fragment number has a dot ("107. STEPH. BYZ. s.
    ..."), a book title inside an author ("1. EYBOIKA", Greek OCR'd as Latin capitals) and a
    section heading ("2. LÄNDER UND VÖLKER"). Authors are numbered upward through a volume, so a new
    name far below or far above the current number is not one.
    """
    name, rest = head.group(2).strip(" .-"), line[head.start(2):]
    if _NOT_AUTHOR.search(name) or canon_witness(rest) or re.match(r"[A-ZÄÖÜ]+\.\s*\S", rest):
        return False
    if name in names:
        return True
    n = int(head.group(1))
    return not names or number - 5 <= n <= number + 40


def _fix_author_numbers(rows: List[Locus]) -> List[Locus]:
    """OCR sometimes misreads a running head's number ("3. HEKATAIOS"); use each name's usual number."""
    seen: Dict[str, List[str]] = {}
    for r in rows:
        n, _, name = r.author.partition(" ")
        seen.setdefault(name, []).append(n)
    usual = {name: max(set(ns), key=ns.count) for name, ns in seen.items()}
    for r in rows:
        n, _, name = r.author.partition(" ")
        r.author = f"{usual[name]} {name}"
    return rows


# ---------------------------------------------------------------------------
# Our texts: where in the citation scheme a word position falls
# ---------------------------------------------------------------------------

def witness_locus(witness, pos: int, headwords: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The normalised passage for word ``pos`` of a loaded witness, if its scheme is mapped."""
    loc = witness.loc[pos]
    if witness.name == "athenaeus":
        m = re.search(r"p\.(\d+)([a-f])", loc)
        return f"{int(m.group(1))}{m.group(2)}" if m else None
    if witness.name == "pindar_scholia":
        # Our files a-d are the Olympians, Pythians, Nemeans, Isthmians; loc = file.poem.scholion[.part]
        parts = loc.split(".")
        if len(parts) >= 3 and parts[0] in "1234abcd" and parts[2].isdigit():
            ode = "OPNI"["1234".index(parts[0])] if parts[0].isdigit() else "OPNI"["abcd".index(parts[0])]
            return f"{ode}.{parts[1]}.{int(parts[2])}" if parts[1].isdigit() else None
        return None
    if witness.name == "suda":
        return (headwords or suda_headwords(witness)).get(loc)
    if witness.name == "euripides_scholia":
        # loc = "sch_Hec.123[.part]". One division (Alcestis) has no name; it follows "arg_Alc".
        work, _, rest = loc.partition(".")
        if work == "?":
            work = "sch_" + _unnamed_euripides(witness).get(pos, "")
        line = re.match(r"\d+", rest)
        return f"{work[4:]}.{int(line.group())}" if work.startswith("sch_") and len(work) > 4 and line else None
    if witness.name == "sophocles_scholia":
        play, _, rest = loc.partition(".")                  # "7.1053" = play 7 (OC), line 1053
        line = re.match(r"\d+", rest)
        return f"{SOPHOCLES_ORDER[int(play) - 1]}.{int(line.group())}" if play.isdigit() and line else None
    if witness.name == "homer_scholia":
        # "file.volume.book.section v.verse": file 1 the Iliad, 2 the Odyssey. The section is a running
        # number, not the line; the verse comes from the scholion's lemma ("307. ...").
        parts = loc.split(" ")[0].split(".")
        verse = re.search(r" v\.(\d+)", loc)
        if len(parts) < 3 or not parts[2].isdigit() or not verse:
            return None
        return f"{'Il' if parts[0] == '1' else 'Od'}.{int(parts[2])}.{int(verse.group(1))}"
    return None


def _unnamed_euripides(witness) -> Dict[int, str]:
    """Word position -> play, for the one unnamed division of the Euripides scholia (cached)."""
    if not hasattr(witness, "_unnamed_plays"):
        plays, last = {}, ""
        for i, loc in enumerate(witness.loc):
            work = loc.partition(".")[0]
            if work.startswith("arg_"):
                last = work[4:]
            elif work == "?":
                plays[i] = last
        witness._unnamed_plays = plays
    return witness._unnamed_plays


def suda_headwords(witness) -> Dict[str, str]:
    """Suda entry location -> its headword (the entry's first Greek word)."""
    heads = {}
    for loc, tok in zip(witness.loc, witness.tok):
        heads.setdefault(loc, tok)
    return heads


def _athenaeus_index(locus: str) -> Optional[int]:
    m = re.fullmatch(r"(\d+)([a-f])", locus)
    return int(m.group(1)) * 6 + "abcdef".index(m.group(2)) if m else None


def near(witness: str, a: Optional[str], b: Optional[str], tolerance: int = 0) -> bool:
    """Do two normalised passages refer to the same place, within ``tolerance``?

    Athenaeus: steps of a Casaubon section (542a -> 542b is 1; a page has a-f).
    Pindar scholia: lines/scholion numbers within the same ode and poem."""
    if not a or not b:
        return False
    if a == b:
        return True
    if witness == "athenaeus":
        x, y = _athenaeus_index(a), _athenaeus_index(b)
        return x is not None and y is not None and abs(x - y) <= tolerance
    if witness in ("pindar_scholia", "euripides_scholia", "sophocles_scholia", "homer_scholia"):
        # "work[.book].line": the same work and book ("?" = unreadable book, matches any), lines close.
        x, y = a.split("."), b.split(".")
        if len(x) != len(y) or not (x[-1].isdigit() and y[-1].isdigit()):
            return False
        same = all(p == q or "?" in (p, q) for p, q in zip(x[:-1], y[:-1]))
        return same and abs(int(x[-1]) - int(y[-1])) <= tolerance
    return False


# How far apart two citations of one passage can be, in each text's units: Casaubon sections;
# lines or scholion numbers (editions number scholia differently: Mueller cites the Euripides
# scholia by older line numbers, 5-20 off Schwartz's); none for Suda headwords.
TOLERANCE = {"athenaeus": 1, "pindar_scholia": 2, "suda": 0, "euripides_scholia": 10, "sophocles_scholia": 10,
             "homer_scholia": 3}


def table_author(row: Locus) -> str:
    """Name key of a table row's author: "244 APOLLODOROS V. ATHEN" and "APOLLODORUS" both give "apollodor"."""
    return name_key(re.sub(r"^\d+\s+", "", row.author))


def as_rows(loci: Iterable[Locus]) -> List[Dict]:
    return [asdict(l) for l in loci]
