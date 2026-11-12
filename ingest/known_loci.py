"""Which passages are already collected as fragments: a table of (text, passage) pairs.

A lead found at a passage some collection already prints is known, however its
wording came out in our OCR or theirs. This module builds that table from two
sources and maps our own texts' locations onto the same scheme.

  FHG       DFHG's source field for each fragment ("Athen. XII, p. 527, F").
  FGrHist   the source line printed above each fragment in Jacoby's *Die
            Fragmente der griechischen Historiker* I-II (1923-30), read from the
            Vanderbilt scans on archive.org. Their Greek OCR is unusable (it came
            out as Latin letters), but these lines are in Latin script and read
            well, e.g. "4 (8) ATHEN. XII 59 p. 542 A: ...". The bracket is the
            matching FHG reference, so the line also gives a concordance.

Only parts I-II: they are public domain in the US. Part III (1940-58) is not.

Passages are normalised per text, in the scheme that text's editors cite:

  athenaeus        Casaubon page and letter      "542a"
  suda             headword, accent-free          "ομηροσ"
  pindar_scholia   ode, poem, line                "O.7.24"

Other texts are kept with their raw citation (``locus`` None) so the table
still says which sources a collection used. Citation schemes differ between
editions (Pindar scholia: Boeckh's line numbers in FHG, Drachmann's scholion
numbers in our text), so matching takes a tolerance; see ``near``.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from ingest.quotations import norm

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
    return None


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
    return None


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
    if witness == "pindar_scholia":
        x, y = a.split("."), b.split(".")
        return x[:2] == y[:2] and abs(int(x[2]) - int(y[2])) <= tolerance
    return False


def as_rows(loci: Iterable[Locus]) -> List[Dict]:
    return [asdict(l) for l in loci]
