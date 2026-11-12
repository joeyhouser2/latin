"""Read a catena into items, each with the author its label gives (phase 4 of docs/quotation-mining-plan.md).

A catena strings together excerpts from earlier commentators, each headed by its author in the
genitive: "Κυρίλλου.", "Θεοδωρήτου.", "Σευήρου Ἀντιοχείας ἐν Ὑπακοῇ.", "Ἐξ ἀνεπιγράφου.". The
label is the attribution, so no citing formula is needed and namesakes matter less. Many of the
authors' commentaries are lost (Apollinaris, Theodore of Mopsuestia, Didymus, Severus in Greek,
Photius on Paul, Ammonius), which makes the items the fragments.

Text: Cramer, Catenae Graecorum Patrum in Novum Testamentum (1838-44), First1KGreek TEI tlg4102,
24 catenae (registered as the witness "cramer_catenae" in ingest.quotations).

The reader finds a label at the start of a paragraph or right after a full stop: one to four words
ending in a full stop, the first a known author's genitive. Cramer's OCR misspells some (Κυτίλλου,
Ἰωάλλου, Ὠριγένουσ), so names are matched by an accent-free key with one letter of slack.
"""
from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ingest.quotations import TEI, norm

# Accent-free genitive (as norm() gives it, final sigma folded) -> author.
AUTHORS = {
    "χρυσοστομου": "CHRYSOSTOM", "ιωαννου": "CHRYSOSTOM",   # "Ἰωάννου (τοῦ Χρυσοστόμου)" in these catenae
    "κυριλλου": "CYRIL", "θεοδωρητου": "THEODORET", "θεοδωριτου": "THEODORET",
    "θεοδωρου": "THEODORE", "αμμωνιου": "AMMONIUS", "σευηριανου": "SEVERIAN", "σευηρου": "SEVERUS",
    "ωριγενουσ": "ORIGEN", "γενναδιου": "GENNADIUS", "φωτιου": "PHOTIUS", "απολιναριου": "APOLLINARIS",
    "απολλιναριου": "APOLLINARIS", "διδυμου": "DIDYMUS", "οικουμενιου": "OECUMENIUS", "γρηγοριου": "GREGORY",
    "διοδωρου": "DIODORE", "ισιδωρου": "ISIDORE", "βασιλειου": "BASIL", "ευσεβιου": "EUSEBIUS",
    "αθανασιου": "ATHANASIUS", "ακακιου": "ACACIUS", "τιτου": "TITUS", "ιππολυτου": "HIPPOLYTUS",
    "ειρηναιου": "IRENAEUS", "κλημεντοσ": "CLEMENT", "βικτοροσ": "VICTOR", "μαξιμου": "MAXIMUS",
    "αναστασιου": "ANASTASIUS", "ευαγριου": "EVAGRIUS", "νειλου": "NILUS", "ανεπιγραφου": "ANONYMOUS",
    "ιουστινου": "JUSTIN", "μεθοδιου": "METHODIUS", "ευστα0ιου": "EUSTATHIUS", "ευσταθιου": "EUSTATHIUS",
    "ηλια": "ELIAS", "πολυχρονιου": "POLYCHRONIUS", "αρεθα": "ARETHAS", "ανδρεου": "ANDREW",
    # Some labels are nominative, in brackets: "(Θεόδωρος.)", "(Σευηριανός.)".
    "θεοδωροσ": "THEODORE", "σευηριανοσ": "SEVERIAN", "ευσεβιοσ": "EUSEBIUS", "χρυσοστομοσ": "CHRYSOSTOM",
    "κυριλλοσ": "CYRIL", "ωριγενησ": "ORIGEN", "διδυμοσ": "DIDYMUS", "φωτιοσ": "PHOTIUS", "γενναδιοσ": "GENNADIUS",
    "θεοδωρητοσ": "THEODORET", "απολιναριοσ": "APOLLINARIS", "αμμωνιοσ": "AMMONIUS", "σευηροσ": "SEVERUS",
}
# Words that may precede or follow the name in a label: "Τοῦ", "Ἐξ", "ἁγίου", sees and offices.
_LABEL_FILLER = {"του", "εκ", "εξ", "αγιου", "αγιωτατου", "αρχιεπισκοπου", "επισκοπου", "πρεσβυτερου", "μοναχου",
                 "αλεξανδρειασ", "αντιοχειασ", "ηρακλειασ", "μοψουεστιασ", "κυρου", "νυσησ", "νυσσησ",
                 "ναζιανζηνου", "θεολογου", "μεγαλου", "χρυσοστομου", "ταρσου", "λαοδικειασ", "κωνσταντινουπολεωσ", "βοστρων", "και", "μομψουεστιασ", "κυρρου", "ιεροσολυμων", "καισαρειασ",
                 "εν", "υπακοηι", "υπακοη", "των", "κατα", "λογου", "λογω", "ομιλιασ"}


def _key(word: str) -> str:
    # "Toῦ" with a Latin T and o: fold Latin look-alikes before dropping non-Greek letters.
    word = word.translate(str.maketrans("ToOoKkAaEeHhIiMmNnPpXxYy", "ΤοΟοΚκΑαΕεΗηΙιΜμΝνΡρΧχΥυ"))
    return norm(word.strip("().·,;“”\"'’ "))


def _filler(word: str) -> bool:
    k = _key(word)
    return k in _LABEL_FILLER or any(_close(k, f) for f in _LABEL_FILLER)


def _close(a: str, b: str) -> bool:
    """Equal, or one substitution apart (Κυτίλλου for Κυρίλλου), for words of 6+ letters."""
    if a == b:
        return True
    return len(a) == len(b) >= 6 and sum(x != y for x, y in zip(a, b)) == 1


def author_of(word: str) -> Optional[str]:
    k = _key(word)
    if k in AUTHORS:
        return AUTHORS[k]
    for name, author in AUTHORS.items():
        if _close(k, name):
            return author
    return None


@dataclass
class Item:
    work: str                 # catena file, e.g. tlg4102.tlg010
    loc: str                  # chapter as the TEI numbers it
    label: str                # the label as printed, "" for text before the first label
    author: Optional[str]     # from the label's name; None = unlabelled
    text: str
    toks: List[str] = field(default_factory=list)


_SENTENCE = re.compile(r"(?<=[.·;])\s+")


def split_labels(text: str) -> List[Tuple[str, str]]:
    """[(label, following text)]: a paragraph cut wherever a label starts it or follows a full stop."""
    pieces = _SENTENCE.split(text.strip())
    out: List[Tuple[str, str]] = []
    label, body = "", []
    i = 0
    while i < len(pieces):
        piece = pieces[i]
        found = _label_at(piece)
        if found:
            if body or label:
                out.append((label, " ".join(body)))
            label, rest = found
            body = [rest] if rest else []
        else:
            body.append(piece)
        i += 1
    if body or label:
        out.append((label, " ".join(body)))
    return out


def _label_at(piece: str) -> Optional[Tuple[str, str]]:
    """If a piece starts with a label: up to 8 words ending at a full stop, among them an author's
    name, otherwise only titles and sees ("Τοῦ ἁγίου Τίτου ἐπισκόπου Βόστρων.") or, after "ἐκ"/"ἐν",
    the work quoted ("Κυρίλλου ἐκ τοῦ κατὰ Ἰουλιανοῦ.")."""
    words = piece.split()
    if not words or not (words[0].lstrip("(")[:1].isupper() or author_of(words[0])):
        return None
    for n in range(1, min(8, len(words)) + 1):
        if not words[n - 1].rstrip(")").endswith("."):
            continue
        head = words[:n]
        names = [i for i, w in enumerate(head) if author_of(w)]
        if not names or names[0] > 3:
            return None
        work = next((i for i, w in enumerate(head) if _key(w) in ("εκ", "εξ", "εν") and i > names[0]), len(head))
        others = [w for w in head[:work] if not author_of(w) and not _filler(w)]
        if not others:
            return " ".join(head), " ".join(words[n:])
        # Garbled sees and titles ("Ἰσιδώρου Πηαοτσιώρου."): a short phrase that opens with an
        # author's genitive is a label too. Nominatives ("Θεόδωρος δέ φησι.") are not: that is a report.
        if names[0] == 0 and n <= 5 and _key(head[0]).endswith(("ου", "ουσ")):
            return " ".join(head), " ".join(words[n:])
        return None
    return None


def read_catena(path, work: str) -> List[Item]:
    """Items of one catena TEI file, in order. Footnotes are skipped."""
    body = ET.parse(path).getroot().find(f"{TEI}text/{TEI}body")
    # An unlabelled paragraph is usually the biblical lemma, sometimes the previous item going on:
    # it gets no author rather than a guessed one.
    items: List[Item] = []
    for div in body.iter(f"{TEI}div"):
        if div.get("type") != "textpart":
            continue
        loc = div.get("n") or "?"
        for p in div.findall(f"{TEI}p"):
            for label, chunk in split_labels(_text(p)):
                named = [a for a in (author_of(w) for w in label.split()) if a]
                if "δαμασκ" in _key(label.replace(" ", "")):   # "Ἰωάννου τοῦ Δαμασκέως" is not Chrysostom
                    named = ["JOHN_DAMASCENE"]
                toks = [norm(w) for w in re.findall(r"[Ͱ-Ͽἀ-῿]+", chunk)]
                if toks:
                    items.append(Item(work, loc, label, named[0] if named else None, chunk, toks))
    return items


def _text(el) -> str:
    """An element's text without footnotes."""
    parts = [el.text or ""]
    for child in el:
        if child.tag != f"{TEI}note":
            parts.append(_text(child))
        parts.append(child.tail or "")
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", "".join(parts)))
