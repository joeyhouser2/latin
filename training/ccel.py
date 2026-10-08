"""Extract structured English text from CCEL's ThML editions of the Ante-Nicene
Fathers (ANF) and Nicene & Post-Nicene Fathers (NPNF).

These are the public-domain English translations of the Church Fathers. CCEL
serves each volume as ThML (a TEI-like markup) at
    https://www.ccel.org/ccel/schaff/<volume>.xml
e.g. anf03 = Tertullian. We parse the div hierarchy: works live at <div2>, their
chapters at <div3>. This is the English half of a patristic Latin parallel corpus
(the Latin half comes from Patrologia Latina; see align step).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import re
import xml.etree.ElementTree as ET

import requests


CCEL_XML = "https://www.ccel.org/ccel/schaff/{volume}.xml"
_DROP = {"note", "scripRef", "pb", "index", "figure"}  # apparatus / non-prose


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman(text: str) -> int:
    """Roman numeral -> int, or 0 if it isn't one (CCEL numbers books 'I'..'XXII')."""
    s = (text or "").strip().upper()
    if not s or any(c not in _ROMAN_VALUES for c in s):
        return 0
    total = 0
    for i, c in enumerate(s):
        v = _ROMAN_VALUES[c]
        nxt = _ROMAN_VALUES.get(s[i + 1]) if i + 1 < len(s) else None
        total += -v if nxt and nxt > v else v
    return total


@dataclass
class EnglishWork:
    volume: str
    title: str                       # work title, e.g. "The Apology"
    chapters: List[Tuple[str, str]]  # (chapter_title, text)


class CCELExtractor:
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly; jth156@case.edu)"}
        )

    def fetch_volume(self, volume: str) -> ET.Element:
        resp = self.session.get(CCEL_XML.format(volume=volume), timeout=self.timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        self._strip(root, _DROP)
        return root

    def books(self, volume: str, work_title: str,
              root: ET.Element = None) -> List[Tuple[int, str]]:
        """[(book_number, prose)] for one multi-book work, e.g. the Confessions.

        The work's container and the book level both move between volumes, so
        this matches on @title at any depth and then takes whatever descendant
        divisions are marked ``type="Book"``:

            npnf101  div1 "The Confessions"      -> div2 type=Book, n=I..XIII
            npnf102  div1 "City of God"          -> div2 type=Book, n=I..XXII
            npnf103  div2 "On the Holy Trinity." -> div3 type=Book, n=I..XV

        `works()` is no help here: it treats div2 as the work, so for npnf101 it
        reports 13 "works" whose titles are really book summaries. Pass ``root``
        to reuse an already-fetched volume (they are several MB each).

        The book numbers are what make per-book alignment possible: the Latin
        side (e.g. The Latin Library's augustine/conf<N>.shtml) is one page per
        book, and aligning book against book beats aligning whole works.
        """
        if root is None:
            root = self.fetch_volume(volume)
        want = work_title.lower()
        container = next(
            (e for e in root.iter()
             if _local(e.tag).startswith("div") and want in (e.get("title") or "").lower()),
            None,
        )
        if container is None:
            return []
        out: List[Tuple[int, str]] = []
        for div in (e for e in container.iter() if _local(e.tag).startswith("div")):
            if (div.get("type") or "").lower() != "book":
                continue
            n = _roman(div.get("n") or "")
            text = self._prose(div)
            if n and text:
                out.append((n, text))
        return sorted(out)

    def works(self, volume: str) -> List[EnglishWork]:
        """Return the volume's works (div2) with their chapters (div3)."""
        root = self.fetch_volume(volume)
        out: List[EnglishWork] = []
        for d2 in (e for e in root.iter() if _local(e.tag) == "div2"):
            chapters: List[Tuple[str, str]] = []
            for d3 in (c for c in d2.iter() if _local(c.tag) == "div3"):
                text = self._prose(d3)
                if text:
                    chapters.append((self._title(d3), text))
            if chapters:
                out.append(EnglishWork(volume, self._title(d2), chapters))
        return out

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _title(div: ET.Element) -> str:
        t = div.get("title")
        if t:
            return t.strip()
        head = next((e for e in div.iter() if _local(e.tag) == "head"), None)
        return re.sub(r"\s+", " ", " ".join(head.itertext())).strip() if head is not None else ""

    @staticmethod
    def _prose(div: ET.Element) -> str:
        """Join <p> prose within the division (notes already stripped)."""
        paras = []
        for p in div.iter():
            if _local(p.tag) == "p":
                txt = re.sub(r"\s+", " ", " ".join(p.itertext())).strip()
                if txt:
                    paras.append(txt)
        return "\n".join(paras)

    @staticmethod
    def _strip(root: ET.Element, drop: set) -> None:
        """Remove apparatus elements, keeping the prose that follows them.

        ElementTree's ``remove`` also discards the element's ``tail`` -- and in
        ThML a footnote marker sits *inside* the paragraph it annotates, so the
        rest of the sentence lives in ``<note>.tail``. Dropping notes naively
        therefore deleted every sentence remainder after a footnote: on ANF vol
        3's Apology that was 52% of the English (98,914 chars kept out of
        206,474). We splice each tail onto the preceding sibling (or the
        parent's text) before removing the node.
        """
        def clean(parent: ET.Element) -> None:
            for child in list(parent):
                clean(child)               # depth-first, so nested drops resolve
            for el in list(parent):
                if _local(el.tag) not in drop:
                    continue
                tail = el.tail or ""
                if tail:
                    siblings = list(parent)
                    idx = siblings.index(el)
                    if idx == 0:
                        parent.text = (parent.text or "") + tail
                    else:
                        prev = siblings[idx - 1]
                        prev.tail = (prev.tail or "") + tail
                parent.remove(el)

        clean(root)
