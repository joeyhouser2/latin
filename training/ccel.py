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
        parents = {c: p for p in root.iter() for c in p}
        for el in list(parents):
            if _local(el.tag) in drop and el in parents:
                parent = parents[el]
                if el in list(parent):
                    parent.remove(el)
