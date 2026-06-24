"""Connector for Corpus Corporum (mlat.uzh.ch), Zurich.

Corpus Corporum is a very large medieval/patristic corpus (it includes the whole
Patrologia Latina). Its site is a JS front end over a BaseX store; we use the two
backend endpoints it calls:
    php_modules/display_text.php?ajax=true&path=<text idno>   -> a work's text (XML)
    php_modules/navigate.php?load=<idno>                      -> the catalogue tree

fetch(text_idno) returns one work. discover(corpus_idno) walks the tree from a
corpus node and returns the text idnos beneath it. List the top-level corpora
with top_corpora().

Note: display_text returns the text's default loaded section; very long works may
be truncated to that section (a known limitation to revisit).
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import re
import xml.etree.ElementTree as ET

import requests

from .base import Connector, RawWork


CC = "https://mlat.uzh.ch/php_modules/"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class CorpusCorporumConnector(Connector):
    name = "corpuscorporum"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "LatinReader-Research/1.0 (scholarly; jth156@case.edu)",
            "X-Requested-With": "XMLHttpRequest",
        })

    # -- fetch ---------------------------------------------------------------

    def fetch(self, text_idno: str, **meta_overrides) -> RawWork:
        idno = str(text_idno).strip()
        root = self._get_xml("display_text.php", {"ajax": "true", "path": idno})

        author = self._first_text(root, "author")
        title = self._first_text(root, "name") or f"Corpus Corporum {idno}"
        prose = self._extract_prose(root)

        meta = {
            "title": title,
            "author": author,
            "source": f"Corpus Corporum ({idno})",
            "license": "Corpus Corporum (mlat.uzh.ch); check per-text terms",
            "language_stage": "medieval",
            "has_existing_translation": False,
        }
        meta.update(meta_overrides)
        return meta, [("Text", prose)]

    # -- discover ------------------------------------------------------------

    def top_corpora(self) -> List[Tuple[str, str]]:
        """Return (idno, name) for the top-level corpora to start discovery."""
        root = self._get_xml("navigate.php", {"load": ""})
        out = []
        for corp in root.iter():
            if _local(corp.tag) == "corpus":
                idno = self._child_text(corp, "idno")
                name = self._child_text(corp, "name")
                if idno:
                    out.append((idno, name or idno))
        return out

    def discover(self, corpus_idno: str, limit: int = 25) -> List[str]:
        """Walk the tree from a corpus idno and return text idnos beneath it."""
        found: List[str] = []
        seen: set = set()
        # Depth-first (stack): dive to leaf texts quickly rather than fanning out
        # across every author of a huge corpus before reaching any text.
        stack: List[str] = [str(corpus_idno).strip()]
        max_requests = max(limit * 8, 60)
        requests_made = 0

        while stack and len(found) < limit and requests_made < max_requests:
            load = stack.pop()
            if load in seen:
                continue
            seen.add(load)
            requests_made += 1
            try:
                root = self._get_xml("navigate.php", {"load": load})
            except Exception:
                continue

            # Any <text> nodes in this response are terminal — collect their idno.
            for el in root.iter():
                if _local(el.tag) == "text":
                    tid = el.get("cc_idno") or self._descendant_text(el, "idno")
                    if tid and tid not in found:
                        found.append(tid)
            if len(found) >= limit:
                break

            # Drill the children of <contents> (skip the path-step header).
            contents = self._find(root, "contents")
            if contents is None:
                continue
            children = []
            for child in list(contents):
                if _local(child.tag) == "text":
                    continue
                cid = child.get("cc_idno") or self._child_text(child, "idno")
                if cid and cid not in seen:
                    children.append(cid)
            # Push in reverse so the first child is explored first (depth-first).
            for cid in reversed(children):
                stack.append(cid)
        return found[:limit]

    # -- xml helpers ---------------------------------------------------------

    def _get_xml(self, endpoint: str, params: dict) -> ET.Element:
        resp = self.session.get(CC + endpoint, params=params, timeout=self.timeout)
        resp.raise_for_status()
        if "frontend index" in resp.text[:200]:
            raise ValueError(f"{endpoint} returned the site shell (no data)")
        return ET.fromstring(resp.content)

    def _extract_prose(self, root: ET.Element) -> str:
        """Concatenate text from <div> elements (the prose); metadata lives elsewhere."""
        chunks: List[str] = []
        for el in root.iter():
            if _local(el.tag) == "div":
                text = " ".join(el.itertext())
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    chunks.append(text)
        if not chunks:  # fall back to any <p>/<l> lines
            for el in root.iter():
                if _local(el.tag) in ("p", "l"):
                    t = re.sub(r"\s+", " ", " ".join(el.itertext())).strip()
                    if t:
                        chunks.append(t)
        return "\n".join(chunks)

    @staticmethod
    def _find(root: ET.Element, localname: str) -> Optional[ET.Element]:
        for el in root.iter():
            if _local(el.tag) == localname:
                return el
        return None

    @staticmethod
    def _first_text(root: ET.Element, localname: str) -> Optional[str]:
        for el in root.iter():
            if _local(el.tag) == localname and el.text and el.text.strip():
                return el.text.strip()
        return None

    @staticmethod
    def _child_text(parent: ET.Element, localname: str) -> Optional[str]:
        for c in parent:
            if _local(c.tag) == localname and c.text and c.text.strip():
                return c.text.strip()
        return None

    @staticmethod
    def _descendant_text(parent: ET.Element, localname: str) -> Optional[str]:
        for c in parent.iter():
            if c is not parent and _local(c.tag) == localname and c.text and c.text.strip():
                return c.text.strip()
        return None
