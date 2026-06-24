"""Generic TEI-XML connector (Perseus, Patrologia, EpiDoc, ...).

Parses a TEI document from a URL or a local path. Pulls title/author from the
teiHeader and turns the <body>'s <div> hierarchy into sections, so books and
chapters survive as structure. Editorial <note>/<bibl>/apparatus elements are
dropped. Stdlib xml.etree only.

Usage:
    from ingest.tei import TEIConnector
    meta, parts = TEIConnector().fetch("https://.../caesar.xml")
    meta, parts = TEIConnector().fetch("local/path/work.xml")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import re
import xml.etree.ElementTree as ET

import requests

from .base import Connector, RawWork


TEI_NS = "http://www.tei-c.org/ns/1.0"
# Editorial/apparatus elements to drop from the body. (teiHeader is kept so we
# can read title/author; body extraction only ever reads <body>.)
_DROP_TAGS = {"note", "bibl", "ref", "gap", "del", "orig", "fw"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class TEIConnector(Connector):
    name = "tei"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly research)"}
        )

    def fetch(self, source: str, **meta_overrides) -> RawWork:
        root = self._load(source)
        return self.parse_root(root, self._basename(source), **meta_overrides)

    def parse_xml(self, xml: "str | bytes", label: str = "tei",
                  **meta_overrides) -> RawWork:
        """Parse an in-memory TEI document (used by source-specific connectors)."""
        root = ET.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
        return self.parse_root(root, label, **meta_overrides)

    def parse_root(self, root: ET.Element, label: str, **meta_overrides) -> RawWork:
        self._strip(root, _DROP_TAGS)

        title = self._header_text(root, "title") or label
        author = (self._header_text(root, "author")
                  or self._header_text(root, "persName")
                  or self._respstmt_name(root))

        body = self._find(root, "body")
        parts: List[Tuple[str, str]] = []
        if body is not None:
            parts = self._sections_from_body(body)
        if not parts:  # no usable divs: take the whole body/text as one section
            text = self._text_of(body if body is not None else root)
            if text:
                parts = [("Text", text)]

        meta = {
            "title": title,
            "author": author,
            "source": f"TEI ({label})",
            "language_stage": "unknown",
            "has_existing_translation": False,
        }
        meta.update(meta_overrides)
        return meta, parts

    def discover(self, directory: str, limit: int = 200) -> List[str]:
        """List .xml files under a local directory for bulk ingestion."""
        paths = sorted(str(p) for p in Path(directory).rglob("*.xml"))
        return paths[:limit]

    # -- parsing helpers -----------------------------------------------------

    def _load(self, source: str) -> ET.Element:
        if re.match(r"^https?://", source):
            resp = self.session.get(source, timeout=self.timeout)
            resp.raise_for_status()
            return ET.fromstring(resp.content)
        return ET.parse(source).getroot()

    def _sections_from_body(self, body: ET.Element) -> List[Tuple[str, str]]:
        """Each leaf <div> (no child <div>) becomes a section; label from @n/@type."""
        sections: List[Tuple[str, str]] = []

        def walk(elem: ET.Element, trail: List[str]):
            child_divs = [c for c in elem if _local(c.tag) == "div"]
            label_bit = self._div_label(elem)
            here = trail + ([label_bit] if label_bit else [])
            if child_divs:
                for c in child_divs:
                    walk(c, here)
            else:
                text = self._text_of(elem)
                if text:
                    sections.append((", ".join(here) or "Text", text))

        top_divs = [c for c in body if _local(c.tag) == "div"]
        if not top_divs:
            return []
        for d in top_divs:
            walk(d, [])
        return sections

    @staticmethod
    def _div_label(div: ET.Element) -> str:
        typ = (div.get("type") or "").strip().lower()
        # The CTS "edition"/"translation" wrapper carries a urn in @n; skip it.
        if typ in ("edition", "translation"):
            return ""
        # Prefer the specific subtype ("book"/"chapter") over generic "textpart".
        kind = div.get("subtype") or (typ if typ and typ != "textpart" else "")
        n = (div.get("n") or "").strip()
        kind = kind.strip().capitalize()
        if kind and n:
            return f"{kind} {n}"
        if kind:
            return kind
        return f"Section {n}" if n else ""

    @staticmethod
    def _text_of(elem: Optional[ET.Element]) -> str:
        if elem is None:
            return ""
        text = " ".join(elem.itertext())
        return re.sub(r"\s+", " ", text).strip()

    def _header_text(self, root: ET.Element, tag: str) -> Optional[str]:
        header = self._find(root, "teiHeader")
        scope = header if header is not None else root
        for el in scope.iter():
            if _local(el.tag) == tag and el.text and el.text.strip():
                return el.text.strip()
        return None

    def _respstmt_name(self, root: ET.Element) -> Optional[str]:
        """DigilibLT and some TEI put the author in <respStmt><name>/<persName>."""
        header = self._find(root, "teiHeader")
        scope = header if header is not None else root
        for el in scope.iter():
            if _local(el.tag) == "respStmt":
                for child in el.iter():
                    if _local(child.tag) in ("persName", "name") and child.text and child.text.strip():
                        return child.text.strip()
        return None

    @staticmethod
    def _find(root: ET.Element, localname: str) -> Optional[ET.Element]:
        for el in root.iter():
            if _local(el.tag) == localname:
                return el
        return None

    @staticmethod
    def _strip(root: ET.Element, drop: set) -> None:
        """Remove unwanted elements (notes, apparatus) in place."""
        parent_map = {c: p for p in root.iter() for c in p}
        for el in list(parent_map):
            if _local(el.tag) in drop and el in parent_map:
                parent = parent_map[el]
                if el in list(parent):
                    parent.remove(el)

    @staticmethod
    def _basename(source: str) -> str:
        return re.sub(r"\.xml$", "", source.rstrip("/").rsplit("/", 1)[-1])
