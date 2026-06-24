"""Mine aligned source->English pairs from Perseus dual editions.

A Perseus work often ships both the original (`-lat` / `-grc`) and an English
(`-eng`) edition. They share a CTS citation hierarchy (book / chapter / section),
but usually at *different depths* — e.g. the Latin is cited to book.chapter.section
while the translation is only to book.chapter. We align at the deepest citation
level the two editions share in common, concatenating the finer side's text within
each shared unit. The result is chapter- (or section-) level parallel pairs.

This is coarser than sentence alignment, but it is real, license-clean parallel
data for fine-tuning, and pairs can be sentence-aligned later within each unit.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import re
import xml.etree.ElementTree as ET

import requests


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


# Citation wrapper levels that aren't real divisions of the text.
_WRAPPER_TYPES = {"edition", "translation"}


@dataclass
class ParallelPair:
    src: str            # Latin or Greek
    tgt: str            # English
    citation: str       # e.g. "1.2" (book.chapter)
    src_lang: str       # "la" | "grc"
    era: str            # language_stage, e.g. "classical"
    source: str         # provenance


class PerseusParallelMiner:
    """Align a Perseus work's source edition against its English edition."""

    def __init__(self, repo: str = "PerseusDL/canonical-latinLit",
                 src_tag: str = "lat", timeout: float = 30.0):
        self.repo = repo
        self.src_tag = src_tag
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly; jth156@case.edu)"}
        )

    # -- public --------------------------------------------------------------

    def mine(self, identifier: str, era: str = "classical",
             src_file: Optional[str] = None, eng_file: Optional[str] = None) -> List[ParallelPair]:
        group, work = self._parse(identifier)
        if src_file is None or eng_file is None:
            # Single-work path: one GitHub API call to find the edition filenames.
            files = [f["name"] for f in self._list_dir(f"data/{group}/{work}")
                     if f.get("type") == "file"]
            src_file = src_file or self._pick(files, self.src_tag)
            eng_file = eng_file or self._pick(files, "eng")
        if not src_file or not eng_file:
            return []  # need both an original and a translation to align

        src_sections = self._sections(group, work, src_file)
        eng_sections = self._sections(group, work, eng_file)
        return self._align(src_sections, eng_sections, era,
                           f"Perseus {group}.{work}")

    def alignable_works(self) -> Dict[str, Tuple[str, str]]:
        """One git-tree call -> {"group.work": (src_file, eng_file)} for every work
        that has both an original and an English edition. Avoids per-work API calls
        (and thus the 60/hour unauthenticated rate limit) when bulk-mining."""
        url = f"https://api.github.com/repos/{self.repo}/git/trees/master?recursive=1"
        tree = self.session.get(url, timeout=60).json().get("tree", [])

        by_dir: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for node in tree:
            path = node.get("path", "")
            parts = path.split("/")
            if path.endswith(".xml") and len(parts) == 4 and parts[0] == "data":
                by_dir[(parts[1], parts[2])].append(parts[3])

        out: Dict[str, Tuple[str, str]] = {}
        for (group, work), files in by_dir.items():
            src = self._pick(files, self.src_tag)
            eng = self._pick(files, "eng")
            if src and eng:
                out[f"{group}.{work}"] = (src, eng)
        return out

    # -- alignment -----------------------------------------------------------

    def _align(self, src: List[Tuple[list, str]], tgt: List[Tuple[list, str]],
               era: str, source: str) -> List[ParallelPair]:
        # Citation level names (book/chapter/section/...) present in each edition.
        src_levels = self._levels(src)
        tgt_levels = self._levels(tgt)
        common = [lvl for lvl in src_levels if lvl in tgt_levels]
        if not common:
            return []

        src_map = self._collapse(src, common)
        tgt_map = self._collapse(tgt, common)

        pairs: List[ParallelPair] = []
        for key in src_map:
            if key in tgt_map:
                s, t = src_map[key].strip(), tgt_map[key].strip()
                if s and t:
                    pairs.append(ParallelPair(
                        src=s, tgt=t, citation=".".join(key),
                        src_lang=("grc" if self.src_tag == "grc" else "la"),
                        era=era, source=source,
                    ))
        return pairs

    @staticmethod
    def _levels(sections: List[Tuple[list, str]]) -> List[str]:
        """Ordered, de-duplicated citation level names (excluding wrappers)."""
        seen: List[str] = []
        for chain, _ in sections:
            for subtype, _n in chain:
                if subtype not in _WRAPPER_TYPES and subtype not in seen:
                    seen.append(subtype)
        return seen

    @staticmethod
    def _collapse(sections: List[Tuple[list, str]], common: List[str]) -> Dict[Tuple, str]:
        """Group text by the @n values of the shared citation levels, in order."""
        out: Dict[Tuple, List[str]] = {}
        for chain, text in sections:
            levels = {subtype: n for subtype, n in chain}
            key = tuple(levels[lvl] for lvl in common if lvl in levels)
            if len(key) != len(common):
                continue  # this leaf doesn't reach the shared depth
            out.setdefault(key, []).append(text)
        return {k: " ".join(v) for k, v in out.items()}

    # -- fetch / parse -------------------------------------------------------

    def _sections(self, group: str, work: str, filename: str) -> List[Tuple[list, str]]:
        raw = (f"https://raw.githubusercontent.com/{self.repo}/master/"
               f"data/{group}/{work}/{filename}")
        resp = self.session.get(raw, timeout=self.timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        self._strip_notes(root)
        body = next((e for e in root.iter() if _local(e.tag) == "body"), None)
        if body is None:
            return []

        out: List[Tuple[list, str]] = []

        def walk(elem: ET.Element, trail: list):
            n = elem.get("n")
            subtype = (elem.get("subtype") or elem.get("type") or "").strip().lower()
            here = trail + ([(subtype, n)] if n else [])
            child_divs = [c for c in elem if _local(c.tag) == "div"]
            if child_divs:
                for c in child_divs:
                    walk(c, here)
            else:
                text = re.sub(r"\s+", " ", " ".join(elem.itertext())).strip()
                if text:
                    out.append((here, text))

        for d in [c for c in body if _local(c.tag) == "div"]:
            walk(d, [])
        return out

    @staticmethod
    def _strip_notes(root: ET.Element) -> None:
        parents = {c: p for p in root.iter() for c in p}
        for el in list(parents):
            if _local(el.tag) in ("note", "bibl", "ref") and el in parents:
                parent = parents[el]
                if el in list(parent):
                    parent.remove(el)

    def _list_dir(self, path: str) -> list:
        url = f"https://api.github.com/repos/{self.repo}/contents/{path}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _pick(files: List[str], tag: str) -> Optional[str]:
        cands = [f for f in files if re.search(rf"-{tag}\d*\.xml$", f)]
        return sorted(cands)[-1] if cands else None

    @staticmethod
    def _parse(identifier: str) -> Tuple[str, str]:
        ident = identifier.split(":")[-1].replace("/", ".")
        parts = ident.split(".")
        return parts[0], parts[1]


def pair_to_dict(p: ParallelPair) -> dict:
    return asdict(p)
