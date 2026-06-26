"""Connector for the Perseus canonical corpora on GitHub.

PerseusDL publishes its texts as TEI in GitHub repos, one work per directory:
    data/<textgroup>/<work>/<textgroup>.<work>.<edition>.xml
e.g. data/phi0448/phi001/phi0448.phi001.perseus-lat2.xml  (Caesar, De bello Gallico)

fetch() resolves an identifier to the work directory, picks the language edition,
and reuses the TEI parser. discover() lists the works under a textgroup. If the
directory also has an English (`-eng`) edition, the work is flagged as already
translated.

Identifiers accepted: a CTS urn (urn:cts:latinLit:phi0448.phi001), a dotted id
(phi0448.phi001), or a path (phi0448/phi001). For discover(), pass just the
textgroup (phi0474, or its urn).

This base class is Latin; GreekPerseusConnector subclasses it for greekLit.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import re

import requests

from .base import Connector, RawWork
from .tei import TEIConnector
from .translation_status import has_parallel_english, TRANSLATED, UNKNOWN


class PerseusConnector(Connector):
    name = "perseus"
    REPO = "PerseusDL/canonical-latinLit"
    EDITION_TAG = "lat"          # pick the *-lat*.xml edition
    DEFAULT_STAGE = "classical"
    LANGUAGE = "la"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LatinReader-Research/1.0 (scholarly; jth156@case.edu)"}
        )
        self._tei = TEIConnector(timeout=timeout)

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        group, work = self._parse_work(identifier)
        files = [f["name"] for f in self._list_dir(f"data/{group}/{work}")
                 if f.get("type") == "file"]

        edition = self._pick_edition(files)
        if edition is None:
            raise ValueError(f"No {self.EDITION_TAG} edition in {group}/{work}: {files}")
        # A sibling -eng edition is near-certain proof a translation exists; its
        # absence is only weak evidence, so we stay at "unknown" rather than
        # claiming "untranslated".
        has_eng = has_parallel_english(files)

        raw = (f"https://raw.githubusercontent.com/{self.REPO}/master/"
               f"data/{group}/{work}/{edition}")
        resp = self.session.get(raw, timeout=self.timeout)
        resp.raise_for_status()

        meta = {
            "source": f"Perseus ({group}.{work})",
            "language_stage": self.DEFAULT_STAGE,
            "language": self.LANGUAGE,
            "license": "Perseus / CC BY-SA",
            "has_existing_translation": has_eng,
            "translation_status": TRANSLATED if has_eng else UNKNOWN,
        }
        meta.update(meta_overrides)
        return self._tei.parse_xml(resp.content, label=f"{group}.{work}", **meta)

    def discover(self, textgroup: str, limit: int = 50) -> List[str]:
        group = self._parse_group(textgroup)
        works = [f["name"] for f in self._list_dir(f"data/{group}")
                 if f.get("type") == "dir"]
        return [f"{group}.{w}" for w in works][:limit]

    # -- helpers -------------------------------------------------------------

    def _pick_edition(self, files: List[str]) -> Optional[str]:
        cands = [f for f in files
                 if re.search(rf"-{self.EDITION_TAG}\d*\.xml$", f)]
        return sorted(cands)[-1] if cands else None  # highest-numbered edition

    def _list_dir(self, path: str) -> list:
        url = f"https://api.github.com/repos/{self.REPO}/contents/{path}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_work(identifier: str) -> Tuple[str, str]:
        ident = identifier.split(":")[-1]                 # strip urn prefix
        ident = ident.replace("/", ".")
        parts = ident.split(".")
        if len(parts) < 2:
            raise ValueError(f"Expected <group>.<work>, got {identifier!r}")
        return parts[0], parts[1]

    @staticmethod
    def _parse_group(identifier: str) -> str:
        ident = identifier.split(":")[-1].replace("/", ".")
        return ident.split(".")[0]


class GreekPerseusConnector(PerseusConnector):
    """Perseus greekLit — classical ancient Greek texts (see the Greek module)."""
    name = "perseus_greek"
    REPO = "PerseusDL/canonical-greekLit"
    EDITION_TAG = "grc"
    DEFAULT_STAGE = "ancient"
    LANGUAGE = "grc"


class First1KGreekConnector(PerseusConnector):
    """First1KGreek — "the first thousand years of Greek", i.e. post-classical and
    patristic/late-antique Greek (2nd-~6th c.) that Perseus's classical canon omits.
    Same TEI-on-GitHub structure, so it reuses the Perseus fetch/discover machinery.
    """
    name = "first1k_greek"
    REPO = "OpenGreekAndLatin/First1KGreek"
    EDITION_TAG = "grc"
    DEFAULT_STAGE = "late_antique"
    LANGUAGE = "grc"
