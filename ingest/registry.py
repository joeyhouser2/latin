"""Connector registry: look up a connector by its short name.

Keeps the CLI and any UI decoupled from concrete connector classes. Add a new
source by writing a Connector subclass and registering it here.
"""

from __future__ import annotations

from typing import Dict, Callable, List

from .base import Connector
from .latin_library import LatinLibraryConnector
from .wikisource import WikisourceConnector
from .tei import TEIConnector
from .files import FileConnector
from .digiliblt import DigilibLTConnector
from .corpus_corporum import CorpusCorporumConnector, ALIMConnector
from .edcs import EDCSConnector
from .corpus_thomisticum import CorpusThomisticumConnector
from .perseus import (PerseusConnector, GreekPerseusConnector,
                      First1KGreekConnector, PTAConnector)
from .mgh import MGHConnector
from .pg_corpus import PGCorpusConnector
from .musamedievalis import MusaMedievalisConnector


_REGISTRY: Dict[str, Callable[[], Connector]] = {
    LatinLibraryConnector.name: LatinLibraryConnector,
    WikisourceConnector.name: WikisourceConnector,
    TEIConnector.name: TEIConnector,
    FileConnector.name: FileConnector,
    DigilibLTConnector.name: DigilibLTConnector,
    CorpusCorporumConnector.name: CorpusCorporumConnector,
    ALIMConnector.name: ALIMConnector,
    EDCSConnector.name: EDCSConnector,
    CorpusThomisticumConnector.name: CorpusThomisticumConnector,
    PerseusConnector.name: PerseusConnector,
    GreekPerseusConnector.name: GreekPerseusConnector,
    First1KGreekConnector.name: First1KGreekConnector,
    PTAConnector.name: PTAConnector,
    MGHConnector.name: MGHConnector,
    PGCorpusConnector.name: PGCorpusConnector,
    MusaMedievalisConnector.name: MusaMedievalisConnector,
}


def get_connector(name: str) -> Connector:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown source {name!r}. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name]()


def available_sources() -> List[str]:
    return sorted(_REGISTRY)
