"""Connector for famous vernacular works that English readers mostly can't get.

The library's spine is Latin and Greek, but a great deal of medieval and
Renaissance Europe wrote in its own languages -- and the canon of each of those
literatures (the works every German, Pole or Hungarian schoolchild is taught)
is often unavailable in English, or only in a century-old abridgement. This
connector brings them in through a *curated catalogue*: a short list of works
chosen because they are famous in their own language, each pinned to the
Wikisource edition that carries its text.

Seven languages: German (``de``), French (``fr``), Italian (``it``), Dutch
(``nl``), Polish (``pl``), Hungarian (``hu``), Russian (``ru``). Documents get
that language code, so the existing pipeline picks it up unchanged: language-aware
segmentation (``core.segmenter``), stock NLLB routing (``pipeline.STOCK_SRC``),
and the reader/library filters.

Honest limits, so nobody is misled by fluent output:

* NLLB was trained on modern text. Middle High German, Middle Dutch, Old
  Hungarian and Old Russian are well outside that; expect a rough gloss, not a
  translation, for the oldest works. Each catalogue entry says how old its
  language stage is (``language_stage``) and, where the edition is a modernised
  one, says so in ``note``.
* ``translation_status`` is left ``unknown``. Whether a famous work is
  untranslated is a claim worth checking (``scripts/enrich_translation_status.py``
  does that against Wikidata and the catalogues), not one a hand-written list
  should assert.

Usage:
    from ingest.vernacular import VernacularConnector
    v = VernacularConnector()
    v.catalog("pl")                       # entries for one language ("all" for every one)
    meta, parts = v.fetch("pl:rej-zywot") # a catalogue key...
    meta, parts = v.fetch("ws:ru:Домострой (Орлов)")   # ...or any Wikisource page

    python scripts/ingest.py vernacular pl:rej-zywot
    python scripts/ingest.py vernacular pl --discover --limit 10
    python scripts/ingest.py vernacular all --discover       # whole catalogue

Pages are cached under ``data/raw/vernacular``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

from core.models import LANGUAGES
from ._html import extract_paragraphs
from .base import Connector, RawWork

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO_ROOT / "data" / "raw" / "vernacular"
_HEADERS = {"User-Agent": "LatinReader-Research/1.0 (scholarly research)"}
_LINE_BREAK = ""   # <br> survives the paragraph extractor's whitespace collapse as this
_SKIP_CLASSES = {
    "mw-editsection", "reference", "noprint", "mw-cite-backlink", "navigation-not-searchable",
    "ws-noexport", "mw-references-wrap", "ws-header", "ws-footer", "prp-pages-output",
    "pagenum", "ws-pagenum", "toc", "mw-headline-anchor", "interwiki-extra",
    "interlanguage-link", "metadata",
}
_BRACKET_NUM = re.compile(r"\[\s*\d+\s*\]")


@dataclass
class Work:
    """One catalogue entry. ``page`` is the Wikisource title; ``expand`` also
    pulls every ``page/...`` subpage, in order, as separate sections."""

    key: str
    lang: str
    title: str
    author: Optional[str]
    century: Optional[int]
    stage: str
    genre: str
    page: str
    expand: bool = False
    verse: bool = False
    note: str = ""


# Keys are "<lang>:<slug>". Add a work by pinning it to a Wikisource page; the
# `check` helper (VernacularConnector.check) reports whether each page resolves
# and how much text it yields, so an entry that points at a stub shows up at once.
CATALOG: List[Work] = [
    # --- Polish ---------------------------------------------------------------
    Work("pl:bogurodzica", "pl", "Bogurodzica", None, 13, "medieval", "hymn",
         "Bogurodzica (pieśń)", verse=True,
         note="Oldest known Polish poem, a Marian hymn also sung as a battle song at Grunwald."),
    Work("pl:kazania-swietokrzyskie", "pl", "Kazania świętokrzyskie", None, 14, "medieval",
         "sermon", "Kazania świętokrzyskie", expand=True,
         note="Oldest surviving Polish prose, a sermon collection in a 14th-c. manuscript."),
    Work("pl:rej-zywot", "pl", "Żywot człowieka poczciwego", "Mikołaj Rej", 16, "early_modern",
         "moral prose", "Żywot człowieka poczciwego (Rej, 1881)", expand=True,
         note="The Polish Renaissance's household-and-conduct classic, by the 'father of "
              "Polish literature'. 1881 edition."),
    Work("pl:odprawa", "pl", "Odprawa posłów greckich", "Jan Kochanowski", 16, "early_modern",
         "drama", "Odprawa posłów greckich (Kochanowski, 1882)", verse=True,
         note="First Renaissance tragedy in Polish; the Trojan embassy before the war."),
    # --- Hungarian ------------------------------------------------------------
    Work("hu:halotti-beszed", "hu", "Halotti beszéd és könyörgés", None, 12, "medieval",
         "funeral oration", "Halotti beszéd és könyörgés",
         note="Oldest continuous Hungarian text (c. 1192-95). The Wikisource page may carry "
              "the modernised rendering alongside the Old Hungarian."),
    # --- Russian --------------------------------------------------------------
    Work("ru:slovo", "ru", "Слово о полку Игореве", None, 12, "medieval", "epic",
         "Слово о полку Игореве/Текст",
         note="The Tale of Igor's Campaign. Wikisource's main page is the Old East Slavic text."),
    Work("ru:domostroy", "ru", "Домострой", "Сильвестр", 16, "early_modern", "household prose",
         "Домострой (Орлов)", note="The Orlov-redaction of the Domostroy."),
    # --- Italian --------------------------------------------------------------
    Work("it:novellino", "it", "Il Novellino", None, 13, "medieval", "tales", "Novellino",
         expand=True,
         note="Le ciento novelle antike: first collection of Italian prose tales."),
    # --- Dutch ----------------------------------------------------------------
    Work("nl:karel-elegast", "nl", "Karel ende Elegast", None, 13, "medieval", "romance",
         "Karel ende Elegast", expand=True, verse=True,
         note="Middle Dutch romance of Charlemagne the night-thief."),
    Work("nl:esmoreit", "nl", "Esmoreit", None, 14, "medieval", "drama", "Esmoreit", verse=True,
         note="One of the four secular 'abele spelen' of the Hulthem manuscript."),
    Work("nl:gloriant", "nl", "Gloriant", None, 14, "medieval", "drama", "Gloriant", verse=True,
         note="Another of the abele spelen."),
    # --- German ---------------------------------------------------------------
    Work("de:ackermann", "de", "Der Ackermann aus Böhmen", "Johannes von Tepl", 14,
         "medieval", "dialogue", "Der Ackermann aus Böhmen (Handschrift 14. Jh.)", expand=True,
         note="A ploughman accuses Death for taking his wife; the first great work of "
              "early New High German prose."),
    # --- more Russian -----------------------------------------------------------
    Work("ru:zadonshchina", "ru", "Задонщина", None, 15, "medieval", "epic",
         "Задонщина/1858 (ДО)", expand=True,
         note="Tale of the Kulikovo victory (1380), modelled on the Slovo; 1858 edition, "
              "pre-reform orthography."),
    Work("ru:nikitin", "ru", "Хождение за три моря", "Афанасий Никитин", 15, "early_modern",
         "travel", "Хождение за три моря Афанасия Никитина",
         note="A Tver merchant's journey to India (1466-72), decades before da Gama."),
    Work("ru:petr-fevronia", "ru", "Повесть о Петре и Февронии Муромских", "Ермолай-Еразм", 16,
         "early_modern", "hagiography", "Повесть о Петре и Февронии Муромских (Еразм)"),
    # --- more Italian -----------------------------------------------------------
    Work("it:trecentonovelle", "it", "Il Trecentonovelle", "Franco Sacchetti", 14, "medieval",
         "tales", "Il Trecentonovelle", expand=True,
         note="Florentine tales of the 1390s, a rival to the Decameron, mostly unavailable "
              "in English."),
    Work("it:fiore", "it", "Il Fiore", "Ser Durante (attrib. Dante)", 13, "medieval",
         "verse romance", "Fiore", expand=True, verse=True,
         note="232 sonnets condensing the Roman de la Rose; attributed to the young Dante."),
    Work("it:morgante", "it", "Morgante maggiore", "Luigi Pulci", 15, "early_modern",
         "epic", "Morgante maggiore", expand=True, verse=True,
         note="Comic chivalric epic of the giant Morgante, Orlando's convert."),
    Work("it:orlando-innamorato", "it", "Orlando innamorato", "Matteo Maria Boiardo", 15,
         "early_modern", "epic", "Orlando innamorato", expand=True, verse=True),
    # --- more French ------------------------------------------------------------
    Work("fr:renart", "fr", "Le Roman de Renart", None, 12, "medieval", "beast epic",
         "Le Roman de Renart", expand=True,
         note="Wikisource's edition is a modern-French rendering of the branches."),
    # --- more Dutch -------------------------------------------------------------
    Work("nl:reynaert", "nl", "Van den vos Reynaerde", None, 13, "medieval", "beast epic",
         "Vanden Vos Reynaerde", expand=True, verse=True),
    Work("nl:elckerlijc", "nl", "Den Spyeghel der Salicheyt van Elckerlijc", None, 15, "medieval",
         "morality play", "Den Spyeghel der Salicheyt van Elckerlijc", verse=True,
         note="Source of the English Everyman."),
    Work("nl:brandaan", "nl", "Reis van Sint-Brandaan", None, 12, "medieval", "voyage",
         "Reis van Sint-Brandaan", expand=True, verse=True),
    # --- more German ------------------------------------------------------------
]

_BY_KEY: Dict[str, Work] = {w.key: w for w in CATALOG}


class VernacularConnector(Connector):
    name = "vernacular"

    def __init__(self, cache_dir: Optional[str] = None, timeout: float = 30.0,
                 delay: float = 1.5):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
        self.timeout = timeout
        self.delay = delay       # Wikisource rate-limits bursts with an HTML error page
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)

    # -- catalogue ---------------------------------------------------------------

    @staticmethod
    def catalog(lang: str = "all") -> List[Work]:
        return [w for w in CATALOG if lang in ("all", w.lang)]

    def discover(self, query: str, limit: int = 50) -> List[str]:
        """``query`` is a language code or "all"; returns catalogue keys."""
        if query != "all" and query not in LANGUAGES:
            raise ValueError(f"unknown language {query!r}; use one of "
                             f"{', '.join(sorted(set(w.lang for w in CATALOG)))} or 'all'")
        return [w.key for w in self.catalog(query)][:limit]

    # -- fetch -------------------------------------------------------------------

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        if identifier.startswith("ws:"):
            _, lang, page = identifier.split(":", 2)
            work = Work(key=identifier, lang=lang, title=page, author=None, century=None,
                        stage="unknown", genre="", page=page)
        elif identifier in _BY_KEY:
            work = _BY_KEY[identifier]
        else:
            raise KeyError(f"Unknown work {identifier!r}. Catalogue keys: "
                           f"{', '.join(w.key for w in CATALOG)}; or 'ws:<lang>:<page>'.")
        if work.lang not in LANGUAGES:
            raise ValueError(f"unsupported language {work.lang!r}")

        titles = [work.page]
        if work.expand:
            titles += self._subpages(work.lang, work.page)
        parts = []
        for t in titles:
            text = self._page_text(work.lang, t, verse=work.verse)
            if text:
                label = t[len(work.page) + 1:] if t.startswith(work.page + "/") else "Text"
                parts.append((label, text))
        if not parts:
            raise ValueError(f"no text extracted from {work.lang}.wikisource.org: {work.page!r}")

        meta = {
            "title": work.title,
            "author": work.author,
            "century": work.century,
            "genre": work.genre or None,
            "language": work.lang,
            "language_stage": work.stage,
            "source": f"{LANGUAGES[work.lang]} Wikisource ({work.page})",
            "license": "CC BY-SA (Wikisource)",
            "has_existing_translation": False,
            "translation_status": "unknown",
            "translation_evidence": work.note or None,
            "_verse": work.verse,
        }
        meta.update({k: v for k, v in meta_overrides.items() if v is not None})
        return meta, parts

    def check(self, lang: str = "all") -> List[dict]:
        """Resolve every catalogue entry and report size -- catches dead page
        titles and stub pages before a bulk ingest does."""
        out = []
        for w in self.catalog(lang):
            try:
                _, parts = self.fetch(w.key)
                out.append({"key": w.key, "sections": len(parts),
                            "chars": sum(len(t) for _, t in parts)})
            except Exception as exc:
                out.append({"key": w.key, "error": str(exc)})
        return out

    # -- internals ---------------------------------------------------------------

    def _api(self, lang: str, **params) -> dict:
        params["format"] = "json"
        url = f"https://{lang}.wikisource.org/w/api.php"
        for attempt in range(5):
            time.sleep(self.delay)
            resp = self.session.get(url, params=params, timeout=self.timeout)
            try:
                return resp.json()
            except ValueError:               # rate-limit page, not JSON
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"{lang}.wikisource.org kept rate-limiting {params.get('page') or params}")

    def _subpages(self, lang: str, page: str) -> List[str]:
        titles: List[str] = []
        params = dict(action="query", list="allpages", apprefix=page + "/",
                      aplimit="500", apnamespace="0")
        while True:
            data = self._api(lang, **params)
            titles += [p["title"] for p in data.get("query", {}).get("allpages", [])]
            cont = data.get("continue")
            if not cont:
                break
            params.update(cont)
        # Natural order: "Deel 2" before "Deel 10", chapters in reading order.
        return sorted(titles, key=lambda t: [int(c) if c.isdigit() else c
                                             for c in re.split(r"(\d+)", t)])

    def _page_text(self, lang: str, title: str, verse: bool) -> str:
        cache = self.cache_dir / lang / (re.sub(r'[\\/:*?"<>|]', "_", title) + ".html")
        if cache.exists():
            html = cache.read_text(encoding="utf-8")
        else:
            data = self._api(lang, action="parse", page=title, prop="text",
                             formatversion="2", redirects="1")
            if "parse" not in data:
                return ""
            html = data["parse"]["text"]
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(html, encoding="utf-8")
        if verse:
            html = re.sub(r"<br\s*/?>", _LINE_BREAK, html, flags=re.I)
        lines = []
        for p in extract_paragraphs(html, skip_classes=_SKIP_CLASSES):
            p = _BRACKET_NUM.sub("", p).strip()
            if verse:
                lines += [l.strip() for l in p.split(_LINE_BREAK) if l.strip()]
            elif sum(c.isalpha() for c in p) >= 15:
                lines.append(p)
        return "\n".join(lines)
