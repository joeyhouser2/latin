"""Connector for papyri.info: published Greek and Latin papyri, tablets and ostraca.

papyri.info joins the Duke Databank of Documentary Papyri (DDbDP: letters,
contracts, receipts, army records -- ~68,000 texts) and the Digital Corpus of
Literary Papyri (DCLP: copies of literature, including the Herculaneum rolls --
~15,000), with dates, places and descriptions from the Heidelberg catalogue
(HGV). Most of it is Greek; about 3% of DDbDP and 6% of DCLP is Latin (a 900-text
sample, 2026-10): the Vindolanda tablets, Chartae Latinae Antiquiores, Roman
military records, the Bu Njem ostraca, Latin letters, Ennius and Virgil on
papyrus. Much of it has never been translated into English.

The Oxyrhynchus rubbish-mound papyri are part of this, but only the published
ones: of the ~500,000 fragments in Oxford, ~5,000 are edited. The unpublished
rest exists only as images (see docs/papyri-htr-feasibility.md).

Access. The papyri.info website sits behind a proof-of-work bot check, which
this connector does not try to pass. The same data is published as the
``papyri/idp.data`` GitHub repository (CC BY 3.0), the project's intended bulk
channel: one EpiDoc XML file per text, named by Trismegistos (TM) number.
Files are mirrored on demand into ``data/raw/papyri`` and refreshed by git blob
hash, as for Capitularia.

Finding the Latin. Language is only marked inside each file
(``<div type="edition" xml:lang="la">``), so discovery needs an index built
by reading the files: ``scripts/papyri_index.py``. A full index reads all ~83,000
files (~530 MB, mirrored as it goes); ``--buckets`` builds part of it.

Rendering. Editions are in EpiDoc (Leiden conventions as XML). They are turned
back into a light Leiden text a reader can follow: ``[abc]`` restored by the
editor, ``[...]`` lost, ``(abc)`` an abbreviation resolved, ``⟦abc⟧`` deleted by
the scribe. Papyrus documents have no punctuation and split words across
lines, so lines are rejoined at word breaks and grouped into segments of about
``SEGMENT_CHARS`` -- sentence splitting would make one segment of a whole
contract. A text with fewer than ``MIN_LETTERS`` surviving letters (outside the
editor's restorations) is refused: many entries are a handful of letters.

    python scripts/papyri_index.py --buckets DDbDP/69,DDbDP/114,DCLP/66
    python scripts/ingest.py papyri latin --discover --limit 20
    python scripts/ingest.py papyri ddbdp:69917          # ChLA 10 415
    python scripts/ingest.py papyri chla;10;415          # same, by edition (needs the index)
    python scripts/ingest.py papyri dclp:59721           # Ennius, P.Herc. 21
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from lxml import etree

from .base import Connector, RawWork

REPO = "papyri/idp.data"
RAW = f"https://raw.githubusercontent.com/{REPO}/master/"
API = f"https://api.github.com/repos/{REPO}/git/trees/"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO_ROOT / "data" / "raw" / "papyri"
TEXT_CORPORA = ("DDbDP", "DCLP")
CORPORA = TEXT_CORPORA + ("HGV_meta_EpiDoc", "Translations")
TREE_MAX_AGE = 14 * 86400          # the repo syncs hourly; a fortnight-old map is fine
TEI = "{http://www.tei-c.org/ns/1.0}"
NS = {"t": "http://www.tei-c.org/ns/1.0"}
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

SEGMENT_CHARS = 200
MIN_LETTERS = 40
LICENSE = "CC BY 3.0 (papyri.info: DDbDP / DCLP / HGV; github.com/papyri/idp.data)"

LANG_ALIASES = {"latin": "la", "greek": "grc", "coptic": "cop", "demotic": "egy-Egyd"}

# HGV catalogues Greek documents, so many Latin texts have no HGV record and no
# date or place. For series that come from one place and period, fall back to
# that: (years, place). Used only when HGV says nothing; ChLA, which spans
# everything, has no fallback.
SERIES_DEFAULTS = {
    "p.ital":    ([445, 700], "Ravenna and Italy"),        # Tjäder's Italian papyri
    "t.vindol":  ([90, 120], "Vindolanda (Britannia)"),
    "o.bu.njem": ([253, 259], "Gholaia / Bu Njem (Africa proconsularis)"),
    "p.dura":    ([200, 256], "Dura-Europos"),
    "p.masada":  ([73, 74], "Masada"),
    "t.alb":     ([493, 496], "Vandal Africa (Tablettes Albertini)"),   # Gunthamund's years
}


# ---------------------------------------------------------------------------
# EpiDoc -> light Leiden text
# ---------------------------------------------------------------------------

def _local(el) -> str:
    return etree.QName(el).localname if isinstance(el.tag, str) else ""


class _Leiden:
    """Walks an edition, collecting (line_no, text, joins_previous) lines.

    Only what is on the papyrus (or restored by the editor) is emitted:
    notes, apparatus readings other than the lemma, and editorial heads are
    skipped. ``letters`` counts letters read from the papyrus itself, so a
    text that is nearly all restoration can be refused.
    """

    def __init__(self):
        self.lines: List[List[Any]] = []      # [n, text, break_no]
        self.letters = 0

    def _add(self, s: str, counted: bool) -> None:
        if not s:
            return
        if not self.lines:
            self.lines.append(["", "", False])
        self.lines[-1][1] += s
        if counted:
            self.letters += sum(c.isalpha() for c in s)

    def walk(self, el, counted: bool = True) -> None:
        """Emit one element (its tail is the caller's business)."""
        tag = _local(el)
        if tag == "lb":
            self.lines.append([el.get("n", ""), "", el.get("break") == "no"])
        elif tag == "gap":
            self._add("[---]" if el.get("unit") == "line" else
                      "[...]" if el.get("reason") == "lost" else "...", False)
        elif tag == "supplied":
            o, c = ("<", ">") if el.get("reason") == "omitted" else ("[", "]")
            self._add(o, False)
            self._children(el, False)
            self._add(c, False)
        elif tag == "del":
            self._add("⟦", False)
            self._children(el, counted)
            self._add("⟧", False)
        elif tag == "ex":
            self._add("(", False)
            self._children(el, False)
            self._add(")", False)
        elif tag == "g":
            self._add("·" if el.get("type") in ("middot", "dot", "interpunct") else "", False)
        elif tag == "space":
            self._add(" ", False)
        elif tag in ("choice", "app", "subst"):
            # One reading of several: regularised spelling over the original,
            # correction over error, expansion over abbreviation, the lemma
            # over variants, the scribe's addition over what it replaced.
            for pref in ("reg", "corr", "expan", "lem", "add"):
                pick = el.find(f"t:{pref}", NS)
                if pick is not None:
                    self.walk(pick, counted)
                    break
            else:
                if len(el):
                    self.walk(el[0], counted)
        elif tag in ("note", "head", "certainty", "handShift", "milestone", "rdg", "figure"):
            pass
        else:
            self._children(el, counted)

    def _children(self, el, counted: bool) -> None:
        if el.text:
            self._add(el.text, counted)
        for child in el:
            self.walk(child, counted)
            if child.tail:
                self._add(child.tail, counted)


def _render(container) -> Tuple[str, int]:
    """One textpart (or a whole edition without textparts) -> (text, letters).

    Lines are rejoined where a word runs over the line end, then grouped into
    segments of about SEGMENT_CHARS, one per output line.
    """
    w = _Leiden()
    w._children(container, True)
    words: List[str] = []
    for _, text, break_no in w.lines:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if break_no and words:
            words[-1] += text            # the word began on the previous line
        else:
            words.append(text)
    joined = " ".join(words)
    joined = re.sub(r"\]\[|\)\(", "", joined)               # [ab][cd] -> [abcd]
    joined = re.sub(r"(?:\[(?:---|\.\.\.)\]\s*){2,}", "[...] ", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    # Group into ~SEGMENT_CHARS chunks at word boundaries.
    segs, cur = [], ""
    for word in joined.split(" "):
        if cur and len(cur) + 1 + len(word) > SEGMENT_CHARS:
            segs.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        segs.append(cur)
    return "\n".join(segs), w.letters


def edition_language(root) -> Optional[str]:
    ed = root.find(".//t:div[@type='edition']", NS)
    return ed.get(XML_LANG) if ed is not None else None


def parse_text(xml: bytes) -> Dict[str, Any]:
    """A DDbDP/DCLP file -> language, edition labels, header metadata, parts."""
    root = etree.fromstring(xml)
    ed = root.find(".//t:div[@type='edition']", NS)
    parts: List[Tuple[str, str]] = []
    letters = 0
    if ed is not None:
        textparts = [d for d in ed.iter(f"{TEI}div")
                     if d.get("type") == "textpart" and not d.findall(f"{TEI}div[@type='textpart']")]
        for i, part in enumerate(textparts or [ed]):
            text, n = _render(part)
            letters += n
            if text:
                parts.append((_part_label(part, i, len(textparts)), text))
    heads = [t.text for t in root.findall(".//t:body/t:head//t:title", NS) if t.text]
    return {"lang": ed.get(XML_LANG) if ed is not None else None,
            "editions": heads, "parts": parts, "letters": letters,
            "header": header_meta(root)}


def _part_label(part, i: int, n: int) -> str:
    if n == 0:
        return "Text"
    head = part.find("t:head", NS)
    if head is not None and (head.text or "").strip():
        return head.text.strip()
    sub, num = part.get("subtype") or "part", part.get("n") or str(i + 1)
    return {"recto": "recto", "verso": "verso"}.get(num, f"{sub} {num}")


def header_meta(root) -> Dict[str, Any]:
    """Title, date, place, material, inventory and TM/hybrid ids from a teiHeader
    (HGV metadata files and DCLP texts share this shape)."""
    def first(xpath: str) -> Optional[str]:
        el = root.find(xpath, NS)
        return " ".join("".join(el.itertext()).split()) if el is not None else None

    def idno(kind: str) -> Optional[str]:
        el = root.find(f".//t:publicationStmt/t:idno[@type='{kind}']", NS)
        return el.text.strip() if el is not None and el.text else None

    date = root.find(".//t:origin/t:origDate", NS)
    years = []
    if date is not None:
        for attr in ("when", "notBefore", "notAfter"):
            m = re.match(r"(-?\d{1,4})", date.get(attr) or "")
            if m:
                years.append(int(m.group(1)))
    return {
        "title": first(".//t:titleStmt/t:title"),
        "date_text": " ".join("".join(date.itertext()).split()) if date is not None else None,
        "years": years,
        "place": first(".//t:origin/t:origPlace"),
        "material": first(".//t:support/t:material"),
        "inventory": first(".//t:msIdentifier//t:idno[@type='invNo']") or first(".//t:msIdentifier/t:idno"),
        "keywords": [" ".join(t.itertext()).strip()
                     for t in root.findall(".//t:textClass/t:keywords/t:term", NS)][:12],
        "tm": idno("TM"),
        "hybrid": idno("ddb-hybrid") or idno("dclp-hybrid"),
    }


def century_of(years: List[int]) -> Optional[int]:
    """Midpoint year -> century, negative for BCE (core.models convention)."""
    if not years:
        return None
    y = round(sum(years) / len(years))
    if y == 0:
        y = 1
    return (y - 1) // 100 + 1 if y > 0 else -((-y - 1) // 100 + 1)


def stage_of(years: List[int]) -> str:
    if not years:
        return "ancient"
    y = sum(years) / len(years)
    return "classical" if y < 250 else "late_antique" if y < 700 else "medieval"


# ---------------------------------------------------------------------------
# the connector
# ---------------------------------------------------------------------------

class PapyriConnector(Connector):
    name = "papyri"

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE, timeout: float = 60.0,
                 workers: int = 8, min_letters: int = MIN_LETTERS):
        self.cache = Path(cache_dir)
        self.timeout = timeout
        self.workers = workers
        self.min_letters = min_letters
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "latin-library/1.0 (research)"})
        # corpus -> {path: blob sha}; "<corpus>#by-tm" -> {TM: path}
        self._trees: Dict[str, Dict[str, str]] = {}

    # -- the repository's file map --------------------------------------------

    def tree(self, corpus: str) -> Dict[str, str]:
        """corpus-relative path -> blob sha, cached for TREE_MAX_AGE.

        Two GitHub API calls per corpus (root, then the corpus subtree) -- the
        unauthenticated limit is 60 an hour, so the map is kept on disk."""
        if corpus in self._trees:
            return self._trees[corpus]
        path = self.cache / "_trees" / f"{corpus}.json"
        if not path.exists() or time.time() - path.stat().st_mtime > TREE_MAX_AGE:
            root = self._api(API + "master")
            sha = next(t["sha"] for t in root["tree"] if t["path"] == corpus)
            sub = self._api(API + sha + "?recursive=1")
            if sub.get("truncated"):
                raise RuntimeError(f"GitHub truncated the {corpus} tree listing")
            blobs = {t["path"]: t["sha"] for t in sub["tree"] if t["type"] == "blob"}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(blobs))
        self._trees[corpus] = json.loads(path.read_text())
        return self._trees[corpus]

    def _api(self, url: str) -> Dict[str, Any]:
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code == 403 and "rate limit" in r.text.lower():
            raise RuntimeError("GitHub API rate limit reached (60/hour unauthenticated); "
                               "try again later -- the file maps are cached once fetched")
        r.raise_for_status()
        return r.json()

    def path_for(self, corpus: str, tm: str) -> Optional[str]:
        """corpus/bucket/file.xml for a TM number. Buckets are not one formula
        (DCLP files sit in TM//1000 or TM//1000 + 1), so look it up."""
        if f"{corpus}#by-tm" not in self._trees:
            self._trees[f"{corpus}#by-tm"] = {rel.rsplit("/", 1)[-1][:-4]: rel
                                              for rel in self.tree(corpus)}
        rel = self._trees[f"{corpus}#by-tm"].get(str(tm))
        return f"{corpus}/{rel}" if rel else None

    def read(self, path: str) -> bytes:
        """A repository file, from the mirror if its blob hash is current."""
        corpus, rel = path.split("/", 1)
        sha = self.tree(corpus).get(rel)
        dest = self.cache / path
        manifest = dest.with_suffix(".sha")
        if dest.exists() and manifest.exists() and manifest.read_text() == sha:
            return dest.read_bytes()
        # raw.githubusercontent.com throttles a long run (a full index is
        # ~83,000 requests): back off and retry rather than lose the file.
        for attempt in range(6):
            try:
                r = self.session.get(RAW + path, timeout=self.timeout)
                if r.status_code not in (429, 500, 502, 503, 504):
                    break
            except requests.ConnectionError:
                if attempt == 5:
                    raise
            time.sleep(min(60, 2 ** attempt * 2))
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        if sha:
            manifest.write_text(sha)
        return r.content

    def read_many(self, paths: Iterable[str]) -> Iterable[Tuple[str, Optional[bytes]]]:
        def one(p: str):
            try:
                return p, self.read(p)
            except requests.RequestException:
                return p, None
        with ThreadPoolExecutor(self.workers) as pool:
            yield from pool.map(one, paths)

    # -- index (scripts/papyri_index.py) --------------------------------------

    @property
    def index_path(self) -> Path:
        return self.cache / "index.json"

    def load_index(self) -> Dict[str, Dict[str, Any]]:
        return json.loads(self.index_path.read_text()) if self.index_path.exists() else {}

    def build_index(self, corpora: Iterable[str] = TEXT_CORPORA,
                    buckets: Optional[Iterable[str]] = None,
                    progress=print) -> Dict[str, Dict[str, Any]]:
        """Read text files and record each one's language and readable size.

        ``buckets`` like ``DDbDP/69`` limits the scan; entries already indexed
        at the current blob hash are not re-read, so a full build can be
        resumed and later refreshed cheaply."""
        index = self.load_index()
        wanted = set(buckets or [])
        todo = []
        for corpus in corpora:
            for rel, sha in self.tree(corpus).items():
                path = f"{corpus}/{rel}"
                if wanted and f"{corpus}/{rel.split('/')[0]}" not in wanted:
                    continue
                if index.get(path, {}).get("sha") != sha:
                    todo.append((path, sha))
        progress(f"papyri index: {len(todo):,} files to read ({len(index):,} already indexed)")
        done = failed = 0
        for path, xml in self.read_many(p for p, _ in todo):
            done += 1
            if done % 1000 == 0:
                progress(f"  overall {done:,}/{len(todo):,}"
                         + (f" ({failed:,} failed)" if failed else ""))
                self._save_index(index)
            if xml is None:
                failed += 1
                continue
            try:
                info = parse_text(xml)
            except etree.XMLSyntaxError:
                continue
            index[path] = {"sha": self.tree(path.split("/")[0]).get(path.split("/", 1)[1]),
                           "tm": path.rsplit("/", 1)[-1][:-4], "lang": info["lang"],
                           "letters": info["letters"], "hybrid": info["header"]["hybrid"],
                           "edition": (info["editions"] or [info["header"]["title"]])[0]}
        self._save_index(index)
        if failed:
            progress(f"papyri index: {failed:,} files could not be fetched -- run again "
                     f"to retry them (indexed files are skipped)")
        return index

    def _save_index(self, index: Dict[str, Any]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False))
        tmp.replace(self.index_path)

    # -- Connector API ---------------------------------------------------------

    def _matches(self, query: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Indexed texts for ``latin`` / ``greek`` / a language code, optionally
        ``:DDbDP`` or ``:DCLP``, with at least ``min_letters`` readable letters,
        longest first -- the three-letter fragments come last, if at all."""
        index = self.load_index()
        if not index:
            raise RuntimeError("no papyri index yet: run scripts/papyri_index.py "
                               "(--buckets for a partial one)")
        lang, _, corpus = (query or "latin").partition(":")
        lang = LANG_ALIASES.get(lang.lower().strip(), lang.strip())
        hits = [(p, e) for p, e in index.items()
                if (e.get("lang") or "").split("-")[0] == lang
                and (not corpus or p.split("/")[0].lower() == corpus.lower())
                and e.get("letters", 0) >= self.min_letters]
        hits.sort(key=lambda pe: -pe[1]["letters"])
        return hits

    def discover(self, query: str, limit: int = 50) -> List[str]:
        """``ddbdp:TM`` / ``dclp:TM`` identifiers; see _matches for the query."""
        return [f"{p.split('/')[0].lower()}:{e['tm']}" for p, e in self._matches(query)[:limit]]

    def catalog(self, query: str = "latin", limit: int = 400) -> List[Dict[str, Any]]:
        """Records for the Find-texts tab, straight from the index (no fetching)."""
        out = []
        for p, e in self._matches(query)[:limit]:
            corpus = p.split("/")[0]
            hybrid = e.get("hybrid")
            out.append({
                "catalogue": "papyri", "identifier": f"{corpus.lower()}:{e['tm']}",
                "title": e.get("edition") or hybrid or f"TM {e['tm']}",
                "author": None, "year": None, "date": None,
                "url": (f"https://papyri.info/{corpus.lower()}/{hybrid}" if hybrid
                        else f"https://www.trismegistos.org/text/{e['tm']}"),
                "fetchable": True,
                "note": f"{corpus}, TM {e['tm']}; {e['letters']:,} letters survive",
                "genre": "literary papyrus" if corpus == "DCLP" else "documentary papyrus",
                "lang_code": (e.get("lang") or "").split("-")[0],
            })
        return out

    def resolve(self, identifier: str) -> str:
        """ddbdp:TM, dclp:TM, a bare TM number, or an edition id like chla;10;415."""
        ident = identifier.strip()
        ident = re.sub(r"^https?://papyri\.info/(ddbdp|dclp)/", "", ident)
        m = re.fullmatch(r"(ddbdp|dclp):(\w+)", ident, re.I)
        if m:
            corpus = "DDbDP" if m.group(1).lower() == "ddbdp" else "DCLP"
            path = self.path_for(corpus, m.group(2))
            if path:
                return path
            raise ValueError(f"no {corpus} text with TM {m.group(2)}")
        if re.fullmatch(r"\d+[a-z]*", ident):
            for corpus in TEXT_CORPORA:
                path = self.path_for(corpus, ident)
                if path:
                    return path
            raise ValueError(f"no papyri.info text with TM {ident}")
        for path, e in self.load_index().items():          # edition id, via the index
            if e.get("hybrid") == ident or e.get("edition") == ident:
                return path
        raise ValueError(f"cannot resolve {identifier!r}: use ddbdp:TM, dclp:TM, or "
                         f"build the index to look up edition ids")

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        path = self.resolve(identifier)
        corpus, tm = path.split("/")[0], path.rsplit("/", 1)[-1][:-4]
        info = parse_text(self.read(path))
        if info["letters"] < self.min_letters:
            raise ValueError(f"{path}: only {info['letters']} letters survive outside the "
                             f"editor's restorations (minimum {self.min_letters})")
        head = info["header"]
        if corpus == "DDbDP":                     # dates and places live in HGV
            hgv = self.path_for("HGV_meta_EpiDoc", tm)
            if hgv:
                head = header_meta(etree.fromstring(self.read(hgv)))
        series = (head.get("hybrid") or "").split(";")[0]
        if not head["years"] and series in SERIES_DEFAULTS:
            years, place = SERIES_DEFAULTS[series]
            head = {**head, "years": years, "place": head.get("place") or place,
                    "date_text": head.get("date_text") or f"{years[0]}-{years[1]} (series)"}
        edition = info["editions"][0] if info["editions"] else (head.get("hybrid") or f"TM {tm}")
        # HGV titles are descriptive ("Édit sur les délais des appels ..."); it
        # writes "keiner" (none) when there is nothing to say, and a text with
        # no HGV record is titled by its file name ("chla.18.659").
        title = head.get("title") or ""
        subject = (None if title in ("", "keiner", edition, f"TM {tm}")
                   or re.fullmatch(r"[a-z][a-z.]*\.[\w.]+", title) else title)
        lang = (info["lang"] or "grc").split("-")[0]
        translation = self.path_for("Translations", tm)

        meta = {
            "title": f"{edition} — {subject}" if subject else edition,
            "century": century_of(head["years"]),
            "genre": ("literary papyrus" if corpus == "DCLP" else "documentary papyrus")
                     + (f" ({head['material'].lower()})" if head.get("material")
                        and head["material"].lower() != "papyrus" else ""),
            "language": lang if lang in ("la", "grc") else lang,
            "language_stage": stage_of(head["years"]),
            "source": "; ".join(b for b in (
                f"papyri.info {corpus} TM {tm}"
                + (f" ({head['hybrid']})" if head.get("hybrid") else ""),
                head.get("place"), head.get("date_text")) if b),
            "shelfmark": head.get("inventory"),
            "license": LICENSE,
            "has_existing_translation": bool(translation),
            "translation_status": "translated" if translation else "unknown",
            "translation_evidence": (f"papyri.info translation file {translation}"
                                     if translation else None),
            "_verse": True,       # one segment per line: _render already grouped the lines
        }
        meta.update(meta_overrides)
        return meta, info["parts"]
