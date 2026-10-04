"""Connector for Capitularia (Cologne): the Frankish royal capitularies, 507-9th c.

The capitularies are the legislation and administrative orders of the
Merovingian and Carolingian kings -- Clovis's letter to the bishops (507),
Charlemagne's *Admonitio generalis* and *De villis*, the programmes of Louis
the Pious -- and apart from a handful of famous pieces they have never been
translated into English. The project (Karl Ubl, Universität zu Köln; funded
by the Nordrhein-Westfalen academy) publishes its data as TEI on GitHub,
``cceh/capitularia``, in two layers this connector joins:

* ``capit/{pre814,ldf,post840,undated}/*.xml`` -- one catalogue record per
  capitulary, numbered by the Boretius-Krause edition (``bk-nr-139``) or by
  Mordek for pieces found since (``mordek-nr-12``). The folders are reigns:
  up to 814, *Ludwig der Fromme* 814-840, after 840. Each record carries a
  ``<listBibl type="translation">`` -- the editors' own list of every published
  translation of that capitulary, in any language.
* ``mss/*.xml`` -- diplomatic transcriptions of the manuscripts. A capitulary
  starts at ``<milestone unit="capitulare" n="BK.221"/>`` and each chapter is
  an ``<ab corresp="BK.221_6">``. One "manuscript", ``bk-textzeuge``, is not
  a manuscript at all but the Boretius-Krause critical edition (MGH, 1883-97)
  transcribed as a pseudo-witness -- edited, normalised text, and public
  domain. It covers 237 of the 257 capitularies that have any transcription.
  So a document here is the capitulary **in the edition text** whenever that
  covers at least 80% of the chapters of the best manuscript (211 of them),
  and otherwise **in its most complete manuscript witness**, which then
  becomes the shelfmark.

Transcriptions are diplomatic, so the text is normalised on the way in: the
scribe's deletions (``<del>``) are dropped and corrections (``<add>``) kept,
``<choice>`` takes the corrected/expanded reading, and the editors' German
notes (``<note>``) are removed -- otherwise "Am linken Seitenrand steht eine
Notiz des Kopisten" would be sent to the Latin translator as Latin.

Translation status is **verified, not guessed**: each listed translation is
resolved against the project bibliography (``bibl/Bibliographie_Capitularia.xml``)
and classified by language from the entry's own note ("enthält dt.
Übersetzungen..."), its title, and its place of publication. An English one
makes the work ``translated`` if it was published before 1929 (public domain,
and in practice online -- Munro's 1900 *Laws of Charles the Great* is), else
``translated_paywalled``. The bibliography's own ``access`` field cannot decide
this: it reads "frei" on all 2,031 entries, King's 1987 print collection
included. Only non-English translations, or none listed at all, makes a work
``untranslated``; a translation whose language cannot be determined keeps it
``unknown`` -- notably *Domínguez 2014*, cited by 68 capitularies but absent
from the project bibliography and unresolvable on its website. The evidence
travels with the document (``translation_evidence``) so every claim can be
checked.

    python scripts/ingest.py capitularia BK.1
    python scripts/ingest.py capitularia pre814 --discover --limit 20
    python scripts/ingest.py capitularia untranslated --discover --limit 400

Data is cached under ``data/raw/capitularia`` (~30MB of XML) and refreshed
by git blob hash, so a re-run downloads only what changed upstream.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import requests

from .base import Connector, RawWork
from .translation_status import (TRANSLATED, TRANSLATED_PAYWALLED, UNKNOWN,
                                 UNTRANSLATED)

REPO = "cceh/capitularia"
RAW = f"https://raw.githubusercontent.com/{REPO}/master/"
TREE = f"https://api.github.com/repos/{REPO}/git/trees/master?recursive=1"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO_ROOT / "data" / "raw" / "capitularia"
TEI = "{http://www.tei-c.org/ns/1.0}"

FOLDER_PERIOD = {"pre814": "to 814", "ldf": "Louis the Pious, 814-840",
                 "post840": "after 840", "undated": "undated"}
LICENSE = ("Free, non-commercial (Capitularia, Universität zu Köln / "
           "AWK NRW; github.com/cceh/capitularia)")

EDITION = "bk-textzeuge"     # the Boretius-Krause edition, transcribed as a witness

# A chapter reference inside a manuscript: BK.221_6, Mordek.12_3, BK.221_inscriptio
_CORRESP = re.compile(r"^(BK|Mordek)\.(\d+[a-z]?)_(.+)$")


# ---------------------------------------------------------------------------
# language of a bibliography entry
# ---------------------------------------------------------------------------

# Ordered: the entry's own note is the strongest evidence ("enthält dt.
# Übersetzungen"), then the title's own language, then where it was printed.
_NOTE_LANG = [
    ("en", r"\bengl"), ("de", r"\bdt\.|\bdeutsch"), ("fr", r"\bfranz|\bfrz\."),
    ("it", r"\bital"), ("es", r"\bspan"), ("nl", r"\bniederl"),
]
_TITLE_WORDS = {
    "en": {"the", "of", "and", "in", "to", "translated", "sources", "from", "early",
           "medieval", "age", "for", "with", "history", "carolingian", "reader",
           "documents", "readings", "translation"},
    "de": {"der", "die", "das", "und", "zur", "zum", "des", "den", "quellen",
           "geschichte", "im", "mittelalter", "übersetzung", "ausgewählte", "von"},
    "fr": {"le", "la", "les", "des", "du", "et", "de", "textes", "histoire", "au", "en"},
    "it": {"il", "lo", "gli", "delle", "della", "di", "e", "storia", "fonti", "nel"},
}
_EN_PLACES = {"london", "oxford", "cambridge", "new york", "philadelphia", "boston",
              "kendal", "liverpool", "manchester", "toronto", "chicago", "princeton",
              "ithaca", "washington", "edinburgh", "harmondsworth", "harlow",
              "berkeley", "los angeles", "baltimore", "dublin", "new haven",
              "leiden/boston", "turnhout/london", "woodbridge", "abingdon"}
_DE_PLACES = {"berlin", "stuttgart", "darmstadt", "münchen", "leipzig", "hannover",
              "köln", "göttingen", "frankfurt", "wien", "zürich", "tübingen", "freiburg"}
_FR_PLACES = {"paris", "lyon", "bruxelles", "genève", "strasbourg"}
_RU_PLACES = {"kazan", "moskva", "moscow", "sankt-peterburg", "st petersburg"}

# Translations the capitulary records cite but the project bibliography never
# entered (25 of the 94 cited keys are missing from it). Only standard English
# source readers identified with certainty are listed; each is marked as
# supplementary in the evidence string. Deliberately absent: Domínguez 2014,
# cited by 68 capitularies, whose language could not be established -- those
# stay "unknown" rather than guessed.
SUPPLEMENTARY = {
    "Hillgarth_1986": ("en", "1986", "J. N. Hillgarth, Christianity and Paganism, 350-750 "
                                     "(Philadelphia 1986)"),
    "Fouracre_1996": ("en", "1996", "P. Fouracre & R. Gerberding, Late Merovingian France "
                                    "(Manchester 1996)"),
    "McNamara_1992": ("en", "1992", "J. A. McNamara et al., Sainted Women of the Dark Ages "
                                    "(Durham NC 1992)"),
    "Ehler_1954": ("en", "1954", "S. Z. Ehler & J. B. Morrall, Church and State through "
                                 "the Centuries (London 1954)"),
}


def norm_key(key: str) -> str:
    """Bibliography keys as the records spell them vary ('La Roncière_1969',
    'La_Ronciere_1969', 'Buehrer-Thierry_2010' vs 'Bührer-Thierry_2010')."""
    import unicodedata
    k = key.strip().lower().replace(" ", "_")
    k = k.replace("ü", "ue").replace("ö", "oe").replace("ä", "ae").replace("ß", "ss")
    k = unicodedata.normalize("NFKD", k)
    return "".join(ch for ch in k if not unicodedata.combining(ch))


def classify_language(note: str, title: str, place: str) -> Tuple[str, str]:
    """(language code or '?', how it was decided) for one translation entry."""
    n = (note or "").lower()
    for code, pat in _NOTE_LANG:
        if re.search(pat, n):
            return code, "bibliography note"
    words = set(re.findall(r"[a-zäöüéè]+", (title or "").lower()))
    scores = {code: len(words & vocab) for code, vocab in _TITLE_WORDS.items()}
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    if ranked[0] >= 2 and ranked[0] > ranked[1]:
        return best, "title language"
    p = (place or "").lower()
    if any(x in p for x in _EN_PLACES):
        return "en", "place of publication"
    if any(x in p for x in _DE_PLACES):
        return "de", "place of publication"
    if any(x in p for x in _FR_PLACES):
        return "fr", "place of publication"
    if any(x in p for x in _RU_PLACES):
        return "ru", "place of publication"
    if ranked[0] >= 1 and ranked[0] > ranked[1]:
        return best, "title language (weak)"
    return "?", "undetermined"


LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "it": "Italian",
              "es": "Spanish", "nl": "Dutch", "ru": "Russian", "?": "language unknown"}


@dataclass
class Translation:
    key: str
    title: str = ""
    year: str = ""
    place: str = ""
    lang: str = "?"
    decided_by: str = ""
    resolved: bool = False
    supplementary: bool = False

    @property
    def public_domain(self) -> bool:
        return self.year.isdigit() and int(self.year) < 1929

    def describe(self) -> str:
        if not self.resolved:
            return f"{self.key} (not in the project bibliography)"
        bits = [self.key.replace("_", " ")]
        if self.title:
            bits.append(f"'{self.title[:70]}'")
        extra = " [supplementary; not in project bibliography]" if self.supplementary else ""
        return f"{' '.join(bits)} -- {LANG_NAMES.get(self.lang, self.lang)}{extra}"



@dataclass
class Capitulary:
    ref: str                  # "BK.1" / "Mordek.12" -- the identifier this connector uses
    folder: str
    title: str
    date: str = ""
    year: Optional[int] = None
    translation_keys: List[str] = field(default_factory=list)
    witnesses: List[Tuple[str, int, int]] = field(default_factory=list)  # (ms, chapters, chars)


def _t(elem: Optional[ET.Element]) -> str:
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip() if elem is not None else ""


def _text_of_ab(ab: ET.Element) -> str:
    """The readable text of a diplomatic <ab>: corrections in, deletions,
    editorial notes and chapter numerals out, line breaks resolved."""
    out: List[str] = []

    def emit(el: ET.Element) -> None:
        """Append el's own content. Its *tail* is the parent's business: the
        tail sits outside the element, so it survives even when the element's
        content is dropped (text after a deleted word is still text)."""
        tag = el.tag.replace(TEI, "")
        if tag in ("note", "del", "surplus", "fw", "figure", "mentioned") or \
                (tag == "seg" and el.get("type") == "num"):
            return
        if tag == "choice":
            kids = {c.tag.replace(TEI, ""): c for c in el}
            pick = next((kids[k] for k in ("corr", "expan", "reg") if k in kids),
                        next(iter(el), None))
            if pick is not None:
                emit(pick)
            return
        if tag in ("lb", "cb", "pb"):
            # break="no" marks a word split across lines: join without a space.
            out.append("" if el.get("break") == "no" else " ")
            return
        if tag == "gap":
            out.append(" [...] ")
            return
        if el.text:
            out.append(el.text)
        for child in el:
            emit(child)
            if child.tail:
                out.append(child.tail)

    emit(ab)
    text = re.sub(r"\s+", " ", "".join(out)).strip()
    return re.sub(r"\s+([.,;:])", r"\1", text)


class CapitulariaConnector(Connector):
    name = "capitularia"

    def __init__(self, cache_dir: str | os.PathLike = DEFAULT_CACHE, timeout: float = 60.0,
                 workers: int = 8):
        self.cache = Path(cache_dir)
        self.timeout = timeout
        self.workers = workers
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "latin-library/1.0 (research)"})
        self._records: Optional[Dict[str, Capitulary]] = None
        self._bibl: Optional[Dict[str, Translation]] = None
        self._synced = False

    # -- local mirror --------------------------------------------------------

    def sync(self, force: bool = False) -> int:
        """Mirror the capit/, mss/ and bibliography XML; returns files downloaded."""
        if self._synced and not force:
            return 0
        self.cache.mkdir(parents=True, exist_ok=True)
        manifest_path = self.cache / "_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        tree = self.session.get(TREE, timeout=self.timeout).json().get("tree", [])
        wanted = [t for t in tree if t["path"].endswith(".xml") and (
            t["path"].startswith(("capit/pre814/", "capit/post840/", "capit/ldf/",
                                  "capit/undated/", "mss/"))
            or t["path"] == "bibl/Bibliographie_Capitularia.xml")]
        todo = [t for t in wanted
                if manifest.get(t["path"]) != t["sha"] or not (self.cache / t["path"]).exists()]

        def fetch(t: Dict[str, Any]) -> Tuple[str, str]:
            r = self.session.get(RAW + t["path"], timeout=self.timeout)
            r.raise_for_status()
            dest = self.cache / t["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            return t["path"], t["sha"]

        if todo:
            print(f"  capitularia: downloading {len(todo)} of {len(wanted)} files...", flush=True)
            with ThreadPoolExecutor(self.workers) as pool:
                for path, sha in pool.map(fetch, todo):
                    manifest[path] = sha
            manifest_path.write_text(json.dumps(manifest))
        self._synced = True
        return len(todo)

    # -- catalogue -------------------------------------------------------------

    def bibliography(self) -> Dict[str, Translation]:
        if self._bibl is None:
            self.sync()
            root = ET.parse(self.cache / "bibl" / "Bibliographie_Capitularia.xml").getroot()
            bibl: Dict[str, Translation] = {}
            for bs in root.iter(f"{TEI}biblStruct"):
                key = next((_t(i) for i in bs.iter(f"{TEI}idno")
                            if i.get("type") == "short_title"), "")
                if not key:
                    continue
                titles = [_t(t) for t in bs.iter(f"{TEI}title") if t.get("type") != "sub"]
                notes = {n.get("type"): _t(n) for n in bs.iter(f"{TEI}note")}
                place = next((_t(p) for p in bs.iter(f"{TEI}pubPlace")), "")
                date = next((d.get("when") or _t(d) for d in bs.iter(f"{TEI}date")), "")
                lang, how = classify_language(notes.get("notes", ""),
                                              titles[0] if titles else "", place)
                bibl[norm_key(key)] = Translation(
                    key=key, title=titles[0] if titles else "", year=str(date)[:4],
                    place=place, lang=lang, decided_by=how, resolved=True)
            for k, (lang, year, cite) in SUPPLEMENTARY.items():
                bibl.setdefault(norm_key(k), Translation(key=k, title=cite, year=year, lang=lang,
                                               decided_by="curated", resolved=True,
                                               supplementary=True))
            self._bibl = bibl
        return self._bibl

    def records(self) -> Dict[str, Capitulary]:
        """Every capitulary, keyed "BK.1" / "Mordek.12", with its witnesses."""
        if self._records is not None:
            return self._records
        self.sync()
        recs: Dict[str, Capitulary] = {}
        for folder in FOLDER_PERIOD:
            for path in sorted((self.cache / "capit" / folder).glob("*.xml")):
                m = re.match(r"(bk|mordek)-nr-(\w+)", path.stem)
                if not m:
                    continue
                ref = f"{'BK' if m.group(1) == 'bk' else 'Mordek'}.{m.group(2).lstrip('0') or '0'}"
                root = ET.parse(path).getroot()
                head = _t(root.find(f".//{TEI}body//{TEI}head"))
                title = re.sub(r"^(BK|Mordek)\s*Nr\.\s*\w+:\s*", "", head) or _t(
                    root.find(f".//{TEI}title"))
                # Only the dates the editors assign to the capitulary itself --
                # the header also carries the record's own date (2015-10-07),
                # which once made Clovis's letter a 21st-century text.
                dates = [d for n in root.iter(f"{TEI}note") if n.get("type") == "date"
                         for d in n.iter(f"{TEI}date")]
                d0 = next((d for d in dates if d.get("resp", "").startswith("Boretius")),
                          dates[0] if dates else None)
                # Boretius-Krause sometimes give only an upper bound ("800 oder
                # früher" for De villis): fall back to notAfter rather than no date.
                y = (d0.get("notBefore") or d0.get("when") or d0.get("notAfter") or ""
                     ) if d0 is not None else ""
                keys = [b.get("corresp", "").lstrip("#") for lb in root.iter(f"{TEI}listBibl")
                        if lb.get("type") == "translation" for b in lb.iter(f"{TEI}bibl")]
                recs[ref] = Capitulary(
                    ref=ref, folder=folder, title=title.strip(' "'),
                    date=_t(d0) if d0 is not None else "",
                    year=int(y[:4]) if y[:4].isdigit() else None,
                    translation_keys=[k for k in keys if k])
        # Witness coverage, from the manuscripts themselves.
        for path in sorted((self.cache / "mss").glob("*.xml")):
            try:
                root = ET.parse(path).getroot()
            except ET.ParseError:
                continue
            per: Dict[str, Tuple[set, int]] = {}
            for ab in root.iter(f"{TEI}ab"):
                for tok in (ab.get("corresp") or "").split():
                    m = _CORRESP.match(tok)
                    if not m:
                        continue
                    ref = f"{m.group(1)}.{m.group(2).lstrip('0') or '0'}"
                    chs, chars = per.get(ref, (set(), 0))
                    chs.add(m.group(3))
                    per[ref] = (chs, chars + len("".join(ab.itertext())))
            for ref, (chs, chars) in per.items():
                if ref in recs:
                    recs[ref].witnesses.append((path.stem, len(chs), chars))
        self._records = recs
        return recs

    def translation_status(self, cap: Capitulary) -> Tuple[str, str]:
        """(status, evidence) from the editors' own translation list."""
        if not cap.translation_keys:
            return UNTRANSLATED, "Capitularia lists no published translation"
        bibl = self.bibliography()
        found = [bibl.get(norm_key(k), Translation(key=k)) for k in cap.translation_keys]
        english = [t for t in found if t.lang == "en"]
        evidence = "; ".join(t.describe() for t in found)
        if english:
            if any(t.public_domain for t in english):
                return TRANSLATED, f"public-domain English translation listed: {evidence}"
            return TRANSLATED_PAYWALLED, f"English translation listed (in copyright): {evidence}"
        if all(t.resolved and t.lang != "?" for t in found):
            return UNTRANSLATED, f"only non-English translations listed: {evidence}"
        return UNKNOWN, f"translation(s) of undetermined language: {evidence}"

    def catalog(self, query: str = "all", limit: int = 400) -> List[Dict[str, Any]]:
        """Decidable records for the Find-texts tab.

        ``query``: ``all``, a reign folder (``pre814`` / ``ldf`` / ``post840``),
        ``untranslated``, a ref (``BK.139``), or words matched against titles.
        """
        q = (query or "all").strip()
        out = []
        for cap in self.records().values():
            status, evidence = self.translation_status(cap)
            if q.lower() in ("", "all"):
                pass
            elif q in FOLDER_PERIOD:
                if cap.folder != q:
                    continue
            elif q.lower() == "untranslated":
                if status != UNTRANSLATED:
                    continue
            elif re.match(r"(?i)^(bk|mordek)[. ]?\d", q):
                if cap.ref.lower() != self._norm_ref(q).lower():
                    continue
            elif q.lower() not in cap.title.lower():
                continue
            best = (next(w for w in cap.witnesses if w[0] == self.choose_witness(cap))
                    if cap.witnesses else None)
            out.append({
                "catalogue": "capitularia", "identifier": cap.ref,
                "title": f"{cap.title} ({cap.ref.replace('.', ' ')})",
                "author": None, "year": cap.year, "date": cap.date,
                "period": FOLDER_PERIOD[cap.folder],
                "url": f"https://capitularia.uni-koeln.de/capit/{cap.folder}/"
                       f"{'bk' if cap.ref.startswith('BK') else 'mordek'}-nr-"
                       f"{cap.ref.split('.')[1].zfill(3 if cap.ref.startswith('BK') else 2)}/",
                "fetchable": best is not None,
                "note": (f"{'edition text' if best[0] == EDITION else 'witness ' + best[0]}"
                         f" ({best[1]} chapters; {len(cap.witnesses)} witnesses)") if best
                        else "no transcribed witness yet",
                "translation_status": status, "translation_evidence": evidence,
                "genre": "capitulary", "lang_code": "la",
            })
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def choose_witness(cap: Capitulary) -> str:
        """The edition text if it is nearly complete, else the fullest manuscript."""
        best = max(cap.witnesses, key=lambda w: (w[1], w[2]))
        edition = next((w for w in cap.witnesses if w[0] == EDITION), None)
        if edition and edition[1] >= 0.8 * best[1]:
            return EDITION
        return best[0]

    def discover(self, query: str = "all", limit: int = 400) -> List[str]:
        return [r["identifier"] for r in self.catalog(query, limit) if r["fetchable"]]

    # -- fetch ---------------------------------------------------------------

    @staticmethod
    def _norm_ref(identifier: str) -> str:
        m = re.match(r"(?i)^\s*(bk|mordek)[ ._-]*(?:nr[ ._-]*)?0*(\w+)\s*$", identifier)
        if not m:
            raise ValueError(f"not a capitulary reference: {identifier!r} (try BK.139)")
        return f"{'BK' if m.group(1).lower() == 'bk' else 'Mordek'}.{m.group(2)}"

    def fetch(self, identifier: str, witness: Optional[str] = None, **meta_overrides) -> RawWork:
        ref = self._norm_ref(identifier)
        cap = self.records().get(ref)
        if cap is None:
            raise ValueError(f"{ref} is not in the Capitularia catalogue")
        if not cap.witnesses:
            raise ValueError(f"{ref} has no transcribed manuscript witness yet")
        ms = witness or self.choose_witness(cap)
        root = ET.parse(self.cache / "mss" / f"{ms}.xml").getroot()

        parts: List[Tuple[str, str]] = []
        for ab in root.iter(f"{TEI}ab"):
            toks = [_CORRESP.match(t) for t in (ab.get("corresp") or "").split()]
            chapters = [m.group(3) for m in toks
                        if m and f"{m.group(1)}.{m.group(2).lstrip('0') or '0'}" == ref]
            if not chapters:
                continue
            is_text = ab.get("type") == "text"
            if not is_text and not any(c.endswith("inscriptio") for c in chapters):
                continue            # explicits, rubric-only meta-text
            text = _text_of_ab(ab)
            if not text:
                continue
            ch = chapters[0]
            if ch.endswith("inscriptio") and not re.search(r"[^\W\d_]{2,}", text):
                continue            # a rubric that is only a numeral ("2.")
            # "2_inscriptio" is chapter 2's rubric: file it with chapter 2.
            num = re.match(r"^(\d+\w*?)(?:_inscriptio)?$", ch)
            label = (f"c. {num.group(1)}" if num
                     else "Inscriptio" if ch == "inscriptio" else ch.replace("_", " "))
            if parts and parts[-1][0] == label:
                parts[-1] = (label, parts[-1][1] + " " + text)
            else:
                parts.append((label, text))

        if ms == EDITION:
            shelfmark = "Boretius-Krause edition (MGH Capitularia regum Francorum, 1883-97)"
        else:
            ident = root.find(f".//{TEI}msIdentifier")
            shelfmark = ", ".join(x for x in (_t(ident.find(f"{TEI}settlement")),
                                              _t(ident.find(f"{TEI}repository")),
                                              _t(ident.find(f"{TEI}idno")))
                                  if x) if ident is not None else ms
        status, evidence = self.translation_status(cap)
        print(f"  translation status: {status} -- {evidence}", flush=True)
        meta = {
            "title": f"{cap.title} ({ref.replace('.', ' ')})",
            "author": None,
            "century": ((cap.year - 1) // 100 + 1) if cap.year else None,
            "genre": "capitulary",
            "language": "la",
            "language_stage": "late_antique" if (cap.year or 800) < 600 else "medieval",
            "source": f"Capitularia ({ref}; {'BK edition' if ms == EDITION else ms})",
            "shelfmark": shelfmark,
            "license": LICENSE,
            "translation_status": status,
            "has_existing_translation": status in (TRANSLATED, TRANSLATED_PAYWALLED),
            "translation_evidence": evidence,
        }
        meta.update({k: v for k, v in meta_overrides.items() if v is not None})
        return meta, parts
