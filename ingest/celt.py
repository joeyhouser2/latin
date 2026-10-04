"""Connector for CELT, the Corpus of Electronic Texts (University College Cork).

CELT's Latin shelf is small -- 28 texts -- but it is Hiberno-Latin, the Latin
of early Christian Ireland and its missions, which the rest of the library
barely touches: saints' Lives, annals, penitential and canonical material.
Texts are served as HTML, one page per text, the TEI header first and the
text after it:

    https://celt.ucc.ie/published/L201040.html

Numbering is the useful part. CELT prefixes by language -- ``L`` Latin,
``G`` Irish, ``T`` English translation -- and a translation it publishes
itself shares its original's number: ``T201040`` is the English of
``L201040``. That makes the strongest translation check free.

Translation status is **verified in two stages**, strongest first:

1. **CELT's own twin** -- if ``T<same number>`` is published, a free English
   translation exists: ``translated`` (9 of the 28).
2. **The text's own bibliography** -- CELT headers list editions and
   translations ("Adomnán of Iona, Life of St Columba, translated by Richard
   Sharpe, London 1995"). Only lists *about the text* count: editions lists
   always, mixed lists ("Editions, secondary and reference works") only for
   entries that name the text, secondary-literature lists never -- audited
   on all 28 texts, every false positive came from those (a lecture titled
   'Translations and Adaptations in Irish', Galen translations, another
   author's history). A translation entry's language comes from explicit
   wording first ("German translation", "traduction", "with translation",
   "ed. and trans."), then -- in editions lists only -- its place of
   publication and its own language. English printed before 1929 (first year
   in the entry, not the reprint) gives ``translated``; later English gives
   ``translated_paywalled``; only non-English gives ``untranslated``; an
   editions list recording no translation gives ``untranslated``, said so;
   no editions list to check stays ``unknown``.

Result on the full shelf: 13 translated, 3 in copyright, 12 untranslated --
the untranslated being the Irish annals, *Vita Ite*, the Hiberno-Latin hymn
*Adelphus adelpha mater*, the *Regimen na Sláinte* medical texts and others.

Text extraction handles CELT's habits: English editorial prefaces inside the
text body are dropped paragraph by paragraph by language; inline tags are
removed without a space (they mark expansions mid-word); folio markers
("{MS [A] folio 22}") and sigla lists are stripped; verse (numbered lines
separated by <br>) becomes one segment per line.

The evidence -- the twin's id or the bibliography entry itself -- travels with
the document as ``translation_evidence``.

    python scripts/ingest.py celt L100003
    python scripts/ingest.py celt untranslated --discover --limit 30

Pages are cached under ``data/raw/celt``.
"""

from __future__ import annotations

import html as _html
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .base import Connector, RawWork
from .translation_status import (TRANSLATED, TRANSLATED_PAYWALLED, UNKNOWN,
                                 UNTRANSLATED)

BASE = "https://celt.ucc.ie/"
INDEX = BASE + "publishd.html"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO_ROOT / "data" / "raw" / "celt"
LICENSE = "CELT, University College Cork: free for research and non-commercial use"

# A bibliography entry that describes a translation of the text.
_TRANSLATION = re.compile(
    r"\b(translat\w*|transl\.|übersetz\w*|traduct\w*|tradu[ci]t\w*|tradott\w*|"
    r"traduzion\w*|traducción|vertaling)", re.I)
# Explicit non-English markers, checked before assuming English. Includes Irish:
# a translation into Modern Irish is not an English translation.
_OTHER_LANG = [
    ("de", re.compile(r"\bgerman\b|übersetz|deutsch", re.I)),
    ("fr", re.compile(r"\bfrench\b|traduct|traduit|française|français", re.I)),
    ("it", re.compile(r"\bitalian\b|tradott|traduzion|italiana", re.I)),
    ("es", re.compile(r"\bspanish\b|traducción|española", re.I)),
    ("nl", re.compile(r"\bdutch\b|vertaling", re.I)),
    ("ga", re.compile(r"\b(modern )?irish translation\b|into irish\b", re.I)),
]
# English only on an explicit statement that this entry *is* a translation.
# A bare "[partial translation]" or "Translations and Adaptations in Irish"
# (a lecture title) is not enough -- both produced false positives in testing.
_ENGLISH_PHRASE = re.compile(
    r"\benglish\b|translated\s+(by|from|into|with)|with\s+(an?\s+)?([\w-]+,?\s+){0,3}"
    r"translations?\b|\(?\bedd?s?\.?\s*(and|&)\s*trans\.?|\(trans\.\)|\btrans\.,|"
    r"text\s+and\s+translation|translation,?\s+(and|with)\s+(notes|introduction|commentary)",
    re.I)

# Which bibliography lists count as evidence about *this* text. Audited on all
# 28 texts: every false positive came from secondary-literature lists (Galen
# translations, a lecture on translation into Irish, another author's history),
# every real translation from an editions list.
_EDITION_HEAD = re.compile(r"edition|translation|printed sources? (for|of) (the )?latin", re.I)
_SECONDARY_HEAD = re.compile(r"secondary|reading|literature|reference|comment|bibliograph|"
                             r"written works|printed source material", re.I)
_SKIP_HEAD = re.compile(r"manuscript|\bms\b|\bmss\b|internet|digital images", re.I)
_STOP = {"the", "and", "of", "de", "in", "ad", "the", "sancti", "sancte", "sanctae", "with",
         "life", "vita", "liber", "written", "by", "a", "an", "et"}


# Inline tags vanish without a space: CELT marks expanded abbreviations and
# emphasis *inside* words, and turning every tag into a space broke "quondam"
# into "Quo n dam". Everything else (p, li, br, h*) is a word boundary.
_INLINE_TAG = re.compile(r"</?(?:i|b|em|strong|span|sup|sub|u|font|abbr|cite|tt|q)\b[^>]*>", re.I)
# The edition's own navigation, inline in the text: "{MS [A] folio 22, a 1}",
# "{Geyer ed page 221}", "[supra 2717]".
_MARKERS = re.compile(r"\{[^{}]{0,120}\}|\[supra\s*[\d,.]+\]|^\s*supra\s+\d+\s+|"
                      r"\(?https?://[^\s)]+\)?", re.I)


def _clean(fragment: str) -> str:
    text = _INLINE_TAG.sub("", fragment)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", _html.unescape(text)).strip()


_PLACES = [
    ("de", re.compile(r"\b(halle|leipzig|berlin|münchen|munich|göttingen|tübingen|stuttgart|"
                      r"freiburg|heidelberg|wien|vienna|bonn|hannover|breslau|darmstadt)\b", re.I)),
    ("fr", re.compile(r"\b(paris|lyon|bruxelles|louvain|genève)\b", re.I)),
    ("it", re.compile(r"\b(roma|milano|firenze|torino|bologna|napoli|spoleto)\b", re.I)),
    ("en", re.compile(r"\b(london|oxford|cambridge|dublin|edinburgh|glasgow|belfast|cork|"
                      r"new york|philadelphia|boston|toronto|manchester|woodbridge|"
                      r"baltimore|washington|kalamazoo|turnhout)\b", re.I)),
]
_EN_PROSE = re.compile(r"\b(and|the|of|with|text|by)\b", re.I)


def classify_entry(entry: str, trusted: bool = False) -> str:
    """Language code of the translation a bibliography entry describes.

    Explicit markers first ("German translation", "traduction"), then an
    explicit English statement ("with translation", "ed. and trans."). For an
    entry in a *trusted* editions list only, two weaker signals follow: the
    place of publication (Schade's "[partial translation]", Halle 1870, is not
    English) and, failing that, the entry's own language (Ó Cróinín's "text
    (283), facsimile (284), translation (285)" in an English journal is).
    """
    for code, pat in _OTHER_LANG:
        if pat.search(entry):
            return code
    if _ENGLISH_PHRASE.search(entry):
        return "en"
    if trusted:
        for code, pat in _PLACES:
            if pat.search(entry):
                return code
        if len(_EN_PROSE.findall(entry)) >= 2:
            return "en"
    return "?"


# Apparatus headings. Matched with the list directly under them (the sigla),
# not as whole sections: in "Adelphus adelpha mater" the poem itself sits
# under "List of witnesses", after the list, with no heading of its own.
_APPARATUS_BLOCK = re.compile(
    r"<h[2-5][^>]*>\s*(?:list of )?(?:witness\w*|sigla|manuscripts?|mss\.?|abbreviations)"
    r"\s*</h[2-5]>\s*<(ul|ol)\b.*?</\1>", re.I | re.S)
_EN_WORDS = re.compile(r"\b(the|of|and|is|this|that|which|was|from|it|by|with|has|been|"
                       r"there|also|who|his|are|were|but|to|as|for|on|not|he|had|be|"
                       r"they|their|time|what)\b", re.I)
_LA_WORDS = re.compile(r"\b(et|est|ad|cum|non|qui|quae|quod|de|sed|ut|per|in|eius|sunt|"
                       r"autem|enim|ab|ex|se|ille|illa|sancta|sanctus|deus|dei)\b", re.I)


def _is_english(para: str) -> bool:
    """An editor's English paragraph (preface, commentary) rather than Latin.

    CELT editions put the editor's introduction inside the text body -- Vita
    Ite opens with Plummer's "This life is from M f. 109c..." -- sometimes
    under no heading of its own, so it is caught by its language instead.
    """
    en, la = len(_EN_WORDS.findall(para)), len(_LA_WORDS.findall(para))
    # Short English footnotes ("through its use, a readier access to the Irish
    # heart") carry only three marker words; with no Latin at all, three is enough.
    return (en >= 4 and en > 2 * la) or (en >= 3 and la == 0)


def _section_text(chunk: str) -> Tuple[str, bool]:
    """(Latin text of one section, whether it is verse).

    Verse is lines separated by <br> ("19] Gibro praxon agathon<br>20] ...");
    it keeps one line per line, with the editor's line numbers removed. Prose
    is paragraphs, with English editorial paragraphs dropped.
    """
    brs = len(re.findall(r"<br\s*/?>", chunk, re.I))
    paras = len(re.findall(r"<p\b", chunk, re.I))
    blocks = re.split(r"</?p\b[^>]*>|</?li\b[^>]*>|</?(?:ol|ul)\b[^>]*>", chunk, flags=re.I)
    if brs >= 4 and brs > paras:
        lines = []
        for block in blocks:
            for raw in re.split(r"<br\s*/?>", block, flags=re.I):
                line = re.sub(r"^\s*\d+\]\s*", "", _MARKERS.sub(" ", _clean(raw))).strip()
                if line and not _is_english(line):
                    lines.append(line)
        return "\n".join(lines), True
    kept = [p for p in (re.sub(r"\s+", " ", _MARKERS.sub(" ", _clean(b))).strip()
                        for b in blocks) if p and not _is_english(p)]
    return " ".join(kept), False


def _year(entry: str) -> Optional[int]:
    """Original publication year: the *first* year, so "Oxford 1922, repr. 1968"
    counts as 1922 (public domain), not 1968."""
    years = [int(y) for y in re.findall(r"\b(1[5-9]\d\d|20\d\d)\b", entry)]
    return years[0] if years else None


def _list_kind(heading: str) -> str:
    """'edition' (trusted), 'mixed' (editions among other things), 'other', or 'skip'."""
    if _SKIP_HEAD.search(heading):
        return "skip"
    ed, sec = bool(_EDITION_HEAD.search(heading)), bool(_SECONDARY_HEAD.search(heading))
    if ed and not sec:
        return "edition"
    if ed or re.search(r"printed source", heading, re.I):
        return "mixed"
    return "other"


class CELTConnector(Connector):
    name = "celt"

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE, timeout: float = 60.0,
                 workers: int = 6):
        self.cache = Path(cache_dir)
        self.timeout = timeout
        self.workers = workers
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "latin-library/1.0 (research)"})
        self._index: Optional[Tuple[List[str], set]] = None

    # -- index & pages -------------------------------------------------------

    def index(self) -> Tuple[List[str], set]:
        """(Latin ids in index order, set of numbers that have a T translation)."""
        if self._index is None:
            page = self.session.get(INDEX, timeout=self.timeout).text
            ids = re.findall(r'published/([A-Z])(\d+[A-Z]?)/index\.html', page)
            latin = list(dict.fromkeys(f"L{n}" for k, n in ids if k == "L"))
            twins = {n for k, n in ids if k == "T"}
            self._index = (latin, twins)
        return self._index

    def page(self, text_id: str) -> str:
        self.cache.mkdir(parents=True, exist_ok=True)
        path = self.cache / f"{text_id}.html"
        if path.exists():
            return path.read_text(encoding="utf-8")
        r = self.session.get(f"{BASE}published/{text_id}.html", timeout=self.timeout)
        r.raise_for_status()
        # CELT pages declare no charset reliably; they are UTF-8 in practice.
        r.encoding = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else "utf-8"
        path.write_text(r.text, encoding="utf-8")
        return r.text

    @staticmethod
    def split(page: str, text_id: str) -> Tuple[str, str]:
        """(header html, body html). The body starts at the second
        "Corpus of Electronic Texts Edition: <id>" banner."""
        marks = [m.start() for m in re.finditer(
            r"Corpus of Electronic Texts Edition:?\s*" + re.escape(text_id), page)]
        if len(marks) >= 2:
            cut = page.rfind("<h4", 0, marks[1])
            return page[:cut if cut > 0 else marks[1]], page[marks[1]:]
        # Fallback: after the revision history list.
        i = page.find("Revision History")
        j = page.find("</ul>", i) if i > 0 else -1
        return (page[:j], page[j:]) if j > 0 else (page, "")

    # -- translation status ----------------------------------------------------

    def bibliography(self, header: str) -> List[Tuple[str, str, str]]:
        """(list kind, heading, entry text) for each bibliography item in the
        header; manuscript and digital-image lists are left out."""
        out = []
        for lst in re.finditer(r"<(ol|ul)[^>]*>(.*?)</\1>", header, re.S | re.I):
            body = lst.group(2)
            head = re.search(r"<lh>(.*?)</lh>", body, re.S | re.I)
            heading = _clean(head.group(1)) if head else ""
            kind = _list_kind(heading) if heading else "skip"
            if kind == "skip":
                continue
            for li in re.findall(r"<li>(.*?)(?=<li>|$)", body, re.S | re.I):
                entry = _clean(li)
                if len(entry) > 15:
                    out.append((kind, heading, entry))
        return out

    def translation_status(self, text_id: str, header: str,
                           title: str = "") -> Tuple[str, str]:
        num = text_id[1:]
        _, twins = self.index()
        if num in twins:
            return TRANSLATED, (f"free English translation published on CELT: T{num} "
                                f"({BASE}published/T{num}.html)")
        entries = self.bibliography(header)
        # Words that identify the text, for entries in mixed lists: an entry
        # there only counts if it names the text ("Visio Tnugdali"), which is
        # what separates an edition of *this* work from a neighbour's.
        keys = {w for w in re.findall(r"[^\W\d_]{4,}", title.lower()) if w not in _STOP}

        def about_this(kind: str, entry: str) -> bool:
            return kind == "edition" or (kind == "mixed" and any(k in entry.lower() for k in keys))

        candidates = [(k, e) for k, _, e in entries if about_this(k, e)]
        found = [(classify_entry(e, trusted=(k == "edition")), _year(e), e)
                 for k, e in candidates if _TRANSLATION.search(e)]
        english = [f for f in found if f[0] == "en"]

        def cite(f):
            return f"{f[2][:170]}{'...' if len(f[2]) > 170 else ''}"

        if english:
            pd = [f for f in english if f[1] and f[1] < 1929]
            if pd:
                return TRANSLATED, f"public-domain English translation ({pd[0][1]}): {cite(pd[0])}"
            return TRANSLATED_PAYWALLED, f"English translation listed: {cite(english[0])}"
        if any(f[0] not in ("en", "?") for f in found) and not any(f[0] == "?" for f in found):
            langs = sorted({f[0] for f in found})
            return UNTRANSLATED, (f"only non-English translations listed ({', '.join(langs)}): "
                                  f"{cite(found[0])}")
        if found:
            return UNKNOWN, f"a translation of undetermined language is listed: {cite(found[0])}"
        editions = [e for k, _, e in entries if k == "edition"]
        if editions:
            return UNTRANSLATED, (f"CELT's editions list ({len(editions)} entries) records no "
                                  f"translation")
        return UNKNOWN, "CELT's header has no editions list to check against"

    # -- catalogue / discover ------------------------------------------------

    def _describe(self, text_id: str) -> Dict[str, Any]:
        page = self.page(text_id)
        header, body = self.split(page, text_id)
        title = _clean((re.search(r"<h1[^>]*>(.*?)</h1>", header, re.S) or [None, text_id])[1])
        author = re.search(r"Author:\s*(?:</?\w+[^>]*>\s*)*([^<\n]+)", header)
        # "Date range: c. 1162&#0150;1370." -- read to the tag, not the first
        # full stop, or "c." swallows the dates.
        dr = re.search(r"Date range:\s*([^<]+)", page)
        years = [int(y) for y in re.findall(r"\b(\d{3,4})\b", _html.unescape(dr.group(1))
                                            .replace("\x96", "-"))
                 if 300 <= int(y) <= 1900] if dr else []
        status, evidence = self.translation_status(text_id, header, title)
        return {
            "catalogue": "celt", "identifier": text_id, "title": title,
            "author": _clean(author.group(1)) if author else None,
            "year": years[0] if years else None,
            "date": _clean(dr.group(1)) if dr else "",
            "url": f"{BASE}published/{text_id}.html",
            "fetchable": len(_clean(body)) > 200,
            "translation_status": status, "translation_evidence": evidence,
            "lang_code": "la", "genre": None,
        }

    def catalog(self, query: str = "all", limit: int = 100) -> List[Dict[str, Any]]:
        """Every Latin text with its verified status. ``query``: ``all``,
        ``untranslated``, a text id, or words matched against title/author."""
        latin, _ = self.index()
        q = (query or "all").strip()
        ids = [q.upper()] if re.match(r"(?i)^L\d", q) else latin
        with ThreadPoolExecutor(self.workers) as pool:
            recs = list(pool.map(self._safe_describe, ids))
        out = []
        for r in recs:
            if r is None:
                continue
            if q.lower() == "untranslated" and r["translation_status"] != UNTRANSLATED:
                continue
            if q.lower() not in ("all", "untranslated", "") and not re.match(r"(?i)^L\d", q) \
                    and q.lower() not in f"{r['title']} {r['author'] or ''}".lower():
                continue
            out.append(r)
        return out[:limit]

    def _safe_describe(self, text_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self._describe(text_id)
        except requests.RequestException:
            return None

    def discover(self, query: str = "all", limit: int = 100) -> List[str]:
        return [r["identifier"] for r in self.catalog(query, limit) if r["fetchable"]]

    # -- fetch -----------------------------------------------------------------

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        text_id = identifier.strip().upper()
        if not re.match(r"^L\d+[A-Z]?$", text_id):
            raise ValueError(f"not a CELT Latin text id: {identifier!r} (e.g. L100003)")
        info = self._describe(text_id)
        _, body = self.split(self.page(text_id), text_id)

        # Page-number anchors ("p.303") and the repeated title banner are
        # apparatus, not text.
        body = re.sub(r"<small>\s*<a name=[^>]*>.*?</a>\s*</small>", " ", body, flags=re.S | re.I)
        body = re.sub(r"<a name=\"[^\"]*\">[^<]*</a>", " ", body, flags=re.I)
        body = re.sub(r"<h4[^>]*>\s*Corpus of Electronic Texts Edition.*?</h4>", " ", body,
                      flags=re.S | re.I)
        body = re.sub(r"<h3[^>]*>\s*<cite>.*?</h3>", " ", body, count=1, flags=re.S | re.I)
        body = _APPARATUS_BLOCK.sub(" ", body)

        parts: List[Tuple[str, str]] = []
        verse_parts = 0
        pieces = re.split(r"<h[2-5][^>]*>(.*?)</h[2-5]>", body, flags=re.S | re.I)
        # re.split with one group: [pre, head1, text1, head2, text2, ...]
        for head, chunk in [("Text", pieces[0])] + list(zip(pieces[1::2], pieces[2::2])):
            label = _clean(head)[:80] or "Section"
            # Editorial introductions are dropped by language, paragraph by
            # paragraph (see _is_english), not by heading.
            text, is_verse = _section_text(chunk)
            if text:
                parts.append((label, text))
                verse_parts += is_verse

        print(f"  translation status: {info['translation_status']} -- "
              f"{info['translation_evidence']}", flush=True)
        year = info["year"]
        meta = {
            "title": info["title"],
            "author": info["author"],
            "century": ((year - 1) // 100 + 1) if year else None,
            "language": "la",
            "language_stage": ("late_antique" if year and year < 600 else
                               "medieval" if year and year < 1500 else
                               "early_modern" if year else "medieval"),
            "source": f"CELT ({text_id})",
            "license": LICENSE,
            "translation_status": info["translation_status"],
            "has_existing_translation": info["translation_status"] in (TRANSLATED,
                                                                       TRANSLATED_PAYWALLED),
            "translation_evidence": info["translation_evidence"],
        }
        if parts and verse_parts > len(parts) / 2:
            # Mostly verse: one segment per line, which is how the reader and
            # scansion want it -- the sentence splitter found almost no
            # punctuation in the hymns and ran 60 lines into 6 segments.
            meta["_verse"] = True
        meta.update({k: v for k, v in meta_overrides.items() if v is not None})
        return meta, parts
