"""Turn per-page OCR output into continuous text a segmenter can use.

Shared by the MDZ connector (which receives hOCR) and the image-OCR connector
(which receives Tesseract plain text). Page-structured OCR has artefacts that
plain running text does not, and each one corrupts sentence segmentation or
translation if left in:

* **line-end hyphenation** -- ``obli-`` / ``gatio`` must rejoin;
* **running heads, folio numbers and signature marks** ("A ij", "12") that sit
  alone on a line at the top or bottom of the page;
* **catchwords** -- early printed books repeat the first word of the next page
  at the foot of each page, which would otherwise be read twice;
* **sentences that span a page break** -- a page ending mid-sentence must join
  the next page's opening rather than be split into two paragraphs.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import List, Optional

# end-of-line hyphen glyphs seen in OCR of early prints: ASCII/soft hyphen,
# not-sign (Tesseract's rendering of the double hyphen), double oblique hyphen.
_HYPHENS = "-­¬‐‑⸗⹀⧼"
_LINE_END_HYPHEN = re.compile(rf"([^\W\d_])[{re.escape(_HYPHENS)}]\s*$")
_FOLIO_LINE = re.compile(r"^[\s\[\]().,:;*\-–—|]*"
                         r"(\d{1,4}|[ivxlcdm]{1,8}|[IVXLCDM]{1,8}|[A-Za-z]{1,2}\s?[ivxIVXjJ\d]{1,3}|\d{1,2}\s?[ivxIVXjJ]{1,3})"
                         r"[\s\[\]().,:;*\-–—|]*$")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_MARGIN_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
_SENTENCE_END = re.compile(r"[.!?:;…»\"')\]]\s*$")


class _HOCRParser(HTMLParser):
    """Collect hOCR into paragraphs of lines (a list of lists of strings)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paragraphs: List[List[str]] = []
        self._par: Optional[List[str]] = None
        self._line: Optional[List[str]] = None
        self._line_depth = 0
        self._depth_stack: List[bool] = []   # per open <span>: is it an ocr_line?

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if tag == "p" and "ocr_par" in cls:
            self._flush_line()
            self._par = []
        elif tag == "span":
            is_line = "ocr_line" in cls or "ocr_header" in cls or "ocr_caption" in cls
            self._depth_stack.append(is_line)
            if is_line:
                self._flush_line()
                self._line = []
        elif tag == "div" and "ocrx_block" in cls:
            self._close_par()

    def handle_endtag(self, tag):
        if tag == "span" and self._depth_stack:
            if self._depth_stack.pop():
                self._flush_line()
        elif tag == "p":
            self._close_par()

    def handle_data(self, data):
        if self._line is not None:
            self._line.append(data)

    def _flush_line(self):
        if self._line is not None:
            text = re.sub(r"\s+", " ", "".join(self._line)).strip()
            if text:
                if self._par is None:
                    self._par = []
                self._par.append(text)
            self._line = None

    def _close_par(self):
        self._flush_line()
        if self._par:
            self.paragraphs.append(self._par)
        self._par = None

    def close(self):
        super().close()
        self._close_par()


def hocr_to_text(html: str) -> str:
    """Plain text (paragraphs separated by a blank line, lines by newlines)."""
    parser = _HOCRParser()
    parser.feed(html)
    parser.close()
    return "\n\n".join("\n".join(par) for par in parser.paragraphs)


def _is_furniture(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line) <= 12 and bool(_FOLIO_LINE.match(line))


def _clean_page(text: str, next_page: Optional[str]) -> str:
    """Drop folio/signature lines at the page edges and a trailing catchword."""
    lines = [l.rstrip() for l in text.replace("\r\n", "\n").split("\n")]
    # edge furniture only: a bare "12" mid-page might be a real number
    while lines and (not lines[0].strip() or _is_furniture(lines[0])):
        lines.pop(0)
    while lines and (not lines[-1].strip() or _is_furniture(lines[-1])):
        lines.pop()
    # catchword: the last line is one token matching the start of the next page
    if lines and next_page is not None:
        last = lines[-1].strip()
        nxt = _WORD.search(next_page)
        toks = _WORD.findall(last)
        if len(toks) == 1 and nxt and len(toks[0]) >= 2:
            a, b = toks[0].lower(), nxt.group().lower()
            if b.startswith(a[:max(2, len(a) - 1)]) or a.startswith(b[:max(2, len(b) - 1)]):
                lines.pop()
    return "\n".join(lines)


def join_pages(pages: List[str]) -> str:
    """Join per-page OCR text into one blob with paragraph (blank-line) breaks.

    Returns text whose paragraphs are separated by blank lines and whose lines
    within a paragraph have been merged (dehyphenated).
    """
    cleaned: List[str] = []
    for i, page in enumerate(pages):
        nxt = pages[i + 1] if i + 1 < len(pages) else None
        cleaned.append(_clean_page(page or "", nxt))

    paragraphs: List[str] = []
    for page in cleaned:
        for par in re.split(r"\n\s*\n", page):
            if not par.strip():
                continue
            # a stray number alone on its line is a margin/line-count mark
            par = "\n".join(l for l in par.split("\n") if not _MARGIN_NUMBER.match(l))
            if not par.strip():
                continue
            merged = _merge_lines(par)
            # OCR layout analysis often cuts a paragraph mid-sentence (column
            # or page break, indented verse of a quotation): rejoin when the
            # previous block is unfinished and this one carries on in lowercase.
            if paragraphs and merged[:1].islower() and (
                    _LINE_END_HYPHEN.search(paragraphs[-1])
                    or not _SENTENCE_END.search(paragraphs[-1])):
                paragraphs[-1] = _join_fragment(paragraphs[-1], merged)
            else:
                paragraphs.append(merged)
    return _fold_long_s("\n\n".join(paragraphs))


def _fold_long_s(text: str) -> str:
    """Long s (U+017F) is always plain s; models and the embedder don't know it."""
    return text.replace("ſ", "s")


def _merge_lines(paragraph: str) -> str:
    out = ""
    for line in paragraph.split("\n"):
        line = line.strip()
        if not line:
            continue
        out = _join_fragment(out, line) if out else line
    return out


def _join_fragment(head: str, tail: str) -> str:
    """Append ``tail`` to ``head``, healing an end-of-line hyphen when the
    continuation starts lowercase (so 'Hans-\\nMüller' style names survive)."""
    m = _LINE_END_HYPHEN.search(head)
    if m and tail[:1].islower():
        return head[:m.end(1)] + tail
    return head + " " + tail


def latin_function_word_rate(text: str) -> float:
    """Share of tokens that are very common Latin function words.

    Real Latin prose scores roughly 0.12-0.25; German/French/English text or
    OCR noise scores near zero. A cheap sanity check that the OCR of a
    'Latin' item is Latin at all -- not a quality metric.
    """
    toks = [t.lower() for t in _WORD.findall(text)]
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in _LATIN_FUNCTION_WORDS)
    return hits / len(toks)


_LATIN_FUNCTION_WORDS = frozenset(
    "et in ad cum non est ut quae qui quod sed de ex per ab si nec enim autem "
    "sunt esse etiam atque vel aut quam quo sit quia ac neque nam tamen hoc "
    "haec ei eius ille illa id ita sic uti sive seu pro sub sine ante post "
    "inter apud contra ergo igitur quoque nisi ne".split()
)
