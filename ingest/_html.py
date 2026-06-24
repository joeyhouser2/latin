"""Shared HTML -> paragraphs extraction for HTML-based connectors.

Collects text from <p> elements while skipping configured tags (script/style)
and elements whose CSS class matches a skip set (e.g. MediaWiki's edit links).
Tolerant of the lightly-malformed HTML these sites serve.
"""

from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
from typing import List, Set
import re


_DEFAULT_SKIP_TAGS = {"script", "style", "head", "sup"}


class ParagraphExtractor(HTMLParser):
    def __init__(self, skip_tags: Set[str] = None, skip_classes: Set[str] = None):
        super().__init__()
        self.skip_tags = set(skip_tags) if skip_tags else set(_DEFAULT_SKIP_TAGS)
        self.skip_classes = set(skip_classes) if skip_classes else set()
        self._skip_stack: List[str] = []   # open tags that triggered a skip
        self._in_p = False
        self._buf: List[str] = []
        self.paragraphs: List[str] = []

    def _should_skip(self, tag, attrs) -> bool:
        if tag in self.skip_tags:
            return True
        classes = set((dict(attrs).get("class") or "").split())
        return bool(classes & self.skip_classes)

    def handle_starttag(self, tag, attrs):
        if self._should_skip(tag, attrs):
            self._skip_stack.append(tag)
            return
        if self._skip_stack:
            return
        if tag == "p":
            self._in_p = True
            self._buf = []
        elif tag == "br" and self._in_p:
            self._buf.append(" ")

    def handle_endtag(self, tag):
        if self._skip_stack and self._skip_stack[-1] == tag:
            self._skip_stack.pop()
            return
        if self._skip_stack:
            return
        if tag == "p" and self._in_p:
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if text:
                self.paragraphs.append(text)
            self._in_p = False
            self._buf = []

    def handle_data(self, data):
        if not self._skip_stack and self._in_p:
            self._buf.append(data)


def extract_paragraphs(html: str, skip_classes: Set[str] = None,
                       skip_tags: Set[str] = None) -> List[str]:
    parser = ParagraphExtractor(skip_tags=skip_tags, skip_classes=skip_classes)
    parser.feed(html)
    return [unescape(p) for p in parser.paragraphs]
