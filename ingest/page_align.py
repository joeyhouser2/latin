"""Sequentially align corpus segments to source PDF page numbers.

Works because both the segment order (from the original ingest) and the page
order (from the PDF) preserve the book's linear reading order, so a segment's
page number is always >= the previous segment's page number. This lets a
simple forward-scanning window (instead of an O(segments x pages) search)
find each segment's page: normalize both segment text and page text down to
bare alphanumerics (dropping whitespace/punctuation, which differ between
the two OCR extractions even when the underlying words agree), then check
whether the segment's leading fragment appears as a substring of any page in
a small forward window from the current pointer.

Garbled segments often won't match at all (that's the whole problem) -- when
no match is found, the pointer just doesn't advance, and the segment gets
tentatively assigned the current pointer's page. Since clean neighboring
segments re-anchor the pointer forward, an unmatched garbled segment's
tentative page is almost always correct anyway (it's necessarily between the
previous and next confidently-matched segment's pages).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_ALNUM = re.compile(r"[A-Za-z0-9]+")


def normalize(text: str) -> str:
    return "".join(_ALNUM.findall(text)).lower()


@dataclass
class AlignedSegment:
    segment_id: int
    page: int
    confident: bool  # True if we found an actual substring match


def align_segments(segments: List[tuple], pages: List[str],
                    key_len: int = 20, min_key_len: int = 10,
                    window: int = 6, max_window: int = 80) -> List[AlignedSegment]:
    """segments: list of (segment_id, latin_text) in document order.
    pages: list of page texts (0-indexed) in document order.

    The forward window widens the longer we go without a confident match
    (front matter -- title pages, dedications -- is exactly where short
    fragments like "Clavdio." fail to match reliably, and a fixed narrow
    window gets permanently stuck before ever reaching the first real
    content page)."""
    norm_pages = [normalize(p) for p in pages]
    results: List[AlignedSegment] = []
    pointer = 0
    stuck = 0
    for seg_id, text in segments:
        key = normalize(text)[:key_len]
        if len(key) < min_key_len:
            results.append(AlignedSegment(seg_id, pointer, False))
            continue
        w = min(window + stuck * 6, max_window)
        found = None
        for p in range(pointer, min(pointer + w, len(norm_pages))):
            if key in norm_pages[p]:
                found = p
                break
        if found is not None:
            pointer = found
            stuck = 0
            results.append(AlignedSegment(seg_id, found, True))
        else:
            stuck += 1
            results.append(AlignedSegment(seg_id, pointer, False))
    return results
