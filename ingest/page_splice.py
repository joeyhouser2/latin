"""Recover a garbled segment's text by locating its position within the
page's OLD (original, already-in-corpus) OCR text, then mapping that
position through to the corresponding span of the page's NEW (Greek+Latin
Tesseract re-OCR) text via whole-page sequence alignment.

Anchoring off individual neighboring segments turned out to be fragile --
garbling often bleeds across a segment boundary (the segment right before a
"fully garbled" one may itself have a garbled tail), so a short exact-text
anchor frequently doesn't exist. Aligning the two FULL PAGES against each
other with difflib is more robust: real Latin words on either side of the
garbled span anchor the alignment naturally over a much longer stretch of
matching context, and small OCR disagreements within the surrounding real
text don't break it the way an exact short anchor phrase would.
"""
from __future__ import annotations

import difflib
import re
from typing import Optional

_ALNUM = re.compile(r"[A-Za-z0-9]+")


def _normalize_with_offsets(text: str):
    """Returns (normalized_string, list of original-text offsets, one per
    normalized character) so a position in the normalized string can be
    mapped back to a position in the original text."""
    norm_chars = []
    offsets = []
    for m in _ALNUM.finditer(text):
        for i, ch in enumerate(m.group()):
            norm_chars.append(ch.lower())
            offsets.append(m.start() + i)
    return "".join(norm_chars), offsets


def find_span(page_text: str, segment_text: str, search_from: int = 0) -> Optional[tuple]:
    """Locate segment_text's (normalized) span within page_text. Returns
    (start, end) character offsets into page_text, or None if not found."""
    norm_page, offsets = _normalize_with_offsets(page_text)
    norm_seg, _ = _normalize_with_offsets(segment_text)
    if not norm_seg:
        return None
    # try the full normalized segment; if OCR noise means it's not a clean
    # substring, fall back to a shrinking prefix.
    for length in (len(norm_seg), 30, 20, 12, 8, 4):
        key = norm_seg[:length]
        if len(key) < 4:
            break
        idx = norm_page.find(key, search_from)
        if idx != -1:
            end_key = norm_seg[max(0, len(norm_seg) - length):]
            end_idx = norm_page.rfind(end_key, idx)
            end_pos = (offsets[end_idx + len(end_key) - 1] + 1
                       if end_idx != -1 and end_idx + len(end_key) <= len(offsets)
                       else offsets[idx] + 1)
            return offsets[idx], max(end_pos, offsets[idx] + 1)
    return None


def map_span_to_new_page(old_page: str, new_page: str, old_start: int, old_end: int,
                          context: int = 400) -> Optional[str]:
    """Map a character span in old_page to the corresponding text in
    new_page via sequence alignment over a local window (not the whole page
    -- cheaper, and long-range alignment drifts more than it helps)."""
    lo = max(0, old_start - context)
    hi = min(len(old_page), old_end + context)
    old_window = old_page[lo:hi]
    rel_start, rel_end = old_start - lo, old_end - lo

    norm_old, old_offsets = _normalize_with_offsets(old_window)
    norm_new, new_offsets = _normalize_with_offsets(new_page)
    if not norm_old or not norm_new:
        return None

    # positions in norm_old corresponding to rel_start/rel_end
    def to_norm_pos(char_pos, offsets):
        for i, off in enumerate(offsets):
            if off >= char_pos:
                return i
        return len(offsets)

    norm_rel_start = to_norm_pos(rel_start, old_offsets)
    norm_rel_end = to_norm_pos(rel_end, old_offsets)

    sm = difflib.SequenceMatcher(None, norm_old, norm_new, autojunk=False)
    blocks = sm.get_matching_blocks()

    def map_pos(p, blocks):
        # find the matching block straddling or nearest to p in norm_old (a),
        # return corresponding position in norm_new (b)
        best = None
        for blk in blocks:
            if blk.a <= p <= blk.a + blk.size:
                return blk.b + (p - blk.a)
            if best is None or abs(blk.a - p) < abs(best.a - p):
                best = blk
        if best is None:
            return None
        return best.b + max(0, min(best.size, p - best.a))

    new_norm_start = map_pos(norm_rel_start, blocks)
    new_norm_end = map_pos(norm_rel_end, blocks)
    if new_norm_start is None or new_norm_end is None or new_norm_start >= new_norm_end:
        return None
    new_norm_start = max(0, min(new_norm_start, len(new_offsets) - 1))
    new_norm_end = max(0, min(new_norm_end, len(new_offsets) - 1))
    a, b = sorted((new_offsets[new_norm_start], new_offsets[new_norm_end]))
    return new_page[a:b + 1].strip()


def recover_segment(old_page: str, new_page: str, segment_text: str,
                     min_len: int = 8, min_ratio: float = 0.4) -> Optional[str]:
    """Full pipeline for one segment: locate + map + quality-gate.

    Returns the recovered text only if it was found AND is substantial
    relative to the original (a very short/truncated recovery -- common for
    tiny 1-3-word garbled segments, where there's too little context for the
    local alignment to anchor confidently -- isn't trustworthy enough to
    prefer over the honest untranslatable placeholder)."""
    span = find_span(old_page, segment_text)
    if span is None:
        return None
    recovered = map_span_to_new_page(old_page, new_page, *span)
    if not recovered:
        return None
    recovered = recovered.strip()
    orig_len = len(_normalize_with_offsets(segment_text)[0])
    rec_len = len(_normalize_with_offsets(recovered)[0])
    if len(recovered) < min_len or (orig_len and rec_len / orig_len < min_ratio):
        return None
    return recovered
