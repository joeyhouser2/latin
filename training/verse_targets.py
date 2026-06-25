"""Curated targets for mining verse-translation parallel data (Workstream A).

Each entry pairs a Perseus source work (shared CTS scheme) with a public-domain
English *verse* translation on Project Gutenberg. Unlike the prose miner, these
are comparable texts aligned by `scripts/mine_verse.py` (the LaBSE / via-MT
aligner), not by citation.

IMPORTANT: the Gutenberg ids below are best-guess starting points and MUST be
verified before a real run — Gutenberg has many editions of each classic and ids
drift. Confirm at https://www.gutenberg.org (search `gutenberg_search`) and pass
the confirmed id to `mine_verse.py --gutenberg <id>`. `source_cts` ids are the
stable Perseus identifiers and are reliable.

`form` is the English verse form (for the `style` tag / later routing). `era` is
the source's language stage (note Homer/Hesiod = "archaic"; see the roadmap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VerseTarget:
    key: str                    # short slug for output filenames
    work: str                   # human title
    source_cts: str             # Perseus CTS id (group.work) — reliable
    src_lang: str               # "la" | "grc"
    greek_repo: bool            # mine from canonical-greekLit (vs latinLit)
    era: str                    # source language stage
    translator: str             # English verse translator
    form: str                   # "heroic_couplet" | "blank_verse" | "prose"
    gutenberg_search: str       # what to search on gutenberg.org
    gutenberg_id: Optional[int] = None   # BEST-GUESS — verify before use
    notes: str = ""


# Core set. Heroic couplets (Dryden/Pope) are the canonical English verse-
# translation register; Butler's prose is included as a faithful comparison anchor.
VERSE_TARGETS: List[VerseTarget] = [
    VerseTarget(
        key="aeneid_dryden", work="Virgil, Aeneid", source_cts="phi0690.phi003",
        src_lang="la", greek_repo=False, era="classical",
        translator="John Dryden", form="heroic_couplet",
        gutenberg_search="Aeneid Virgil Dryden", gutenberg_id=228,
        notes="VERIFIED 2026-06-24: 228 = Dryden, 12 books. Mined: 1101 pairs.",
    ),
    VerseTarget(
        key="georgics", work="Virgil, Georgics", source_cts="phi0690.phi002",
        src_lang="la", greek_repo=False, era="classical",
        translator="(verse; confirm translator)", form="verse",
        gutenberg_search="Georgics Virgil", gutenberg_id=232,
        notes="VERIFIED 2026-06-24: 232 = verse, 'GEORGIC I-IV' headers, 4 books. "
              "Mined: 369 pairs. (Translator not Dryden; confirm if it matters.)",
    ),
    VerseTarget(
        key="iliad_pope", work="Homer, Iliad", source_cts="tlg0012.tlg001",
        src_lang="grc", greek_repo=True, era="archaic",
        translator="Alexander Pope", form="heroic_couplet",
        gutenberg_search="Iliad Homer Pope", gutenberg_id=6130,
        notes="VERIFIED 2026-06-24: 6130 = Pope, 24 books. Mined: 402 pairs. "
              "Source doubles as Homeric-Greek training data (Workstream B).",
    ),
    VerseTarget(
        key="odyssey_pope", work="Homer, Odyssey", source_cts="tlg0012.tlg002",
        src_lang="grc", greek_repo=True, era="archaic",
        translator="Alexander Pope", form="heroic_couplet",
        gutenberg_search="Odyssey Homer Pope", gutenberg_id=3160,
        notes="VERIFIED 2026-06-24: 3160 = Pope, 24 books. Mined: 200 pairs.",
    ),
    VerseTarget(
        key="odyssey_butler", work="Homer, Odyssey", source_cts="tlg0012.tlg002",
        src_lang="grc", greek_repo=True, era="archaic",
        translator="Samuel Butler", form="prose",
        gutenberg_search="Odyssey Homer Butler", gutenberg_id=1727,
        notes="Prose; faithful anchor / Workstream-B Homeric pairs.",
    ),
    VerseTarget(
        key="metamorphoses_more", work="Ovid, Metamorphoses",
        source_cts="phi0959.phi006", src_lang="la", greek_repo=False,
        era="classical", translator="Brookes More", form="blank_verse",
        gutenberg_search="Metamorphoses Ovid Brookes More", gutenberg_id=21765,
        notes="VERIFIED 2026-06-24: Brookes More blank verse, split across TWO "
              "ebooks — 21765 (books I-VII -> 956 pairs) and 26073 (VIII-XV, "
              "word-ordinal headers -> 1156 pairs). Mine each separately by-book.",
    ),
    VerseTarget(
        key="theogony_anon", work="Hesiod, Theogony", source_cts="tlg0020.tlg001",
        src_lang="grc", greek_repo=True, era="archaic",
        translator="various (verify)", form="blank_verse",
        gutenberg_search="Hesiod Theogony Works and Days",
        notes="Already ingested as a reader doc; archaic Greek.",
    ),
]


def by_key(key: str) -> Optional[VerseTarget]:
    return next((t for t in VERSE_TARGETS if t.key == key), None)


def keys() -> List[str]:
    return [t.key for t in VERSE_TARGETS]
