"""How interesting is a summarized work? A local-LLM rating, for sorting.

The Summaries tab can sort by this, so the hand-written travel account or the
monk's vision of hell surfaces above the fortieth volume of sequences for
feasts. The model reads only the catalogue summary and topics -- not the
text -- so a rating costs one short prompt (~2 s), and the whole library can be
rated in well under an hour.

Asking a model for one 1-10 score gets 7 for everything. So it scores four
narrower things, 1-5 each, against anchored descriptions, and the overall
score is computed from those, not asked for:

* ``subject``  -- unusual or surprising content, vs. the formulaic;
* ``story``    -- narrative, human detail, an individual voice;
* ``history``  -- evidence about people, places and daily life;
* ``neglect``  -- how likely the work is to be little-studied and untranslated.
  This library exists to surface the understudied, so Cicero scores 1 here.

A garbled text (the summary says the OCR is too poor to tell) is capped, since
there is nothing yet to read.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

AXES = ("subject", "story", "history", "neglect")
GARBLED_CAP = 3.0

SCHEMA = {
    "type": "object",
    "properties": {
        **{a: {"type": "integer", "minimum": 1, "maximum": 5} for a in AXES},
        "garbled": {"type": "boolean"},
        "hook": {"type": "string"},
    },
    "required": [*AXES, "garbled", "hook"],
}

SYSTEM = (
    "You are an editor choosing, from a catalogue of mostly untranslated Latin and "
    "Greek texts, which ones a curious reader would most want to open. You are "
    "discriminating: most entries in such a catalogue are routine -- liturgy, "
    "charters, school verse, standard theology -- and you score them low without "
    "apology, keeping high scores for the genuinely unusual."
)

RUBRIC = """Score each from 1 to 5. Use the whole scale; a typical entry is 2.
- "subject": 1 = formulaic or routine for its genre (hymns for feasts, deeds of gift,
  commonplace devotion); 3 = a normal example of an interesting genre; 5 = strange,
  surprising or one of a kind (a vision of the otherworld, a voyage, a trial, a
  quarrel, a scientific oddity).
- "story": 1 = lists, formulae, doctrine; 3 = some narrative or a personal voice;
  5 = vivid events, characters, or a writer speaking about their own life.
- "history": 1 = tells us little about real people or the world; 3 = names real
  people and places; 5 = rare evidence about daily life, events, customs, or a
  place and time that is otherwise poorly documented.
- "neglect": 1 = a famous, much-translated classic (Cicero, Aquinas, Virgil);
  3 = known to specialists; 5 = obscure, probably never translated.
- "garbled": true ONLY if the summary says the text is unreadable -- OCR noise or
  corruption so bad that what it says cannot be made out. A short, fragmentary or
  damaged text whose content the summary does describe is NOT garbled.
- "hook": one plain sentence, under 25 words, naming the concrete thing a reader
  would find in it (who, what happens, what it reveals). No hype: never "glimpse",
  "unique", "lost world", "fascinating", "promises". If nothing stands out, say
  plainly what it is."""


def describe(title: str, author: Optional[str], language: Optional[str],
             century: Optional[int], translation_status: Optional[str] = None) -> str:
    """One line about the work. translation_status is corpus.db's: what the
    translation-audit found out about published translations, if anything."""
    lang = {"la": "Latin", "grc": "Greek"}.get(language or "", language or "")
    extra = ", ".join(b for b in (lang, f"{century}th c." if century else "") if b)
    known = {"translated": "a published translation exists",
             "translated_paywalled": "a published translation exists",
             "untranslated": "no published translation is known"}.get(translation_status or "")
    return (f"{title}{f' by {author}' if author else ''}{f' ({extra})' if extra else ''}"
            f"{f'; {known}' if known else ''}")


def prompt(work: str, summary: str, topics: List[str]) -> str:
    return (f"Work: {work}\nCatalogue summary: {summary}\nTopics: {', '.join(topics)}\n\n"
            f"{RUBRIC}\n\nReturn JSON with subject, story, history, neglect, garbled, hook.")


def score(out: Dict[str, Any]) -> float:
    """Four 1-5 axes -> one 1.0-10.0 score."""
    axes = [min(5, max(1, int(out.get(a) or 1))) for a in AXES]
    s = 1 + 9 * (sum(axes) - len(axes)) / (4 * len(axes))
    return round(min(s, GARBLED_CAP) if out.get("garbled") else s, 1)


def rate(client, model: str, work: str, summary: str, topics: List[str]) -> Dict[str, Any]:
    out = client.chat_json(model, SYSTEM, prompt(work, summary, topics), SCHEMA,
                           num_ctx=4096, num_predict=200)
    axes = {a: min(5, max(1, int(out.get(a) or 1))) for a in AXES}
    hook = re.sub(r"\s+", " ", str(out.get("hook", ""))).strip()
    return {**axes, "garbled": bool(out.get("garbled")), "hook": hook, "score": score(out)}
