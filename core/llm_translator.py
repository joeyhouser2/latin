"""LLM translation for old vernacular texts that NLLB cannot read.

NLLB was trained on modern text. Hand it the 12th-century Hungarian funeral
oration, the phonetically spelled Bogurodzica or the Early New High German
*Ackermann* and it produces fluent, confident, wrong English ("It's the head of
a poultryman"), or loops on one phrase. A 12B instruct model run locally through
Ollama has actually *read* a good deal of Old Polish, Middle Dutch and Old East
Slavic; given a short briefing on the stage of the language -- what its spelling
conventions are, which grammar it still has -- it translates these texts
usefully, and it can say when a line is beyond it.

The briefing is the point. Each ``(language, stage)`` gets a paragraph on the
orthography of the manuscripts (so ``vnd`` is read as *und*, ``ſ`` as *s*, Old
Polish ``cz`` as *č*) and on the grammar a modern speaker would trip over
(aorist and imperfect tenses, the dual, verb-final order). The work's own
metadata -- title, author, century, note -- goes in too, so the model knows it is
a funeral oration or a beast epic rather than guessing.

Segments go in small numbered batches with the two lines before as read-only
context (verse and chronicles depend on their neighbours), and the answer is
forced into a JSON schema. If a batch comes back with the wrong number of lines
it is retried one line at a time, so a misaligned answer can never shift every
later translation by one.

    with PrivateOllama() as server:
        tr = LLMTranslator(server.client, "gemma4:12b", language="pl",
                           stage="medieval", title="Bogurodzica", century=13)
        english = tr.translate_batch(lines)

It is a ``Translator``, so it drops in anywhere NLLB does.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from core.local_llm import OllamaClient
from core.models import LANGUAGES
from core.translator import Translator

SYSTEM = """You are a philologist translating {language_name} texts into English for readers who \
know no {language_name}. The source is {stage_desc}. You work from the original, not from a \
modern paraphrase.

{briefing}

Rules:
- Translate faithfully and completely; do not summarise, moralise or add explanation.
- Plain, readable modern English that keeps the register (prayer, epic, sermon, chronicle). \
Keep verse lines as single lines.
- Keep proper names in a recognisable English form (Igor, Charlemagne, Boyan).
- Spelling is irregular and may contain scribal or OCR errors. Read through them to the word \
intended; do not translate the misspelling literally.
- If a word or phrase is genuinely unintelligible, write [?] for it rather than inventing a \
meaning. Never repeat a phrase to fill space.
- Editorial marks such as [VIIv], folio numbers, [!] or line numbers are not text: drop folio \
and page marks, keep nothing of them.
- Output the translation only, as the JSON requested. Lines marked CONTEXT are for your \
understanding only; do not translate them."""

# What a reader needs to know about each stage's spelling and grammar. Keyed
# (language, stage); the ("*") entry for a language is the fallback.
BRIEFINGS: Dict[Tuple[str, str], str] = {
    ("pl", "medieval"): (
        "Old Polish (13th-15th c.). Manuscripts spell phonetically before the standard "
        "orthography: cz = č, sz = š, rz = ž/ř, ch/h, v/u/w interchangeable, a bar or ¦ "
        "marks a nasal vowel, and words may be run together or split. Grammar you must "
        "recognise: aorist and imperfect tenses (rzekł, bieszę), the dual number, "
        "'iże' = that, 'zwolena/zwolony' = chosen, 'Bożycze' = O child of God, 'ma' = "
        "has/my, 'Kyrie eleison' is Greek liturgy and stays as is. The earliest texts "
        "(Bogurodzica, the Holy Cross Sermons) are religious: Marian hymns, sermons "
        "quoting Scripture."),
    ("hu", "medieval"): (
        "Old Hungarian (12th-13th c.), written in a Latin-letter orthography that predates "
        "any standard: ſ = s, ch/c/k variants, u/v interchangeable, z = s or z, 'ae/e' "
        "forms, and no word-spacing conventions. The Funeral Sermon and Prayer (c. 1192) "
        "is a liturgical text about man as dust and ashes, Adam's sin and the hope of "
        "mercy; the Old Hungarian Lament of Mary is a Marian poem. Read the sense from "
        "the surrounding liturgical formulas ('Latiatuc feleym' = Behold, my brothers)."),
    ("ru", "medieval"): (
        "Old East Slavic with Church Slavonic admixture (11th-15th c.). Letters ѣ (yat), "
        "ъ ь (jers), ѳ, ꙋ, ѡ, titlos (abbreviation marks) and superscript letters; "
        "'и' = and, 'бяшетъ' = was, aorists and imperfects (рече, иде, бяху), the dual, "
        "long pronoun forms, reflexive 'ся', and word order that places the verb early. "
        "Chronicle and epic idiom: 'князь' prince, 'дружина' retinue, 'поганый' pagan, "
        "'плъкъ' host/army, 'Половци' the Cumans. For the Slovo o polku Igoreve, proper "
        "names (Boyan, Vsevolod, Yaroslavna, Donets) and the rhythmic, image-laden style "
        "are intended, not errors."),
    ("ru", "early_modern"): (
        "Early Muscovite Russian (15th-17th c.), still with Church Slavonic forms, "
        "pre-reform spelling and the old past tenses. Modern editorial commentary may be "
        "mixed into the text: translate it plainly, as modern Russian."),
    ("de", "medieval"): (
        "Late medieval / Early New High German (14th-15th c.). Orthography: v/u "
        "interchangeable ('vnd' = und, 'vch' = euch, 'vber' = über), ſ = s, 'ey' = ei, "
        "'w' for 'b' or 'u' in places, doubled consonants, no umlaut marks, and the "
        "lowercase noun. Vocabulary shifts: 'wol' = well/indeed, 'iht' = anything, "
        "'nicht/niht', 'tugent' = virtue, 'mynne' = courtly love. The Ackermann aus Böhmen "
        "is a prose dispute between a bereaved ploughman and Death, whose rhetoric "
        "is learned and legalistic; translate it as a debate."),
    ("nl", "medieval"): (
        "Middle Dutch (13th-15th c.), as in the Hulthem manuscript's abele spelen, "
        "Reynaert, Karel ende Elegast. Spelling varies freely: ghi/gi = you, 'ende' = and, "
        "'si' = she/they, 'hi' = he, 'sijn' = his/to be, 'doen' = to do, past forms in "
        "-de/-te, Old verb endings -en/-et, and many words that survive in changed "
        "forms ('vrouwe' = lady, 'ridder' = knight, 'cleine' = small). Verse lines rhyme "
        "in couplets; keep each line its own."),
    ("it", "medieval"): (
        "Old Tuscan Italian (13th-14th c.): 'e' for modern 'è' or 'ed', 'ch/gh' before "
        "e/i, 'anzi' = rather/before, 'cavaliere', 'ser', 'messer' = sir, double "
        "consonants in different places, pro-drop and clitic placement that follows the "
        "verb, and the Novellino/Sacchetti tale idiom of short anecdotes closing on a "
        "witty reply."),
    ("it", "early_modern"): (
        "Renaissance Italian (15th-16th c.) chivalric verse (Pulci, Boiardo) in ottava "
        "rima: eight-line stanzas. Keep every line, name the paladins plainly "
        "(Orlando, Rinaldo, Morgante), and keep the comic register of Pulci."),
    ("fr", "medieval"): (
        "Old/Middle French or a modern-French rendering of it. Keep the tone of the "
        "beast epic: Renart the fox, Isengrin the wolf, Noble the lion."),
    ("*", "*"): (
        "A medieval or early-modern vernacular whose spelling is not standardised. Read "
        "through variant spellings and archaic grammar to the intended sense."),
}

_STAGE_DESC = {
    "medieval": "a medieval manuscript text",
    "early_modern": "an early-modern (16th-17th century) text",
    "archaic": "an archaic-period text",
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}, "english": {"type": "string"},
                               "confidence": {"type": "string",
                                              "enum": ["high", "medium", "low"]}},
                "required": ["n", "english", "confidence"],
            },
        }
    },
    "required": ["translations"],
}

# Appended to a line the model says it is largely guessing at, so a reader (and a
# later pass) can tell a reconstruction from a reading.
UNCERTAIN = " [uncertain]"

_FOLIO = re.compile(r"\[[IVXLC]*\d*[rv]\]|\[[ivxlc]+[rv]?\]")


def briefing_for(language: str, stage: str) -> str:
    for key in ((language, stage), (language, "*"), ("*", "*")):
        if key in BRIEFINGS:
            return BRIEFINGS[key]
    return BRIEFINGS[("*", "*")]


class LLMTranslator(Translator):
    """A stage-aware translator backed by a local Ollama chat model."""

    def __init__(self, client: OllamaClient, model: str, language: str, stage: str = "medieval",
                 title: str = "", author: Optional[str] = None, century: Optional[int] = None,
                 note: Optional[str] = None, genre: Optional[str] = None,
                 batch: int = 8, context: int = 2, num_ctx: int = 8192):
        self.client = client
        self.model = model
        self.language = language
        self.stage = stage
        self.batch = batch
        self.context = context
        self.num_ctx = num_ctx
        self.meta = {"title": title, "author": author, "century": century,
                     "genre": genre, "note": note}
        self.system = SYSTEM.format(
            language_name=LANGUAGES.get(language, language),
            stage_desc=_STAGE_DESC.get(stage, "a pre-modern text"),
            briefing=briefing_for(language, stage))
        self.failures = 0     # lines that fell back to the unreadable marker
        self.uncertain = 0    # lines the model flagged low-confidence

    # -- Translator ----------------------------------------------------------

    def translate(self, text: str) -> str:
        return self.translate_batch([text])[0]

    def translate_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[str]:
        size = self.batch if batch_size is None else min(batch_size, self.batch)
        out: List[str] = []
        for i in range(0, len(texts), size):
            chunk = texts[i:i + size]
            before = texts[max(0, i - self.context):i]
            out.extend(self._translate_chunk(chunk, before))
        return out

    # -- internals -----------------------------------------------------------

    def _user(self, chunk: List[str], before: List[str]) -> str:
        m = self.meta
        head = [f"Work: {m['title']}" + (f" by {m['author']}" if m["author"] else "")
                + (f" ({m['century']}th century)" if m["century"] else "")]
        if m["genre"]:
            head.append(f"Genre: {m['genre']}")
        if m["note"]:
            head.append(f"About this work: {m['note']}")
        lines = list(head)
        if before:
            lines.append("")
            lines += [f"CONTEXT: {_FOLIO.sub('', t).strip()}" for t in before]
        lines.append("")
        lines.append("Translate each numbered line. Return one entry per line, with the same n.")
        lines += [f"{n}. {_FOLIO.sub('', t).strip()}" for n, t in enumerate(chunk, 1)]
        return "\n".join(lines)

    def _ask(self, chunk: List[str], before: List[str]) -> Optional[List[str]]:
        n = len(chunk)
        data = self.client.chat_json(
            self.model, self.system, self._user(chunk, before), _SCHEMA,
            num_ctx=self.num_ctx, temperature=0.2,
            num_predict=max(300, 220 * n), retries=1)
        got = {}
        for e in data.get("translations", []):
            if not isinstance(e.get("n"), int):
                continue
            text = (e.get("english") or "").strip()
            if text and e.get("confidence") == "low":
                text += UNCERTAIN
                self.uncertain += 1
            got[e["n"]] = text
        if set(got) != set(range(1, n + 1)):
            return None
        return [got[k] for k in range(1, n + 1)]

    def _translate_chunk(self, chunk: List[str], before: List[str]) -> List[str]:
        try:
            res = self._ask(chunk, before)
        except Exception:
            res = None
        if res is not None:
            return res
        # Misaligned or failed batch: one line at a time, so nothing can shift.
        out: List[str] = []
        ctx = list(before)
        for text in chunk:
            try:
                one = self._ask([text], ctx[-self.context:])
            except Exception:
                one = None
            if one is None or not one[0]:
                self.failures += 1
                out.append("[?]")
            else:
                out.append(one[0])
            ctx.append(text)
        return out
