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
    ("la", "early_modern"): (
        "Neo-Latin hexameter epic of the 15th-17th century, in imitation of Virgil, Statius and "
        "Lucan (for example Basinio da Parma's Hesperis). The Latin is classical in grammar but "
        "full of Virgilian echoes, epithets and periphrases: 'Tyrrhenum' = the Tyrrhenian sea or "
        "coast, 'Martes' = war, 'Latina gens' = the Italians, 'Iberi' = Spaniards or Aragonese, "
        "'Musa' = the Muse (a goddess, never Moses), 'Genitor' = the Father of the gods. Sentences "
        "run on across line breaks, so read each line with its neighbours, but give each Latin "
        "line its own English line and do not move words from one line to another. Keep names "
        "of people and places in a recognisable English form; if a name is unfamiliar, keep the "
        "Latin spelling rather than substituting a similar-sounding one."),
    ("la", "papyrus"): (
        "Latin documents written on papyrus, wooden tablets and potsherds (1st-7th c. AD): "
        "letters, receipts, army rosters and reports, contracts, deeds, court records. They "
        "are printed in a modern edition with the Leiden conventions, and here square "
        "brackets are NOT folio marks: [abc] = letters lost and restored by the editor "
        "(translate them; they are the best reading); [...] or [---] = text lost, and "
        "'...' inside a word = illegible letters; for both, write … in the English at that "
        "point and never guess what was there; (abc) = an abbreviation the editor expanded "
        "(read the full word); ⟦abc⟧ = deleted by the scribe (leave out); <abc> = letters "
        "the scribe left out. Spelling is often sub-classical: u for v, e for ae, i/e and "
        "o/u confused, b for v, final -m dropped. Formulae: 'salutem' = greetings, "
        "'uale/vale' = farewell, 'frater/soror' are affectionate address, not always kin; "
        "dates by consuls, emperors' regnal years and indictions, and by Kalends/Nones/Ides "
        "(iii Idus Septembres = 11 September); 'n(umero)' = in number, 'd(enarios)' = "
        "denarii, 'v(ir) c(larissimus)' and similar are titles. A line that is mostly gaps "
        "may come out as a few words and …; that is correct."),
    ("grc", "papyrus"): (
        "Greek passages inside Latin documents on papyrus, tablets and potsherds (1st-7th c. "
        "AD), mostly from Roman and Byzantine Egypt and 6th-century Ravenna: Greek "
        "subscriptions and signatures under Latin deeds and receipts, Greek letters and "
        "accounts, dockets, and bilingual school texts. They are printed in a modern edition "
        "with the Leiden conventions, and here square brackets are NOT folio marks: [abc] = "
        "letters lost and restored by the editor (translate them; they are the best reading); "
        "[...] or [---] = text lost, and '...' inside a word = illegible letters; for both, "
        "write … in the English at that point and never guess what was there; (abc) = an "
        "abbreviation the editor expanded (read the full word); ⟦abc⟧ = deleted by the "
        "scribe (leave out); <abc> = letters the scribe left out. Three kinds of text occur. "
        "(1) Koine Greek, spelled as it was pronounced: ι/ει/η/οι/υ interchange (itacism), "
        "αι for ε, ο for ω, iota adscript dropped or misplaced, γ before ι dropped. Formulae: "
        "'χαίρειν' = greetings, 'ἔρρωσο' / 'ἐρρῶσθαί σε εὔχομαι' = farewell / I pray for "
        "your health, 'ἄδελφε', 'κύριέ μου' are courteous address, not always kin; 'ἔγραψα "
        "ὑπὲρ αὐτοῦ μὴ εἰδότος γράμματα' = I wrote for him, since he does not know letters; "
        "'σεσημείωμαι' = I have signed; 'ἀπέχω' = I have received; 'ὁμολογῶ' = I acknowledge. "
        "Dates by regnal year of the emperor ((ἔτους) or L = year) and Egyptian months "
        "(Thoth, Phaophi, Hathyr, Choiak, Tybi, Mecheir, Phamenoth, Pharmouthi, Pachon, Payni, "
        "Epeiph, Mesore) or indiction. Money and measures: drachmas, obols, talents, artabas, "
        "arouras, 'νομισμάτιον' = solidus. Latin army and office words appear in Greek "
        "letters (κεντυρίων centurion, λεγεών legion, οὐετρανός veteran, πραιπόσιτος "
        "praepositus). (2) Latin written in Greek letters, as in the Ravenna witnesses' "
        "subscriptions: 'σολιδος' = solidos, 'τραδιτος βιδι' = traditos vidi (I saw them "
        "handed over), 'πρεσεντι' = praesenti, 'β(ιρ) κ(λαρισσιμος)' = v(ir) c(larissimus). "
        "Read these as Latin and translate the Latin. (3) Bilingual school texts, where a "
        "Latin word or phrase is followed by its Greek equivalent (a word-for-word glossed "
        "Virgil or Cicero): translate the passage once, not each word twice. Latin words "
        "mixed into a line are part of it; translate them too. Translate only what is "
        "written: a line of names stays a line of names, and you never add a greeting or "
        "phrase that is not in the Greek. A line that is mostly gaps may come out as a few "
        "words and …; that is correct."),
    ("*", "*"): (
        "A medieval or early-modern vernacular whose spelling is not standardised. Read "
        "through variant spellings and archaic grammar to the intended sense."),
}

_STAGE_DESC = {
    "medieval": "a medieval manuscript text",
    "early_modern": "an early-modern (16th-17th century) text",
    "archaic": "an archaic-period text",
    "papyrus": "a document on papyrus, a tablet or an ostracon, in a modern edition",
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

_MODE_SWITCH = re.compile(r"\s*/(?:no_)?think\b")
_FOLIO = re.compile(r"\[[IVXLC]*\d*[rv]\]|\[[ivxlc]+[rv]?\]")

# What a papyrus line too broken to translate becomes. Without this the model
# hands the Latin back as its "English" (seen on the Bu Njem ostraca).
FRAGMENT = "[too fragmentary to translate]"
_GAP = re.compile(r"\[\.\.\.\]|\[---\]|\.\.\.|…")
_TOKEN = re.compile(r"[^\W\d_]{3,}")


def _echoes_fragment(source: str, english: str) -> bool:
    """True when the "translation" mostly repeats a gap-riddled source.

    Both conditions are needed: a roster line of names legitimately comes back
    nearly unchanged, but it is not full of gaps."""
    src = {w.lower() for w in _TOKEN.findall(source)}
    out = [w.lower() for w in _TOKEN.findall(english)]
    if len(src) < 3 or not out or len(_GAP.findall(source)) < 4:
        return False
    return sum(w in src for w in out) / len(out) >= 0.5


def _copies_source(source: str, english: str) -> bool:
    """True when a readable line came back as the Latin itself.

    Name rosters legitimately survive translation almost unchanged, so the
    caller retries once with a nudge and keeps the second answer either way."""
    src = {w.lower() for w in _TOKEN.findall(source)}
    out = [w.lower() for w in _TOKEN.findall(english)]
    if len(src) < 4 or not out:
        return False
    return sum(w in src for w in out) / len(out) >= 0.6

NUDGE = ("Your previous answer repeated the {language_name}. Give an English translation; "
         "keep only personal and place names as they are.")


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
        # In a Leiden edition "[c]" or "[ii]" is a restored letter, not a folio mark.
        self.strip_folios = stage != "papyrus"
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
        clean = (lambda t: _FOLIO.sub("", t).strip()) if self.strip_folios else str.strip
        if before:
            lines.append("")
            lines += [f"CONTEXT: {clean(t)}" for t in before]
        lines.append("")
        lines.append("Translate each numbered line. Return one entry per line, with the same n.")
        lines += [f"{n}. {clean(t)}" for n, t in enumerate(chunk, 1)]
        return "\n".join(lines)

    def _ask(self, chunk: List[str], before: List[str], nudge: bool = False) -> Optional[List[str]]:
        n = len(chunk)
        nudge_text = NUDGE.format(language_name=LANGUAGES.get(self.language, self.language))
        user = self._user(chunk, before) + ("\n\n" + nudge_text if nudge else "")
        data = self.client.chat_json(
            self.model, self.system, user, _SCHEMA,
            num_ctx=self.num_ctx, temperature=0.2,
            num_predict=max(300, 220 * n), retries=1)
        entries = [e for e in data.get("translations", []) if isinstance(e, dict)]
        if n == 1 and not before and len(entries) > 1:
            # Asked for one long line, the model sometimes answers in several numbered
            # pieces (a sentence each). With no context lines to confuse them with,
            # they can only be parts of that line: join them in order.
            parts = sorted((e for e in entries if isinstance(e.get("n"), int)), key=lambda e: e["n"])
            conf = [e.get("confidence") for e in parts]
            entries = [{"n": 1,
                        "english": " ".join((e.get("english") or "").strip() for e in parts).strip(),
                        "confidence": "low" if "low" in conf else "medium" if "medium" in conf else "high"}]
        got = {}
        for e in entries:
            if not isinstance(e.get("n"), int):
                continue
            # qwen3 sometimes echoes its own "/no_think" mode switch into the answer.
            text = _MODE_SWITCH.sub("", e.get("english") or "").strip()
            if not text:
                continue          # a blank answer counts as missing: retry line by line
            if self.stage == "papyrus" and 1 <= e["n"] <= n and _echoes_fragment(chunk[e["n"] - 1], text):
                text = FRAGMENT
            elif self.stage == "papyrus" and 1 <= e["n"] <= n and not nudge \
                    and _copies_source(chunk[e["n"] - 1], text):
                try:
                    again = self._ask([chunk[e["n"] - 1]], before, nudge=True)
                except Exception:
                    again = None
                if again and again[0]:
                    got[e["n"]] = again[0]
                    continue
            if text and text != FRAGMENT and e.get("confidence") == "low":
                text += UNCERTAIN
                self.uncertain += 1
            got[e["n"]] = text
        if set(got) != set(range(1, n + 1)):
            return None
        # A batch can come back with the right number of entries that are one merged
        # paragraph repeated (seen on the Hesperis opening, 8 lines -> 1 sentence x 8).
        # The count matches, so only this catches it; retry line by line instead.
        if n >= 3 and len(set(got.values())) <= n // 2:
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
