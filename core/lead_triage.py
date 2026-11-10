"""A local model's first read of a quotation lead, to make reviewing faster.

The model is asked narrow questions, each with a fixed set of answers, plus a
short English rendering. Its tags only sort and filter the Leads tab: nothing
here decides whether a fragment is new. 12-14B local models read ancient Greek
unevenly, so every answer carries a confidence and "unsure" is always allowed.

Citation leads (a named author beside "says", "in book X"):
  same_person   does the name here mean the DFHG historian, or someone else
                with the name (a playwright, a king, a place)?
  reference     quotation | paraphrase | mention | none
  claim         what the passage says the author said, in one English sentence

Reuse leads (a DFHG fragment's wording found in the text):
  relation      verbatim | close | loose | coincidental
  names_source  who, if anyone, the passage credits
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

SYSTEM = (
    "You are a careful classical philologist helping to find quotations of lost Greek historians in "
    "later texts. You read Byzantine and ancient Greek, including lexicon entries and scholia, and you "
    "know who the famous bearers of common Greek names are. You never invent a source or a reference."
)

# Field order matters: the model writes these in order, so it translates the
# passage and says who the name refers to *before* it commits to yes/no. Asked
# for yes/no first, qwen3:14b answered "unsure" to 98 of 102 citations whose
# answer the pilot review knew.
CITATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"},
        "who_is": {"type": "string"},
        "same_person": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "reference": {"type": "string", "enum": ["quotation", "paraphrase", "mention", "none"]},
        "claim": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["translation", "who_is", "same_person", "reference", "claim", "confidence"],
}

REUSE_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": ["verbatim", "close", "loose", "coincidental"]},
        "names_source": {"type": "string"},
        "translation": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["relation", "names_source", "translation", "confidence"],
}


def author_profile(author: str, fragments: List[Dict[str, Any]], examples: int = 2) -> str:
    """What DFHG knows about an author: work titles and the opening of a fragment or two."""
    own = [f for f in fragments if f["author"] == author]
    works = sorted({f["work"] for f in own if f.get("work")})[:8]
    name, _, epithet = author.title().partition(" ")
    lines = [f"The lost historian: {name}{f' ({epithet})' if epithet else ''}, one of the minor Greek historians "
             f"in Mueller's Fragmenta Historicorum Graecorum ({len(own)} fragments). Other, better-known people "
             f"may share the name {name}."]
    if works:
        lines.append("Works: " + "; ".join(works))
    for f in [f for f in own if len(f["toks"]) >= 12][:examples]:
        lines.append(f"Example fragment (from {f['witness'][:60]}): {f['text'][:220]}")
    return "\n".join(lines)


def citation_prompt(lead: Dict[str, Any], profile: str) -> str:
    return f"""{profile}

Passage from {lead['witness_label']} ({lead['loc']}); the name in question is «{lead['name']}»:
{lead['context']}

Fill in, in this order:
- "translation": a plain English translation of the passage, under 70 words.
- "who_is": who «{lead['name']}» refers to in this passage, as specifically as you can (for example
  "the comic poet Aristophanes, quoting Clouds", "the orator Demosthenes, Against Leptines", "Mount
  Olympus", "Patroclus, Achilles' friend", "the historian of Thebes").
- "same_person": "yes" if that is the lost historian above, "no" if it is someone or something else,
  "unsure" only if the passage gives no clue at all. A passage that quotes a play, a speech, a poem
  or a surviving philosophical work is quoting that famous author, not the lost historian.
- "reference": "quotation" (his words given), "paraphrase" (his content reported), "mention"
  (only named), or "none" (not a reference to an author at all).
- "claim": one English sentence: what the passage says he said. "" if nothing.
- "confidence": 1 (guess) to 5 (certain).
Return JSON."""


def reuse_prompt(lead: Dict[str, Any]) -> str:
    return f"""A fragment of the lost historian {lead['author']} (DFHG fragment {lead['fragment']}),
which Mueller took from: {lead.get('dfhg_source') or 'unknown'}
Fragment: {lead.get('fragment_text') or ''}

Passage from {lead['witness_label']} ({lead['loc']}):
{lead['context']}

Questions:
- "relation": how the passage relates to the fragment: "verbatim", "close" (same content, some
  wording changed), "loose" (same story or fact, different words), or "coincidental".
- "names_source": the author the passage credits for this, as written; "" if none.
- "translation": a plain English translation of the passage, under 70 words.
- "confidence": 1 (guess) to 5 (certain).
Return JSON."""


def _clean(text: Any, limit: int = 600) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def normalize(kind: str, out: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a model's JSON into the expected shape, whatever it returned."""
    conf = out.get("confidence")
    conf = min(5, max(1, int(conf))) if isinstance(conf, (int, float)) or str(conf).isdigit() else 1
    if kind == "citation":
        same = out.get("same_person") if out.get("same_person") in ("yes", "no", "unsure") else "unsure"
        ref = out.get("reference") if out.get("reference") in ("quotation", "paraphrase", "mention", "none") else "none"
        return {"same_person": same, "who_is": _clean(out.get("who_is"), 160),
                "reference": ref, "claim": _clean(out.get("claim"), 300),
                "translation": _clean(out.get("translation")), "confidence": conf}
    rel = out.get("relation") if out.get("relation") in ("verbatim", "close", "loose", "coincidental") else "coincidental"
    return {"relation": rel, "names_source": _clean(out.get("names_source"), 120),
            "translation": _clean(out.get("translation")), "confidence": conf}


def rank(kind: str, tag: Dict[str, Any]) -> float:
    """Higher = more worth your time. Only orders the list; never hides anything."""
    c = tag["confidence"] / 5
    if kind == "citation":
        person = {"yes": 1.0, "unsure": 0.5, "no": 0.02}[tag["same_person"]]
        ref = {"quotation": 1.0, "paraphrase": 0.8, "mention": 0.3, "none": 0.1}[tag["reference"]]
        weight = 0.5 + 0.5 * c if tag["same_person"] == "yes" else 1.0
        return round(person * ref * weight, 3)
    return round({"verbatim": 0.9, "close": 0.8, "loose": 0.5, "coincidental": 0.05}[tag["relation"]], 3)


def tag_lead(client, model: str, lead: Dict[str, Any], profile: str = "") -> Dict[str, Any]:
    if lead["kind"] == "citation":
        out = client.chat_json(model, SYSTEM, citation_prompt(lead, profile), CITATION_SCHEMA,
                               num_ctx=4096, num_predict=400)
    else:
        out = client.chat_json(model, SYSTEM, reuse_prompt(lead), REUSE_SCHEMA, num_ctx=4096, num_predict=400)
    return normalize(lead["kind"], out)
