"""Find quotations of lost Greek authors in later texts.

The baseline is DFHG (Mueller's Fragmenta Historicorum Graecorum). A witness is
any later text flattened to a token stream with a location per token. Two
independent signals come out of comparing them:

  reuse      a DFHG fragment's wording appears in the witness (`score`)
  citation   the witness names a DFHG author beside a citing formula, and the
             surrounding words match no DFHG fragment (`find_citations`)

Neither proves anything new. DFHG is a 19th-century edition of the historians
only, so a passage missing from it may well be in Jacoby/BNJ: treat every result
as a lead to check by hand. See docs/quotation-mining-pilot.md for measured
recall and for what the false positives look like.
"""
import collections
import json
import pathlib
import re
import unicodedata
import xml.etree.ElementTree as ET

K = 3  # words per shingle; 3 keeps the shuffled-word control at about 0
TEI = "{http://www.tei-c.org/ns/1.0}"
WORD = re.compile(r"[Ͱ-Ͽἀ-῿]+")
# Editorial matter inside a TEI edition, not the witness's own text.
TEI_SKIP = {"note", "bibl", "teiHeader"}

# Mueller's "same source as the previous fragment", which the API returns as is:
# "Idem XII, p. 518", "[Id. v. X", or a bare "XI, p. 483".
SAME_WITNESS = re.compile(r"^[\s\[(]*(?:(?:idem|id\.|ibid|ibidem|eadem|eodem|ead\.)|[IVXL]+\s*,\s*p\.)", re.I)
# Some DFHG witness fields use Greek capitals inside Latin words ("Αthenaeus").
HOMOGLYPHS = str.maketrans("ΑΒΕΖΗΙΚΜΝΟΡΤΥΧ", "ABEZHIKMNOPTYX")

# Formula evidence beside a name.
CITING_VERBS = {"φησι", "φησιν", "ιστορει", "ιστορησε", "ιστορησεν", "γραφει", "αναγραφει", "μεμνηται",
                "ειρηκε", "ειρηκεν", "λεγει", "γεγραφε", "γεγραφεν", "μνημονευει"}
BOOK_ARTICLES = {"τω", "τη", "τοισ", "ταισ"}
# First names of DFHG authors that mostly mean someone else (or a place) in these
# texts. Reading all 135 unmatched Suda and Pindar-scholia citations in the pilot
# found these were nearly all collisions: Aristophanes the playwright for the
# Boeotian historian, Demosthenes the orator, Olympia for Olympus, Patroclus for
# Patrocles. Dionysius and Demetrius lose real historians too, but each covers too
# many people to separate. Pass ambiguous=frozenset() to see them anyway.
AMBIGUOUS_NAMES = frozenset({
    "ALCIMUS", "ANTIOCHUS", "APOLLONIUS", "ARISTAGORAS", "ARISTIDES", "ARISTOCRATES", "ARISTOPHANES", "ARISTOTELES",
    "ARTEMON", "CAPITO", "DEMETRIUS", "DEMOSTHENES", "DINARCHUS", "DIOGENES", "DIONYSIUS", "GORGIAS", "HERACLIDES",
    "MACAREUS", "MELANTHIUS", "NICANOR", "OLYMPUS", "PAMPHILA", "PAMPHILUS", "PATROCLES", "PHILIPPUS", "SIMONIDES",
    "SOCRATES", "THEODORUS", "ZOPYRUS",
})


def norm(token):
    """Accent- and case-free Greek, with lunate and final sigma folded."""
    decomposed = unicodedata.normalize("NFD", token)
    bare = "".join(c for c in decomposed if unicodedata.category(c) != "Mn").lower()
    return bare.replace("ϲ", "σ").replace("ς", "σ").replace("ϐ", "β").replace("ϑ", "θ")


def tokenize(text):
    """[(original, normalized)] for each Greek word in text."""
    return [(m.group(0), norm(m.group(0))) for m in WORD.finditer(text)]


class Witness:
    """A text as parallel lists of original tokens, normalized tokens and locations."""

    def __init__(self, name, parts):
        self.name = name
        self.orig, self.tok, self.loc = [], [], []
        for text, loc in parts:
            for original, normalized in tokenize(text):
                self.orig.append(original)
                self.tok.append(normalized)
                self.loc.append(loc)
        self.index = collections.defaultdict(list)
        for i in range(len(self.tok) - K + 1):
            self.index[tuple(self.tok[i:i + K])].append(i)

    def context(self, start, end):
        return " ".join(self.orig[max(0, start):min(len(self.orig), end)])


def _walk_tei(el, loc, out):
    tag = el.tag.replace(TEI, "")
    if tag in TEI_SKIP:
        return
    if tag == "div" and el.get("type") == "textpart":
        loc = loc + [el.get("n") or "?"]
    if el.text:
        out.append((el.text, ".".join(loc)))
    for child in el:
        _walk_tei(child, loc, out)
        if child.tail:
            out.append((child.tail, ".".join(loc)))


def read_tei(paths):
    """Text parts of one or more TEI editions. Several files get a part prefix: 1, 2, ..."""
    parts = []
    for i, path in enumerate(paths, 1):
        body = ET.parse(path).getroot().find(f"{TEI}text/{TEI}body")
        _walk_tei(body, [str(i)] if len(paths) > 1 else [], parts)
    return parts


def read_pg(path):
    """A Patrologia Graeca OCR volume, located by its $8=page $9=column markers."""
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    parts, loc = [], "?"
    for line in re.sub(r"[-‐]\n", "", text).split("\n"):
        marker = re.match(r"\$0=\S+ \$8=(\S+) \$9=(\S+)", line)
        if marker:
            loc = f"p.{marker.group(1)} col.{marker.group(2)}"
        else:
            parts.append((line, loc))
    return parts


def read_plain(paths):
    """Plain OCR text such as archive.org's _djvu.txt, located by file name."""
    parts = []
    for path in paths:
        path = pathlib.Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        parts.append((re.sub(r"-\s*\n\s*", "", text), path.stem))
    return parts


def resolve_witnesses(records):
    """Give each record its real witness, carrying "Idem ..." forward within an author."""
    last = ""
    for record in records:
        witness = (record.get("witness") or "").strip().translate(HOMOGLYPHS)
        if SAME_WITNESS.match(witness):
            witness = f"{last} | {witness}"
        elif witness:
            last = witness
        record["witness"] = witness
    return records


def load_dfhg(directory):
    """All fragments from a folder of per-author API responses."""
    fragments = []
    for path in sorted(pathlib.Path(directory).glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            continue
        for r in resolve_witnesses(records):
            fragments.append({
                "author": r.get("author"),
                "n": f"{r.get('fragment_number')}{r.get('fragment_letter') or ''}",
                "witness": r["witness"],
                "text": r.get("text") or "",
                "toks": [n for _, n in tokenize(r.get("text") or "")],
                "blob": norm(" ".join(str(r.get(k) or "") for k in ("text", "translation", "commentary", "note", "fragment_note"))),
            })
    return fragments


def score(fragment_tokens, witness, max_frequency=300):
    """(coverage, position): the share of a fragment's words found together in the best passage."""
    hits = []
    for gi in range(len(fragment_tokens) - K + 1):
        positions = witness.index.get(tuple(fragment_tokens[gi:gi + K]))
        if positions and len(positions) <= max_frequency:
            hits.extend((p, gi) for p in positions)
    if not hits:
        return 0.0, None
    hits.sort()
    span = int(len(fragment_tokens) * 1.6) + 10
    best, where, left = 0.0, None, 0
    for right in range(len(hits)):
        while hits[right][0] - hits[left][0] > span:
            left += 1
        covered = set()
        for _, gi in hits[left:right + 1]:
            covered.update(range(gi, gi + K))
        coverage = len(covered) / len(fragment_tokens)
        if coverage > best:
            best, where = coverage, hits[left][0]
    return best, where


def _translit(latin_name):
    s = latin_name.lower().split()[0]
    if s.startswith("h"):  # Latin h- is a breathing in Greek, which norm() removes
        s = s[1:]
    for a, b in (("ae", "e"), ("oe", "i"), ("ph", "f"), ("th", "0"), ("ch", "x"), ("rh", "r"), ("y", "i"), ("c", "k")):
        s = s.replace(a, b)
    return re.sub(r"(.)\1", r"\1", re.sub(r"(us|um|es|as|is|on|a|e|o|ius)$", "", s))


def _skeleton(greek):
    table = str.maketrans({"α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "e", "θ": "0", "ι": "i",
                           "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p", "ρ": "r", "σ": "s",
                           "τ": "t", "υ": "i", "φ": "f", "χ": "x", "ψ": "ps", "ω": "o"})
    s = greek.replace("αι", "ε").replace("ει", "ι").replace("οι", "ι").replace("ου", "u").replace("γγ", "ng").replace("γκ", "nk")
    return re.sub(r"(.)\1", r"\1", s.translate(table))


def author_stems(fragments, ambiguous=AMBIGUOUS_NAMES):
    """{DFHG author: Greek stem}, kept only if the author's own fragments use that name twice or more."""
    by_author = collections.defaultdict(list)
    for fragment in fragments:
        if fragment["author"] and fragment["author"].split()[0] not in ambiguous:
            by_author[fragment["author"]].append(fragment)
    stems = {}
    for author, own in by_author.items():
        target = _translit(author)
        if len(target) < 5:
            continue
        counts = collections.Counter()
        for fragment in own:
            for original, normalized in tokenize(fragment["text"]):
                if original[0].isupper() and _skeleton(normalized).startswith(target):
                    counts[normalized[:-2] if len(normalized) > 6 else normalized] += 1
        if counts and counts.most_common(1)[0][1] >= 2:
            stems[author] = counts.most_common(1)[0][0]
    return stems


def find_citations(witness, stems, fragments, before=15, after=45):
    """Capitalized DFHG names beside a citing formula, with the share of nearby words DFHG also has."""
    baseline = set()
    for fragment in fragments:
        t = fragment["toks"]
        baseline.update(tuple(t[i:i + K]) for i in range(len(t) - K + 1))
    by_prefix = collections.defaultdict(list)
    author_of = {}
    for author, stem in stems.items():
        by_prefix[stem[:3]].append(stem)
        author_of.setdefault(stem, author)
    for prefix in by_prefix:
        by_prefix[prefix].sort(key=len, reverse=True)

    results, seen = [], set()
    for i, (original, normalized) in enumerate(zip(witness.orig, witness.tok)):
        if not original[0].isupper():
            continue
        stem = next((s for s in by_prefix.get(normalized[:3], ()) if normalized.startswith(s)), None)
        if not stem or len(normalized) > len(stem) + 4:
            continue
        low, high = max(0, i - 6), i + 7
        near = witness.tok[low:high]
        formula = "verb" if CITING_VERBS & set(near) else None
        if any(a == "εν" and b in BOOK_ARTICLES for a, b in zip(near, near[1:])):
            formula = "book"
        if not formula or (stem, i // 30) in seen:
            continue
        seen.add((stem, i // 30))
        start, end = max(0, i - before), min(len(witness.tok), i + after)
        window = witness.tok[start:end]
        covered = set()
        for k in range(len(window) - K + 1):
            if tuple(window[k:k + K]) in baseline:
                covered.update(range(k, k + K))
        results.append({"witness": witness.name, "loc": witness.loc[i], "author": author_of[stem], "name": original,
                        "formula": formula, "dfhg_share": round(len(covered) / max(1, len(window)), 2),
                        "context": witness.context(start, end)})
    return results
