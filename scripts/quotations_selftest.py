"""Self-test for ingest/quotations.py. Needs no network or downloaded data.

    python scripts/quotations_selftest.py
"""
import os
import pathlib
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import lead_triage
from core.leads import LeadStore
from ingest.quotations import WITNESSES, Witness, read_plain, author_stems, find_citations, load_dfhg, norm, read_pg, read_tei, resolve_witnesses, score

PASSAGE = "πρώτην ἐπιστολὰς συντάξαι Ἄτοσσαν τὴν Περσῶν βασιλεύσασαν φησὶν Ἑλλάνικος ἐν τοῖς Περσικοῖς καὶ ἄλλα πολλὰ ἱστορεῖ"
TEI_DOC = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div type="edition">
<div type="textpart" n="2"><p>λόγος πρῶτος<note>σχόλιον ἐκδότου</note> τέλος</p></div>
</div></body></text></TEI>"""


def fragment(author, text):
    return {"author": author, "n": "1", "witness": "", "text": text, "toks": [norm(w) for w in text.split()], "blob": norm(text)}


def main():
    assert norm("Ἑλλάνικοϲ") == norm("ἑλλάνικος") == "ελλανικοσ"

    records = resolve_witnesses([{"witness": "Athen. II, p. 1"}, {"witness": "Idem XII, p. 2"}, {"witness": "[Id. v. X"},
                                 {"witness": "XI, p. 3:"}, {"witness": "Αthenaeus I"}, {"witness": ""}])
    assert all(r["witness"].startswith("Athen") for r in records[:4]), records
    assert records[4]["witness"] == "Athenaeus I" and records[5]["witness"] == ""
    # DFHG's typos and other "same as above" forms still count as Athenaeus.
    chained = resolve_witnesses([{"witness": "Athenoeus XIV, p. 638"}, {"witness": "— Ibid.:"},
                                 {"witness": "Adde XV, p. 677"}, {"witness": "Atehn. IV, p. 140"},
                                 {"witness": "XLIV, 41:"}])
    assert all(re.search(WITNESSES["athenaeus"][3], r["witness"]) for r in chained), chained
    for other in ("Athenagoras in Deprecat.", "Diogen. Laert. IV, 13, de Cratete Atheniensi"):
        assert not re.search(WITNESSES["athenaeus"][3], other), other

    witness = Witness("test", [("ἀρχή τις " + PASSAGE + " τέλος", "1.1")])
    tokens = [norm(w) for w in PASSAGE.split()]
    coverage, where = score(tokens, witness)
    assert coverage == 1.0 and where == 2, (coverage, where)
    assert score(tokens[::-1], witness)[0] == 0.0
    assert score(["αβγ", "δεζ", "ηθι", "κλμ"], witness) == (0.0, None)

    with tempfile.TemporaryDirectory() as tmp:
        tei = pathlib.Path(tmp) / "a.xml"
        tei.write_text(TEI_DOC, encoding="utf-8")
        words = Witness("tei", read_tei([tei]))
        assert words.orig == ["λόγος", "πρῶτος", "τέλος"] and set(words.loc) == {"2"}, words.orig
        pg = pathlib.Path(tmp) / "b.txt"
        pg.write_text("$0=87_1 $8=18 $9=1\nλόγ-\nος πρῶτος\n", encoding="utf-8")
        words = Witness("pg", read_pg(pg))
        assert words.tok == ["λογοσ", "πρωτοσ"] and words.loc[0] == "p.18 col.1", words.tok
        ocr = pathlib.Path(tmp) / "c_djvu.txt"
        ocr.write_text("ΤΙΤΛΟΣ\nλόγ-\nος πρῶτος\nτέλος", encoding="utf-8")
        words = Witness("plain", read_plain([ocr]))
        assert words.tok == ["τιτλοσ", "λογοσ", "πρωτοσ", "τελοσ"], words.tok
        assert words.loc == ["c_djvu line 1", "c_djvu line 2", "c_djvu line 2", "c_djvu line 3"], words.loc
        assert resolve_witnesses([{"witness": "Suid."}, {"witness": "Idemque"}])[1]["witness"] == "Idemque"
        api = pathlib.Path(tmp) / "x.json"
        api.write_text('[{"author": "X", "fragment_number": "1", "witness": "Suid.", "text": "λόγος"}]', encoding="utf-8")
        assert load_dfhg(tmp)[0]["witness"] == "Suid."

    own = [fragment("HELLANICUS", "τάδε φησὶν Ἑλλάνικος ἐν βιβλίῳ"), fragment("HELLANICUS", "ὡς Ἑλλάνικος ἱστορεῖ περὶ τούτων")]
    stems = author_stems(own, frozenset())
    assert stems == {"HELLANICUS": "ελλανικ"}, stems
    assert author_stems([fragment("HELLANICUS LESBIUS", f["text"]) for f in own], frozenset({"HELLANICUS"})) == {}

    unmatched = find_citations(witness, stems, own)
    assert len(unmatched) == 1 and unmatched[0]["name"] == "Ἑλλάνικος" and unmatched[0]["formula"] == "book", unmatched
    assert unmatched[0]["dfhg_share"] < .3, unmatched
    collected = find_citations(witness, stems, own + [fragment("HELLANICUS", PASSAGE)])
    assert collected[0]["dfhg_share"] > .5, collected
    lowercase = Witness("t", [("φησὶν ἑλλάνικος μόνον", "1")])
    assert find_citations(lowercase, stems, own) == []
    assert unmatched[0]["pos"] == 10   # word index of the name in the witness

    # Lead store: re-mining keeps the model's tag and your verdict.
    with tempfile.TemporaryDirectory() as tmp:
        store = LeadStore(os.path.join(tmp, "leads.db"))
        lead = dict(unmatched[0], kind="citation", score=0.0)
        reuse = {"kind": "reuse", "witness": "suda", "loc": "1.A.1", "author": "X", "fragment": "3", "score": .9,
                 "context": "λόγος", "fragment_text": "λόγος", "dfhg_source": "Harpocr."}
        assert store.upsert([lead, reuse]) == 2
        first = store.search(sort="recent")["items"][-1]
        tag = lead_triage.normalize("citation", {"same_person": "yes", "reference": "quotation", "confidence": 4,
                                                 "claim": " he  says ", "translation": "x", "who_is": " the  historian "})
        assert tag["claim"] == "he says" and tag["who_is"] == "the historian"
        store.set_tag(first["id"], tag, "m", lead_triage.rank("citation", tag))
        store.set_verdict(first["id"], "new", "BNJ?")
        store.upsert([dict(lead, context="changed")])
        again = store.get(first["id"])
        assert again["tag"]["same_person"] == "yes" and again["verdict"] == "new" and again["context"] == "changed"
        assert store.search(tag="same_person:yes")["total"] == 1 and store.search(verdict="none")["total"] == 1
        assert store.search(tag="untagged")["items"][0]["kind"] == "reuse"
        assert len(store.to_tag()) == 1 and len(store.to_tag(force=True)) == 2
        assert store.stats()["by_verdict"] == {"new": 1, "none": 1}
        # A rerun that no longer finds them drops the unreviewed lead and keeps the reviewed one, marked stale.
        assert store.replace_witness("suda", []) == {"stored": 0, "dropped": 1, "kept_reviewed": 0}
        assert store.replace_witness("test", []) == {"stored": 0, "dropped": 0, "kept_reviewed": 1}
        assert store.get(first["id"])["stale"] == 1 and store.stats()["total"] == 1
        # "Most promising" puts citations with a high Mueller-only prior first, whatever the model said.
        from ingest.quotations import author_priors
        known = [fragment("PHERECYDES", "λόγος"), fragment("PHERECYDES", "λόγος")]
        for f in known:
            f["witness"] = "Schol. Hom. Il. A, 1"
        cites = [dict(lead, author="PHERECYDES", pos=1), dict(lead, author="ZENODOTUS", pos=2),
                 dict(lead, author="ZENODOTUS", pos=3)]
        author_priors(cites, known, WITNESSES["homer_scholia"][3])
        assert cites[0]["credited"] == 2 and cites[0]["prior"] > cites[1]["prior"], cites
        store.upsert([dict(c, witness="homer_scholia") for c in cites])
        promising = [r["author"] for r in store.search(witness="homer_scholia")["items"]]
        assert promising[0] == "PHERECYDES", promising
        try:
            store.set_verdict(first["id"], "maybe")
            raise AssertionError("bad verdict accepted")
        except ValueError:
            pass
        store.close()

    # The model's answers are coerced, and ranking puts collisions last.
    junk = lead_triage.normalize("citation", {"same_person": "perhaps", "confidence": "high"})
    assert junk["same_person"] == "unsure" and junk["reference"] == "none" and junk["confidence"] == 1
    no = lead_triage.normalize("citation", {"same_person": "no", "reference": "quotation", "confidence": 5})
    assert lead_triage.rank("citation", tag) > lead_triage.rank("citation", junk) > lead_triage.rank("citation", no)
    assert lead_triage.normalize("reuse", {"relation": "close", "confidence": 3})["relation"] == "close"

    # Job queue wiring.
    from web.jobs import JOB_KINDS, describe
    argv = JOB_KINDS["mine_quotes"].build({"witnesses": ["suda"], "include_ambiguous": True})
    assert argv[-4:] == ["run", "--witness", "suda", "--include-ambiguous"], argv
    assert not JOB_KINDS["mine_quotes"].uses_gpu and JOB_KINDS["triage_leads"].uses_gpu
    assert not JOB_KINDS["triage_leads"].writes_corpus
    try:
        JOB_KINDS["mine_quotes"].build({"witnesses": ["nonsense"]})
        raise AssertionError("unknown text accepted")
    except ValueError:
        pass
    assert describe("triage_leads", {"model": "qwen3:14b"}) == "Sort leads: untagged leads (qwen3:14b)"
    assert set(WITNESSES) >= {"athenaeus", "suda", "pindar_scholia", "homer_scholia"}
    scholia_checks()
    known_loci_checks()
    print("quotations self-test passed")


JACOBY_PAGE = """60 3. PHEREKYDES VON ATHEN
3 a) SCHOL. PINDAR. P III 59: örı &v Auxeoelaı &ıreı Kogwris,
Osooaktrois (III): Borßıas. b) SCHOL. EURIP. Alkest. 1: Ava&ardoiöng ypyow (F 35a)
4 (8) ATHEN. XII 59 p. 542 A: Yulmvöog 0’ ö Kahlarıavös
5 STEPH. BYZ. s. ‘Yoia' sölıg Bowriag
7 Meineke 21—22 Ö& hinter äöpaoav P, hinter maoddofov cett
12 Ö& Taurıwv eioiv
2. LÄNDER UND VÖLKER
13 — XIII 60,5: Aaumows de adrov"""


# Part III: dates after the name, running heads over several authors, Greek book titles OCR'd as
# Latin capitals, source lines numbered with a dot, Greek section letters and apparatus lines.
JACOBY_III_PAGE = """298. AUTESION s.III/IIs ?
1 (IV 346, 1) ATHEN. 11,2 p. 460 D: τὸ δὲ ποτήριον
299. PYTHAINETOS -- 300. THEOGENES 3
2 (2) ATHEN. 14, 79 p. 660 ΑΒ: καὶ
1. EYBOIKA
3 (-) SUDA s. V. Τρικέφαλος * Ἑρμῆς
107. STEPH. BYZ. s. Ὀρχομενός πόλις
7 Athen. 1, 5 p. 4 A 6 αὐτὸς Sintenis αὐτοὺς O
8 ATHEN. 7 Athen. 1, 5 p. 4 A 6 αὐτὸς
304. DEMETRIOS (DAMEN) <VON ARGOS> 5. IVa
1 (-) ATHEN. 4, 31 p. 148 F—149 C: λέγει"""


SCHOLIA_DOC = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div type="edition">
<div type="textpart" subtype="book" n="5"><div type="textpart" subtype="section" n="1">
<p>1. Τιθωνοῖο] Τιθωνὸς Λαομέδοντος παῖς</p><p>ἄλλως ἠὼς ἡ πρωΐα</p></div>
<div type="textpart" subtype="section" n="2"><p>ἔτι λόγος</p><p>7. ἐκ τῆς Ἰλιάδος</p></div></div>
</div></body></text></TEI>"""


def scholia_checks():
    from ingest import quotations
    from ingest.known_loci import witness_locus
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "od.xml"
        path.write_text(SCHOLIA_DOC, encoding="utf-8")
        parts = read_tei([path], verses=True)
        # The lemma's verse carries over a section break, until the next lemma.
        assert [loc for text, loc in parts if text.strip()] == ["5.1 v.1", "5.1 v.1", "5.2 v.1", "5.2 v.7"], parts
        assert [loc for text, loc in read_tei([path]) if text.strip()][0] == "5.1"
    homer = Witness("homer_scholia", [("ἡ ἱστορία παρὰ Φερεκύδει", "2.1.5.3 v.257"), ("λόγος", "1.2.20.96 v.307")])
    assert witness_locus(homer, 0) == "Od.5.257" and witness_locus(homer, 4) == "Il.20.307"
    soph = Witness("sophocles_scholia", [("λόγος", "7.1053")])
    assert witness_locus(soph, 0) == "OC.1053"
    # Short names and Latin -er names get a stem from the author's own fragments.
    assert quotations._translit("ISTER") == "istr" and quotations._translit("EPHORUS") == "efor"
    own = [fragment("EPHORUS", "ὡς Ἔφορος ἱστορεῖ"), fragment("EPHORUS", "φησὶν Ἔφορος ἐν τῇ")]
    assert author_stems(own, frozenset()) == {"EPHORUS": "εφοροσ"}
    # "ἡ ἱστορία παρὰ Φερεκύδει" is a citation; wording is checked against the cited author's own fragments.
    pher = [fragment("PHERECYDES", "Φερεκύδης φησὶν ἐν τῇ"), fragment("PHERECYDES", "ὡς Φερεκύδης λέγει")]
    other = fragment("HELLANICUS", "ὁ Τιθωνὸς Λαομέδοντος παῖς ἦν")
    text = Witness("t", [("ὁ Τιθωνὸς Λαομέδοντος παῖς ἦν ἡ ἱστορία παρὰ Φερεκύδει", "1")])
    found = find_citations(text, author_stems(pher, frozenset()), pher + [other])
    assert len(found) == 1 and found[0]["formula"] == "story", found
    assert found[0]["dfhg_share"] > .3 and found[0]["author_share"] == 0, found
    # Names of the commented work's characters, in any inflection.
    characters = {"μενελαοσ", "μενελαου", "ατρειδησ"}
    assert quotations.is_character("Μενελάου", "μενελα", characters)
    assert not quotations.is_character("Φερεκύδης", "φερεκυδ", characters)


def known_loci_checks():
    from ingest.known_loci import from_dfhg, from_jacoby, near, parse_locus
    got = [(r.author, r.fragment, r.witness, r.locus) for r in from_jacoby(JACOBY_III_PAGE, "test")]
    assert got == [("298 AUTESION", "1", "athenaeus", "460d"), ("298 AUTESION", "2", "athenaeus", "660a"),
                   ("298 AUTESION", "3", "suda", "τρικεφαλοσ"), ("298 AUTESION", "8", "athenaeus", None),
                   ("304 DEMETRIOS (DAMEN) <VON ARGOS>", "1", "athenaeus", "148f")], got
    rows = from_jacoby(JACOBY_PAGE, "test")
    got = [(r.author, r.fragment, r.witness, r.locus) for r in rows]
    assert got == [("3 PHEREKYDES VON ATHEN", "3a", "pindar_scholia", "P.3.59"),
                   ("3 PHEREKYDES VON ATHEN", "3b", "euripides_scholia", "Alc.1"),
                   ("3 PHEREKYDES VON ATHEN", "4", "athenaeus", "542a"),
                   ("3 PHEREKYDES VON ATHEN", "5", "stephanus", None),
                   ("3 PHEREKYDES VON ATHEN", "13", "stephanus", None)], got
    assert rows[2].concordance == "8"
    for raw, want in (("ATHENAI. II 70 A", "70a"), ("Athenaeus XIV, 619, D", "619d"), ("ATHENAI. II 35 AB", "35a")):
        assert parse_locus("athenaeus", raw) == want, raw
    for raw, want in (("SCHOL. PINDAR. J IV 104", "I.4.104"), ("Schol. Pindar. Isthm. III (IV), 104:", "I.3.104"),
                      ("Schol. Pindari Olymp. XIII, vs. 32:", "O.13.32")):
        assert parse_locus("pindar_scholia", raw) == want, raw
    assert parse_locus("suda", "Suidas v. Ὅμηρος:") == "ομηροσ"
    for witness, raw, want in (("euripides_scholia", "Schol. Eurip. Orest. 1645:", "Or.1645"),
                               ("euripides_scholia", "SCHOL. EURIP, Hek. 123", "Hec.123"),
                               ("euripides_scholia", "Schol. Eur. Phoen. 670:", "Ph.670"),
                               ("sophocles_scholia", "ScHoL. SOPHORL. O. K. 1053", "OC.1053"),
                               ("sophocles_scholia", "Schol. Soph. Oed. Col. v. 489, ἄπυστα:", "OC.489"),
                               ("sophocles_scholia", "Schol. Soph. OEd. Tyr. 21:", "OT.21"),
                               ("sophocles_scholia", "Schol. Sophocl. Trachin. 253:", "Tr.253"),
                               ("homer_scholia", "Schol. Hom. Il. XIV, 119:", "Il.14.119"),
                               ("homer_scholia", "Schol. Homer. Il. Λ, 674:", "Il.11.674"),
                               ("homer_scholia", "Schol. Homer. Odyss. O, 223:", "Od.15.223"),
                               ("homer_scholia", "SCHOL. A HOMER. I!. T 116", "Il.19.116")):
        assert parse_locus(witness, raw) == want, (raw, parse_locus(witness, raw))
    # Jacoby's Homer books are unreadable OCR: any book, same poem and line.
    jacoby_homer = from_jacoby("12. HELLANIKOS\n7 (5) SCHOL. A HOM. Il. T 116: ὡς\n", "test")
    assert [r.locus for r in jacoby_homer] == ["Il.?.116"], jacoby_homer
    assert near("homer_scholia", "Il.?.116", "Il.19.118", 3) and not near("homer_scholia", "Il.?.116", "Od.19.116", 3)
    assert near("euripides_scholia", "Ph.159", "Ph.161", 2) and not near("euripides_scholia", "Ph.159", "Or.159", 2)
    assert near("athenaeus", "542f", "543a", 1) and not near("athenaeus", "542a", "542c", 1)
    assert near("pindar_scholia", "O.9.104", "O.9.106", 2) and not near("pindar_scholia", "O.9.104", "P.9.104", 2)
    dfhg = from_dfhg([{"author": "X", "n": "1", "witness": "Athen. XII, p. 527, F | Idem XIV, p. 649, F:"}])
    assert (dfhg[0].witness, dfhg[0].locus) == ("athenaeus", "649f"), dfhg


if __name__ == "__main__":
    main()
