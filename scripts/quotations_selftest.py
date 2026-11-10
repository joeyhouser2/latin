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
    assert set(WITNESSES) >= {"athenaeus", "suda", "pindar_scholia"}
    print("quotations self-test passed")


if __name__ == "__main__":
    main()
