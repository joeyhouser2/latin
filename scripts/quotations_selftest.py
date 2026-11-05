"""Self-test for ingest/quotations.py. Needs no network or downloaded data.

    python scripts/quotations_selftest.py
"""
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.quotations import Witness, author_stems, find_citations, load_dfhg, norm, read_pg, read_tei, resolve_witnesses, score

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
    print("quotations self-test passed")


if __name__ == "__main__":
    main()
