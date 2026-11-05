"""Offline self-test for Corpus Corporum text extraction. Touches no network or corpus.

    python scripts/corpus_corporum_selftest.py

Checks that nested sections yield each sentence once, in order, and that the
site's paratext and catalogue metadata stay out of the text. Exit code 1 on failure.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from ingest.base import split_page_markers
from ingest.corpus_corporum import CorpusCorporumConnector

SAMPLE = """<xml_response xmlns="http://mlat.uzh.ch/2.0">
<response_meta><author>Auctor</author><name>Opus</name></response_meta>
<response_text xmlns=""><div class="text">
  <div class="text_section hlvl_0">
    <h1 class="head"><span class="s"/>Titulus</h1>
    <p><span class="s"/>Prima sententia.<span class="s"/>Secunda sententia.</p>
    <div class="text_section hlvl_1">
      <h2 class="head"><span class="s"/>Caput unum</h2>
      <p><span class="s"/>Tertia <milestone/>sententia.</p>
      <div class="text_section hlvl_2"><p><span class="s"/>Quarta sententia.</p></div>
    </div>
    <p><span class="s"/>Quinta sententia.</p>
  </div>
</div></response_text>
<response_paratext xmlns=""><div>p. 0273A * (Psal. XLIX)</div></response_paratext>
</xml_response>"""

EXPECTED = ["Titulus", "Prima sententia. Secunda sententia.", "Caput unum",
            "Tertia sententia.", "Quarta sententia.", "Quinta sententia."]


MARKED = ["Titulus.", "p. 0719A | Prima sententia.", "Secunda p. 0719B | sententia.",
          "Tertia p. 0720 | sententia.", "p. 0721C |", "Quarta sententia."]
UNMARKED = [("Titulus.", None), ("Prima sententia.", "p. 0719A"),
            ("Secunda sententia.", "p. 0719A"), ("Tertia sententia.", "p. 0719B"),
            ("Quarta sententia.", "p. 0721C")]


def main() -> int:
    connector = CorpusCorporumConnector.__new__(CorpusCorporumConnector)
    got = connector._extract_prose(ET.fromstring(SAMPLE)).split("\n")
    failed = 0
    if got == EXPECTED:
        print("PASS  nested sections yield each sentence once, without paratext")
    else:
        failed = 1
        print("FAIL  extraction differs")
        print("  expected:", EXPECTED)
        print("  got:     ", got)
    got_pages = split_page_markers(MARKED)
    if got_pages == UNMARKED:
        print("PASS  page markers leave the text and give each sentence its starting page")
    else:
        failed = 1
        print("FAIL  page markers differ")
        print("  expected:", UNMARKED)
        print("  got:     ", got_pages)
    return failed


if __name__ == "__main__":
    sys.exit(main())
