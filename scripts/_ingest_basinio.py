"""One-off ingest of Basinio da Parma's *Hesperis* (13 books, Neo-Latin hexameter epic).

Source: the text extracted from Drudi's 1794 Rimini edition (Basinii Parmensis poetae
opera praestantiora, vol. 1, pp. 1-288; Google Books id AuzlAAAAMAAJ), OCR'd, repaired
and polished in the sibling book_creator repo. See docs/roadmap-basinio-hesperis-htr.md
for why no manuscript HTR was needed.

    python scripts/_ingest_basinio.py [path-to-hesperis_la_polished.txt] [--dry-run]

One section per book, one segment per verse line. Stray line-number marginalia and
scan symbols are stripped; running heads are dropped.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from pipeline import Library
from ingest.base import Connector

DEFAULT_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "book_creator", "cache", "basinio", "hesperis_la_polished.txt")

HEAD = re.compile(r"^LIBER (\d+)$")
RUNNING_HEAD = re.compile(r"^(PAR\s*MENSIS|BASINII.*|HESPERIDOS.*)$")


def clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"[\s,.]*\d{1,4}[,.]?\s*$", lambda m: "," if "," in m.group(0) else "", line)
    line = re.sub(r"\s*[•^*\"]+\s*$", "", line)      # trailing scan symbols
    line = re.sub(r"^[•^*]+\s*", "", line)           # leading scan symbols
    return line.strip()


def split_books(text: str):
    books, cur = [], None
    for raw in text.splitlines():
        s = raw.strip()
        m = HEAD.match(s)
        if m:
            cur = (int(m.group(1)), [])
            books.append(cur)
            continue
        if cur is None or not s or RUNNING_HEAD.match(s):
            continue
        c = clean_line(raw)
        if c:
            cur[1].append(c)
    return books


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = args[0] if args else DEFAULT_SRC
    books = split_books(open(src, encoding="utf-8").read())
    assert [n for n, _ in books] == list(range(1, 14)), [n for n, _ in books]
    for n, lines in books:
        print(f"  Liber {n}: {len(lines)} lines")
    print(f"  total {sum(len(l) for _, l in books)} lines")
    if "--dry-run" in sys.argv:
        return
    meta = {
        "title": "Hesperis",
        "author": "Basinius Parmensis",
        "century": 15,
        "genre": "poetry",
        "language": "la",
        "language_stage": "early_modern",
        "source": "Basinii Parmensis poetae opera praestantiora, vol. 1, ed. L. Drudi "
                  "(Rimini, 1794), pp. 1-288; text via Google Books AuzlAAAAMAAJ",
        "translation_status": "untranslated",
        "_verse": True,
    }
    parts = [(f"Liber {n}", "\n".join(lines)) for n, lines in books]
    doc = Connector.build_document(meta, parts, use_cltk=False)
    lib = Library()
    try:
        lib.ingest(doc)
        print(f"+ [{doc.id}] {doc.author} — {doc.title}  ({len(list(doc.iter_segments()))} segments)")
    finally:
        lib.close()


if __name__ == "__main__":
    main()
