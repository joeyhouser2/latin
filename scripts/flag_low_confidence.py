"""Mark documents' machine translations as low-confidence (documents.mt_confidence).

    python scripts/flag_low_confidence.py --source-prefix Capitularia          # dry run
    python scripts/flag_low_confidence.py --source-prefix Capitularia --apply
    python scripts/flag_low_confidence.py --source-prefix Capitularia --clear --apply

Spot-check basis for Capitularia: comes->"committee", names garbled, clauses
inverted or repeated. The flag is a reader-facing warning, not a gate.
"""
import argparse
import sqlite3
import sys

sys.path.insert(0, ".")
from core.store import Store  # noqa: E402  (runs the additive migration)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-prefix", required=True)
    ap.add_argument("--clear", action="store_true", help="remove the flag instead")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--db", default="data/corpus.db")
    a = ap.parse_args()
    Store(a.db)  # ensure the column exists
    conn = sqlite3.connect(a.db, timeout=60)
    value = None if a.clear else "low"
    n = conn.execute("SELECT COUNT(*) FROM documents WHERE source LIKE ?",
                     (a.source_prefix + "%",)).fetchone()[0]
    print(f"{n} documents match source prefix {a.source_prefix!r}; "
          f"{'clearing' if a.clear else 'setting low'} mt_confidence")
    if a.apply:
        conn.execute("UPDATE documents SET mt_confidence = ? WHERE source LIKE ?",
                     (value, a.source_prefix + "%"))
        conn.commit()
        print("applied")
    else:
        print("dry run; pass --apply")


if __name__ == "__main__":
    main()
