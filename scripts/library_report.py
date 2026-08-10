"""Generate a prose summary of the library's translated holdings, using a local
GPU LLM to turn the raw DB stats into a readable report.

Usage:
    python scripts/library_report.py [--out data/library_report.md] [--no-llm]

``--no-llm`` just prints the raw stats table (no GPU needed) -- useful for a
quick check before spending GPU time on the prose pass.
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_stats(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    total_docs = conn.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
    total_segs = conn.execute("SELECT COUNT(*) n FROM segments").fetchone()["n"]
    translated_segs = conn.execute(
        "SELECT COUNT(*) n FROM segments WHERE english_text IS NOT NULL AND english_text != ''"
    ).fetchone()["n"]
    styled_segs = conn.execute(
        "SELECT COUNT(*) n FROM segments WHERE english_styled IS NOT NULL AND english_styled != ''"
    ).fetchone()["n"]

    by_source: Counter = Counter()
    for row in conn.execute("SELECT source FROM documents"):
        key = re.sub(r"\s*\(.*", "", row["source"] or "unknown").strip()
        by_source[key] += 1

    by_language = {r["language"]: r["n"] for r in conn.execute(
        "SELECT language, COUNT(*) n FROM documents GROUP BY language"
    )}
    by_stage = {(r["language_stage"] or "unknown"): r["n"] for r in conn.execute(
        "SELECT language_stage, COUNT(*) n FROM documents GROUP BY language_stage ORDER BY n DESC"
    )}
    by_genre = {(r["genre"] or "unclassified"): r["n"] for r in conn.execute(
        "SELECT genre, COUNT(*) n FROM documents GROUP BY genre ORDER BY n DESC"
    )}
    century_row = conn.execute(
        "SELECT MIN(century) mn, MAX(century) mx FROM documents WHERE century IS NOT NULL"
    ).fetchone()

    top_docs = conn.execute("""
        SELECT d.title, d.author, d.source, COUNT(seg.id) n_seg
        FROM documents d
        JOIN sections s ON s.doc_id = d.id
        JOIN segments seg ON seg.section_id = s.id
        GROUP BY d.id ORDER BY n_seg DESC LIMIT 8
    """).fetchall()

    conn.close()
    return {
        "total_docs": total_docs,
        "total_segs": total_segs,
        "translated_segs": translated_segs,
        "styled_segs": styled_segs,
        "by_source": dict(by_source.most_common()),
        "by_language": by_language,
        "by_stage": by_stage,
        "by_genre": by_genre,
        "century_min": century_row["mn"],
        "century_max": century_row["mx"],
        "top_docs": [dict(r) for r in top_docs],
    }


def render_stats_table(stats: dict) -> str:
    lines = [
        f"Documents: {stats['total_docs']}",
        f"Segments: {stats['translated_segs']:,} / {stats['total_segs']:,} translated "
        f"({100 * stats['translated_segs'] / max(stats['total_segs'], 1):.1f}%), "
        f"{stats['styled_segs']:,} stylized",
        f"Era span: {stats['century_min']}c to {stats['century_max']}c",
        "",
        "By source:",
    ]
    lines += [f"  {n:4d}  {k}" for k, n in stats["by_source"].items()]
    lines += ["", "By language:"] + [f"  {n:4d}  {k}" for k, n in stats["by_language"].items()]
    lines += ["", "By era stage:"] + [f"  {n:4d}  {k}" for k, n in stats["by_stage"].items()]
    lines += ["", "By genre:"] + [f"  {n:4d}  {k}" for k, n in stats["by_genre"].items()]
    lines += ["", "Largest works:"]
    lines += [f"  {d['n_seg']:6d} segs  {d['title']} ({d['author'] or 'anon'}) [{d['source']}]"
              for d in stats["top_docs"]]
    return "\n".join(lines)


def render_prose(stats: dict) -> str:
    """Feed the stats to the local GPU LLM and get a narrative report back."""
    from core.stylizer import LocalLLMStylizer

    system = (
        "You are a librarian's assistant writing a status report for a personal "
        "digital library of understudied Latin and Greek texts (late-antique, "
        "medieval, largely untranslated into English)."
    )
    user = (
        "Write a clear, well-organized prose report (4-6 short paragraphs, plain "
        "text, no markdown headers) summarizing the current state of the library "
        "from the statistics below. Cover: overall scale, the era/language spread, "
        "which sources dominate and what they are (e.g. Musa Medievalis is "
        "medieval Latin poetry, ALIM is medieval Italian Latin, DigilibLT is "
        "late-antique Latin, PTA/PG Corpus are patristic Greek), translation "
        "coverage, and what stands out about the largest works. Do not invent "
        "numbers not given below.\n\n" + render_stats_table(stats)
    )
    stylizer = LocalLLMStylizer()
    report = stylizer._generate(system, user)
    stylizer.close()
    return report.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("--out", default="data/library_report.md")
    ap.add_argument("--no-llm", action="store_true", help="skip the GPU prose pass")
    args = ap.parse_args()

    stats = compute_stats(args.db)
    table = render_stats_table(stats)
    print(table)
    print()

    if args.no_llm:
        return

    print("Generating prose report on GPU...")
    prose = render_prose(stats)
    print()
    print(prose)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(prose + "\n\n---\n\n" + table + "\n", encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
