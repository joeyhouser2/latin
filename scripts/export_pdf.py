"""Export a document's stylized (or plain) English text to a readable PDF.

Usage:
    python scripts/export_pdf.py --doc-id 376
    python scripts/export_pdf.py --doc-id 376 --out data/exports/salmasius.pdf
"""
from __future__ import annotations

import argparse
import os
import sys
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.garble_detect import UNTRANSLATABLE_PLACEHOLDER

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from xml.sax.saxutils import escape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", type=int, required=True)
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    c = conn.cursor()
    c.execute("SELECT title, author, century, source FROM documents WHERE id=?", (args.doc_id,))
    doc_row = c.fetchone()
    if not doc_row:
        print(f"No document with id {args.doc_id}")
        return
    title, author, century, source = doc_row

    c.execute(
        """SELECT s.id, s.english_styled, s.english_text
           FROM segments s JOIN sections sec ON s.section_id = sec.id
           WHERE sec.doc_id = ? ORDER BY sec.ord, s.ord""",
        (args.doc_id,),
    )
    rows = c.fetchall()

    out_path = args.out or f"data/exports/doc{args.doc_id}_victorian.pdf"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleBig", parent=styles["Title"], fontSize=22, spaceAfter=6)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=13,
                                     alignment=TA_CENTER, textColor=colors.grey, spaceAfter=4)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontName="Times-Roman",
                                 fontSize=11.5, leading=16, spaceAfter=10,
                                 firstLineIndent=18, alignment=4)  # 4 = justify
    note_style = ParagraphStyle("Note", parent=styles["Normal"], fontName="Times-Italic",
                                 fontSize=10, textColor=colors.grey, spaceAfter=10,
                                 leftIndent=18)

    doc = SimpleDocTemplate(out_path, pagesize=LETTER,
                             topMargin=1 * inch, bottomMargin=1 * inch,
                             leftMargin=1.1 * inch, rightMargin=1.1 * inch,
                             title=title, author=author or "")

    story = []
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph(escape(title), title_style))
    if author:
        story.append(Paragraph(escape(author), subtitle_style))
    meta_bits = []
    if century:
        meta_bits.append(f"{century}th century")
    story.append(Paragraph("Victorian-prose stylized English translation", subtitle_style))
    if meta_bits:
        story.append(Paragraph(", ".join(meta_bits), subtitle_style))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "Machine-translated from the Latin (NLLB) and rendered into Victorian-prose "
        "style by a local LLM stylizer; source OCR long-s corruption corrected by an "
        "automated dictionary + trained-classifier pipeline. A small number of segments "
        "were untranslatable (garbled embedded Greek quotations in the original scan) "
        "and are marked as such below.",
        subtitle_style))
    story.append(PageBreak())

    n_placeholder = 0
    for seg_id, styled, plain in rows:
        text = (styled or "").strip() or (plain or "").strip()
        if not text or text == UNTRANSLATABLE_PLACEHOLDER:
            n_placeholder += 1
            story.append(Paragraph("[untranslatable fragment in source]", note_style))
            continue
        story.append(Paragraph(escape(text), body_style))

    doc.build(story)
    print(f"Wrote {out_path}")
    print(f"  {len(rows):,} segments, {n_placeholder} untranslatable placeholders")


if __name__ == "__main__":
    main()
