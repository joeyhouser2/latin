"""Gradio app: discover and read AI-translated Latin side by side.

Entry point for the reading + discovery library. Talks only to pipeline.Library,
so the same logic backs a future web frontend.

Run:
    python app.py
    # open http://localhost:7860
"""

from __future__ import annotations

from html import escape
from typing import List, Optional

import gradio as gr

from pipeline import Library
from core.models import Document, LANGUAGE_STAGES


library: Optional[Library] = None


def get_library() -> Library:
    global library
    if library is None:
        library = Library()
    return library


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

READER_CSS = """
<style>
.reader-meta { color: #666; margin: 0 0 1rem 0; font-size: 0.9rem; }
table.reader { width: 100%; border-collapse: collapse; }
table.reader th { text-align: left; padding: 0.4rem 0.8rem; border-bottom: 2px solid #ccc;
                  font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #888; }
table.reader td { vertical-align: top; padding: 0.5rem 0.8rem; border-bottom: 1px solid #eee;
                  width: 50%; line-height: 1.5; }
table.reader td.la { font-family: Georgia, 'Times New Roman', serif; }
table.reader td.en { color: #333; }
table.reader td.en.styled { font-family: Georgia, 'Times New Roman', serif; }
.scansion { display: block; color: #a07; font-size: 0.85rem; letter-spacing: 0.15em;
            margin-top: 0.2rem; font-family: 'Courier New', monospace; }
.untranslated { color: #b00; font-style: italic; opacity: 0.7; }
.unstyled { color: #888; font-style: italic; }
.hit { padding: 0.6rem 0.8rem; border: 1px solid #e5e5e5; border-radius: 8px; margin-bottom: 0.6rem; }
.hit .src { color: #888; font-size: 0.8rem; }
.hit .la { font-family: Georgia, serif; margin: 0.3rem 0; }
.hit .en { color: #333; }
.hit .score { float: right; color: #aaa; font-size: 0.8rem; }
</style>
"""


def _doc_label(doc: Document) -> str:
    author = doc.author or "Anon."
    stage = doc.language_stage.replace("_", " ")
    lang = f"{doc.language_name}, " if doc.language != "la" else ""
    flag = "" if doc.has_existing_translation else "  · untranslated"
    return f"{author} — {doc.title} ({lang}{stage}{flag})"


def document_choices() -> List[tuple]:
    return [(_doc_label(d), d.id) for d in get_library().list_documents()]


def render_reader(doc_id: Optional[int], view: str = "Literal") -> str:
    if not doc_id:
        return READER_CSS + "<p>Select a document to read.</p>"
    doc = get_library().get_document(int(doc_id))
    if doc is None:
        return READER_CSS + "<p>Document not found.</p>"

    meta_bits = [doc.author or "Anon.", doc.language_stage.replace("_", " ")]
    if doc.century:
        meta_bits.append(_century_label(doc.century))
    if doc.genre:
        meta_bits.append(doc.genre)
    if doc.source:
        meta_bits.append(doc.source)

    styled_view = view == "Stylized"
    en_header = "English (stylized)" if styled_view else "English (AI)"
    rows = []
    for seg in doc.iter_segments():
        la = escape(seg.latin_text)
        if seg.scansion:
            la += f'<span class="scansion">{escape(seg.scansion)}</span>'
        en_class = "en"
        if not seg.is_translated:
            en = '<span class="untranslated">— not yet translated —</span>'
        elif styled_view:
            if seg.is_styled:
                en, en_class = escape(seg.english_styled), "en styled"
            else:
                en = ('<span class="unstyled">— not yet stylized —</span><br>'
                      + escape(seg.english_text))
        else:
            en = escape(seg.english_text)
        rows.append(
            f'<tr><td class="la">{la}</td>'
            f'<td class="{en_class}">{en}</td></tr>'
        )

    table = (
        f'<table class="reader"><thead><tr><th>{escape(doc.language_name)}</th>'
        f'<th>{escape(en_header)}</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    return (
        READER_CSS
        + f"<h2>{escape(doc.title)}</h2>"
        + f'<p class="reader-meta">{escape(" · ".join(str(b) for b in meta_bits))}</p>'
        + table
    )


def _century_label(c: int) -> str:
    if c < 0:
        return f"{abs(c)}c BCE"
    return f"{c}c CE"


def translate_and_render(doc_id: Optional[int], view: str = "Literal") -> str:
    """Translate any untranslated segments of the document, then re-render."""
    if not doc_id:
        return render_reader(doc_id, view)
    get_library().translate_document(int(doc_id))
    return render_reader(doc_id, view)


# Reader preset label -> Stylizer preset name.
STYLE_PRESETS = {
    "Victorian prose": "victorian_prose",
    "Blank verse": "verse_blank",
    "Heroic couplets": "verse_couplet",
}


def stylize_and_render(doc_id: Optional[int], preset_label: str) -> str:
    """Stylize the document with the chosen preset, then show the Stylized view."""
    if not doc_id:
        return render_reader(doc_id, "Stylized")
    get_library().stylize_document(int(doc_id), STYLE_PRESETS.get(preset_label, "victorian_prose"))
    return render_reader(doc_id, "Stylized")


def scan_and_render(doc_id: Optional[int], meter: str, view: str = "Literal") -> str:
    """Scan the document's verse lines, then re-render with scansion shown."""
    if not doc_id:
        return render_reader(doc_id, view)
    get_library().scan_document(int(doc_id), meter=(meter or "hexameter"))
    return render_reader(doc_id, view)


def do_search(query: str, k: int, only_untranslated: bool, stage: str) -> str:
    if not query.strip():
        return READER_CSS + "<p>Enter a query (English or Latin).</p>"
    hits = get_library().search(
        query, k=int(k),
        only_untranslated_works=only_untranslated,
        language_stage=(stage or None),
    )
    if not hits:
        return READER_CSS + "<p>No results.</p>"

    blocks = []
    for hit in hits:
        en = escape(hit.segment.english_text) if hit.segment.is_translated else \
            '<span class="untranslated">— not yet translated —</span>'
        src = f"{hit.document.author or 'Anon.'} — {hit.document.title}"
        if hit.segment.source_loc:
            src += f" · {hit.segment.source_loc}"
        blocks.append(
            f'<div class="hit"><span class="score">{hit.score:.3f}</span>'
            f'<div class="src">{escape(src)}</div>'
            f'<div class="la">{escape(hit.segment.latin_text)}</div>'
            f'<div class="en">{en}</div></div>'
        )
    return READER_CSS + "".join(blocks)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def build_interface():
    with gr.Blocks(title="Latin Reader") as demo:
        gr.Markdown(
            "# 📜 Latin Reader\n"
            "Discover and read AI-translated Latin — medieval, late-antique, and "
            "otherwise untranslated texts — with the Latin and English side by side."
        )

        with gr.Tab("📖 Read"):
            with gr.Row():
                doc_dropdown = gr.Dropdown(
                    label="Document", choices=document_choices(),
                    value=None, scale=4,
                )
                refresh_btn = gr.Button("↻", scale=0, min_width=50)
                translate_btn = gr.Button("Translate untranslated", scale=1)
            with gr.Row():
                view_radio = gr.Radio(
                    ["Literal", "Stylized"], value="Literal",
                    label="English column", scale=1,
                )
                preset_dropdown = gr.Dropdown(
                    label="Style preset", choices=list(STYLE_PRESETS),
                    value="Victorian prose", scale=1,
                )
                stylize_btn = gr.Button("Stylize", scale=1)
                meter_dropdown = gr.Dropdown(
                    label="Metre", choices=["hexameter", "pentameter", "elegiac"],
                    value="hexameter", scale=1,
                )
                scan_btn = gr.Button("Scan verse", scale=1)
            reader_html = gr.HTML(render_reader(None))

            doc_dropdown.change(render_reader, [doc_dropdown, view_radio], reader_html)
            view_radio.change(render_reader, [doc_dropdown, view_radio], reader_html)
            translate_btn.click(translate_and_render, [doc_dropdown, view_radio], reader_html)
            stylize_btn.click(stylize_and_render, [doc_dropdown, preset_dropdown], reader_html)
            scan_btn.click(scan_and_render, [doc_dropdown, meter_dropdown, view_radio], reader_html)
            refresh_btn.click(
                lambda: gr.update(choices=document_choices()), None, doc_dropdown
            )

        with gr.Tab("🔍 Discover"):
            with gr.Row():
                query_box = gr.Textbox(
                    label="Search (English or Latin)",
                    placeholder="sermons on usury · Quid est tempus · the siege of the city",
                    scale=3, lines=1,
                )
                search_btn = gr.Button("Search", variant="primary", scale=1)
            with gr.Row():
                k_slider = gr.Slider(1, 20, value=5, step=1, label="Results")
                stage_dropdown = gr.Dropdown(
                    label="Language stage", choices=[""] + list(LANGUAGE_STAGES),
                    value="",
                )
                untranslated_only = gr.Checkbox(
                    label="Only works without an existing translation", value=False
                )
            search_html = gr.HTML(READER_CSS + "<p>Enter a query.</p>")
            search_btn.click(
                do_search,
                [query_box, k_slider, untranslated_only, stage_dropdown],
                search_html,
            )
            query_box.submit(
                do_search,
                [query_box, k_slider, untranslated_only, stage_dropdown],
                search_html,
            )

    return demo


if __name__ == "__main__":
    print("Initializing library...")
    get_library()
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                theme=gr.themes.Soft())
