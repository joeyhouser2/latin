"""
Latin RAG Web Interface

A simple Gradio interface for querying Latin texts with translation.

Install:
    pip install gradio transformers torch faiss-cpu sentence-transformers

Run:
    python latin_rag_ui.py
"""

import gradio as gr
from latin_rag_pipeline import LatinRAG, LatinPassage
from typing import List, Tuple
import time


# ============================================================================
# GLOBAL STATE
# ============================================================================

rag = None  # Will be initialized on startup


def initialize_rag():
    """Initialize the RAG pipeline with sample data."""
    global rag
    
    print("Initializing Latin RAG pipeline...")
    rag = LatinRAG()
    
    # Sample texts - replace with your actual corpus
    sample_texts = [
        ("""
        Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, 
        aliam Aquitani, tertiam qui ipsorum lingua Celtae, nostra Galli appellantur. 
        Hi omnes lingua, institutis, legibus inter se differunt. Gallos ab Aquitanis 
        Garumna flumen, a Belgis Matrona et Sequana dividit. Horum omnium fortissimi 
        sunt Belgae, propterea quod a cultu atque humanitate provinciae longissime 
        absunt, minimeque ad eos mercatores saepe commeant atque ea quae ad 
        effeminandos animos pertinent important.
        """, "Caesar, De Bello Gallico, Book 1"),
        
        ("""
        Arma virumque cano, Troiae qui primus ab oris Italiam, fato profugus, 
        Laviniaque venit litora, multum ille et terris iactatus et alto vi superum 
        saevae memorem Iunonis ob iram; multa quoque et bello passus, dum conderet 
        urbem, inferretque deos Latio, genus unde Latinum, Albanique patres, atque 
        altae moenia Romae.
        """, "Virgil, Aeneid, Book 1"),
        
        ("""
        Quid est ergo tempus? Si nemo ex me quaerat, scio; si quaerenti explicare 
        velim, nescio. Fidenter tamen dico scire me quod, si nihil praeteriret, 
        non esset praeteritum tempus, et si nihil adveniret, non esset futurum 
        tempus, et si nihil esset, non esset praesens tempus.
        """, "Augustine, Confessions, Book 11"),
        
        ("""
        Confiteantur tibi, Domine, omnia opera tua et sancti tui benedicant tibi. 
        Gloriam regni tui dicent et potentiam tuam loquentur, ut notam faciant 
        filiis hominum potentiam tuam et gloriam magnificentiae regni tui.
        Regnum tuum regnum omnium saeculorum, et dominatio tua in omni generatione 
        et generationem.
        """, "Psalms 144 (Vulgate)"),
        
        ("""
        In principio creavit Deus caelum et terram. Terra autem erat inanis et vacua, 
        et tenebrae erant super faciem abyssi: et spiritus Dei ferebatur super aquas.
        Dixitque Deus: Fiat lux. Et facta est lux. Et vidit Deus lucem quod esset bona: 
        et divisit lucem a tenebris.
        """, "Genesis 1 (Vulgate)"),
        
        ("""
        Cogito, ergo sum. Ego sum, ego existo, quoties a me profertur, vel mente 
        concipitur, necessario esse verum. Sed nondum satis intelligo, quisnam sim 
        ego ille, qui jam necessario sum; deincepsque cavendum est ne forte quid 
        aliud imprudenter assumam in locum mei.
        """, "Descartes, Meditationes (Latin)"),
    ]
    
    print("Indexing sample texts...")
    rag.index_texts(sample_texts, chunk_size=400, overlap=100)
    print("Ready!")
    
    return rag


# ============================================================================
# QUERY FUNCTION
# ============================================================================

def query_latin(
    query: str, 
    num_results: int = 3, 
    translate: bool = True
) -> Tuple[str, str]:
    """
    Query the Latin corpus.
    
    Returns:
        Tuple of (formatted_results, status_message)
    """
    global rag
    
    if rag is None:
        return "Error: RAG pipeline not initialized", "❌ Not ready"
    
    if not query.strip():
        return "Please enter a query", "⚠️ Empty query"
    
    try:
        start_time = time.time()
        results = rag.query(query, k=num_results, translate=translate)
        elapsed = time.time() - start_time
        
        # Format results
        output_parts = []
        for i, result in enumerate(results, 1):
            output_parts.append(f"## Result {i}")
            output_parts.append(f"**Source:** {result.passage.source}")
            output_parts.append(f"**Relevance Score:** {result.score:.3f}")
            output_parts.append("")
            output_parts.append("### Latin")
            output_parts.append(f"_{result.passage.text}_")
            output_parts.append("")
            
            if result.translation:
                output_parts.append("### English Translation")
                output_parts.append(result.translation)
            
            output_parts.append("")
            output_parts.append("---")
            output_parts.append("")
        
        formatted = "\n".join(output_parts)
        status = f"✅ Found {len(results)} results in {elapsed:.2f}s"
        
        return formatted, status
        
    except Exception as e:
        return f"Error: {str(e)}", f"❌ Error: {str(e)}"


def add_text_to_corpus(latin_text: str, source_name: str) -> str:
    """Add new text to the corpus."""
    global rag
    
    if rag is None:
        return "❌ RAG pipeline not initialized"
    
    if not latin_text.strip() or not source_name.strip():
        return "⚠️ Please provide both text and source name"
    
    try:
        rag.index_texts([(latin_text, source_name)], chunk_size=400, overlap=100)
        return f"✅ Added '{source_name}' to corpus ({len(latin_text)} characters)"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def translate_only(latin_text: str) -> str:
    """Translate Latin text without RAG."""
    global rag
    
    if rag is None:
        return "❌ Pipeline not initialized"
    
    if not latin_text.strip():
        return "⚠️ Please enter Latin text"
    
    try:
        rag._ensure_translator()
        translation = rag.translator.translate(latin_text)
        return translation
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Latin RAG + Translation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 📜 Latin RAG + Translation
        
        Search through Latin texts and get English translations.
        
        **How it works:**
        1. Enter a query in English or Latin
        2. The system finds relevant passages using semantic search
        3. Each passage is translated to English using NLLB-200
        """)
        
        with gr.Tab("🔍 Search"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="Query",
                        placeholder="What does Augustine say about time?",
                        lines=2
                    )
                    with gr.Row():
                        num_results = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="Number of results"
                        )
                        translate_checkbox = gr.Checkbox(
                            value=True, 
                            label="Translate to English"
                        )
                    search_btn = gr.Button("🔍 Search", variant="primary")
                
                with gr.Column(scale=1):
                    status_output = gr.Textbox(label="Status", interactive=False)
            
            results_output = gr.Markdown(label="Results")
            
            search_btn.click(
                fn=query_latin,
                inputs=[query_input, num_results, translate_checkbox],
                outputs=[results_output, status_output]
            )
            
            # Example queries
            gr.Examples(
                examples=[
                    ["What is time?"],
                    ["Gaul divided into parts"],
                    ["creation of the world"],
                    ["I think therefore I am"],
                    ["arms and the man"],
                ],
                inputs=query_input
            )
        
        with gr.Tab("📝 Translate"):
            gr.Markdown("### Direct Latin → English Translation")
            
            latin_input = gr.Textbox(
                label="Latin Text",
                placeholder="Enter Latin text to translate...",
                lines=4
            )
            translate_btn = gr.Button("🔄 Translate", variant="primary")
            translation_output = gr.Textbox(
                label="English Translation",
                lines=4,
                interactive=False
            )
            
            translate_btn.click(
                fn=translate_only,
                inputs=latin_input,
                outputs=translation_output
            )
            
            gr.Examples(
                examples=[
                    ["Gallia est omnis divisa in partes tres."],
                    ["Cogito, ergo sum."],
                    ["In principio creavit Deus caelum et terram."],
                    ["Veni, vidi, vici."],
                ],
                inputs=latin_input
            )
        
        with gr.Tab("➕ Add Texts"):
            gr.Markdown("### Add New Latin Texts to Corpus")
            
            new_text = gr.Textbox(
                label="Latin Text",
                placeholder="Paste your Latin text here...",
                lines=10
            )
            source_name = gr.Textbox(
                label="Source Name",
                placeholder="e.g., Cicero, De Officiis, Book 1"
            )
            add_btn = gr.Button("➕ Add to Corpus", variant="primary")
            add_status = gr.Textbox(label="Status", interactive=False)
            
            add_btn.click(
                fn=add_text_to_corpus,
                inputs=[new_text, source_name],
                outputs=add_status
            )
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ### About This System
            
            **Components:**
            - **Embedding Model:** `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)
            - **Vector Database:** FAISS (Facebook AI Similarity Search)
            - **Translation Model:** NLLB-200 (No Language Left Behind)
            
            **Architecture:**
            ```
            Query (English/Latin)
                    ↓
            Embed with multilingual model
                    ↓
            Search FAISS index
                    ↓
            Retrieve top-k Latin passages
                    ↓
            Translate with NLLB-200
                    ↓
            Display results
            ```
            
            **Supported Languages:**
            - Queries: English or Latin
            - Corpus: Latin
            - Output: Latin + English translation
            
            **Limitations:**
            - NLLB-200 is trained mostly on modern/ecclesiastical Latin
            - Classical Latin may have lower translation quality
            - Consider fine-tuning for specialized domains
            
            **To improve results:**
            1. Add more texts to the corpus
            2. Fine-tune NLLB on classical Latin parallel texts
            3. Use Latin BERT for embeddings (better Latin representation)
            """)
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("LATIN RAG + TRANSLATION")
    print("="*60)
    
    # Initialize RAG
    initialize_rag()
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False  # Set True to create public link
    )