"""
Latin RAG + Translation Pipeline

Architecture:
1. Embed Latin texts with multilingual model (works for both Latin and English queries)
2. Store in FAISS vector database
3. Retrieve relevant passages
4. Translate retrieved Latin to English with NLLB-200

Requirements:
    pip install transformers torch faiss-cpu sentence-transformers accelerate
"""

import os
import json
import torch
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LatinPassage:
    """A chunk of Latin text with metadata."""
    text: str
    source: str  # e.g., "Augustine, Confessions, Book 1"
    chunk_id: int
    metadata: Optional[Dict] = None


@dataclass 
class RetrievalResult:
    """A retrieved passage with score and translation."""
    passage: LatinPassage
    score: float
    translation: Optional[str] = None


# ============================================================================
# EMBEDDING MODEL (for RAG retrieval)
# ============================================================================

class LatinEmbedder:
    """
    Embeds Latin text for semantic search.
    
    Uses multilingual model so you can query in English OR Latin.
    Options:
        - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (fast, decent)
        - intfloat/multilingual-e5-base (better quality)
        - For pure Latin: use Latin BERT directly
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts, returns [n_texts, embedding_dim] array."""
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.model.encode([query], convert_to_numpy=True)[0]


# ============================================================================
# VECTOR DATABASE
# ============================================================================

class LatinVectorDB:
    """
    FAISS-based vector store for Latin passages.
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine sim with normalized vecs)
        self.passages: List[LatinPassage] = []
        self.embedding_dim = embedding_dim
    
    def add(self, passages: List[LatinPassage], embeddings: np.ndarray):
        """Add passages and their embeddings to the index."""
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        self.passages.extend(passages)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[LatinPassage, float]]:
        """Search for k most similar passages."""
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.passages):
                results.append((self.passages[idx], float(score)))
        return results
    
    def save(self, path: str):
        """Save index and passages to disk."""
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.passages.json", 'w') as f:
            json.dump([
                {"text": p.text, "source": p.source, "chunk_id": p.chunk_id, "metadata": p.metadata}
                for p in self.passages
            ], f)
    
    def load(self, path: str):
        """Load index and passages from disk."""
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.passages.json", 'r') as f:
            data = json.load(f)
            self.passages = [
                LatinPassage(d["text"], d["source"], d["chunk_id"], d.get("metadata"))
                for d in data
            ]


# ============================================================================
# TRANSLATION MODEL
# ============================================================================

class LatinTranslator:
    """
    Translates Latin to English using NLLB-200.
    
    NLLB language codes:
        - Latin: lat_Latn
        - English: eng_Latn
    """
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        print(f"Loading translation model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Get language token IDs
        self.eng_token_id = self.tokenizer.convert_tokens_to_ids("eng_Latn")
    
    def translate(self, latin_text: str, max_length: int = 256) -> str:
        """Translate Latin text to English."""
        # Tokenize with Latin as source
        inputs = self.tokenizer(latin_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate English
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.eng_token_id,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    
    def translate_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Translate multiple texts."""
        translations = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.eng_token_id,
                    max_length=256,
                    num_beams=4
                )
            
            batch_translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(batch_translations)
        
        return translations


# ============================================================================
# TEXT CHUNKING
# ============================================================================

def chunk_latin_text(
    text: str, 
    source: str,
    chunk_size: int = 500,  # characters
    overlap: int = 100
) -> List[LatinPassage]:
    """
    Split Latin text into overlapping chunks.
    
    For better results, you might want to use sentence-aware chunking
    with CLTK's Latin sentence tokenizer.
    """
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary (period followed by space/newline)
        if end < len(text):
            # Look for sentence boundary in last 100 chars of chunk
            search_start = max(start + chunk_size - 100, start)
            search_region = text[search_start:end]
            
            # Find last period followed by space or newline
            for marker in ['. ', '.\n', '? ', '?\n', '! ', '!\n']:
                last_break = search_region.rfind(marker)
                if last_break != -1:
                    end = search_start + last_break + len(marker)
                    break
        
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(LatinPassage(
                text=chunk_text,
                source=source,
                chunk_id=chunk_id
            ))
            chunk_id += 1
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


# ============================================================================
# MAIN RAG PIPELINE
# ============================================================================

class LatinRAG:
    """
    Complete RAG pipeline for Latin texts.
    
    Usage:
        rag = LatinRAG()
        rag.index_texts([("path/to/text.txt", "Source Name"), ...])
        results = rag.query("What does Augustine say about memory?")
    """
    
    def __init__(
        self,
        embedder: Optional[LatinEmbedder] = None,
        translator: Optional[LatinTranslator] = None,
        vector_db: Optional[LatinVectorDB] = None
    ):
        self.embedder = embedder or LatinEmbedder()
        self.translator = translator  # Lazy load - expensive
        self.vector_db = vector_db or LatinVectorDB(embedding_dim=384)
    
    def _ensure_translator(self):
        """Load translator on first use."""
        if self.translator is None:
            self.translator = LatinTranslator()
    
    def index_texts(
        self, 
        texts: List[Tuple[str, str]],  # (text_content, source_name)
        chunk_size: int = 500,
        overlap: int = 100
    ):
        """
        Index Latin texts for retrieval.
        
        Args:
            texts: List of (text_content, source_name) tuples
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
        """
        all_passages = []
        
        for text_content, source_name in texts:
            passages = chunk_latin_text(text_content, source_name, chunk_size, overlap)
            all_passages.extend(passages)
            print(f"Chunked '{source_name}' into {len(passages)} passages")
        
        print(f"\nEmbedding {len(all_passages)} total passages...")
        embeddings = self.embedder.embed([p.text for p in all_passages])
        
        self.vector_db.add(all_passages, embeddings)
        print(f"Indexed {len(all_passages)} passages")
    
    def index_files(self, file_paths: List[Tuple[str, str]]):
        """
        Index Latin text files.
        
        Args:
            file_paths: List of (file_path, source_name) tuples
        """
        texts = []
        for file_path, source_name in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append((f.read(), source_name))
        self.index_texts(texts)
    
    def query(
        self,
        query: str,
        k: int = 5,
        translate: bool = True
    ) -> List[RetrievalResult]:
        """
        Query the Latin corpus.
        
        Args:
            query: Search query (English or Latin)
            k: Number of results to return
            translate: Whether to translate results to English
        
        Returns:
            List of RetrievalResult with passages, scores, and optional translations
        """
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Search
        search_results = self.vector_db.search(query_embedding, k=k)
        
        # Optionally translate
        results = []
        for passage, score in search_results:
            translation = None
            if translate:
                self._ensure_translator()
                translation = self.translator.translate(passage.text)
            
            results.append(RetrievalResult(
                passage=passage,
                score=score,
                translation=translation
            ))
        
        return results
    
    def save(self, path: str):
        """Save the vector database to disk."""
        self.vector_db.save(path)
        print(f"Saved index to {path}")
    
    def load(self, path: str):
        """Load the vector database from disk."""
        self.vector_db.load(path)
        print(f"Loaded index from {path} ({len(self.vector_db.passages)} passages)")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def interactive_mode(rag: LatinRAG):
    """Run interactive query mode."""
    print("\n" + "="*60)
    print("LATIN RAG - Interactive Mode")
    print("="*60)
    print("Enter queries in English or Latin. Type 'quit' to exit.\n")
    
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
        
        print("\nSearching...")
        results = rag.query(query, k=3, translate=True)
        
        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"--- Result {i} (score: {result.score:.3f}) ---")
            print(f"Source: {result.passage.source}")
            print(f"Latin: {result.passage.text[:300]}...")
            if result.translation:
                print(f"English: {result.translation}")
            print()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Index some sample Latin text
    
    sample_texts = [
        ("""
        Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, 
        aliam Aquitani, tertiam qui ipsorum lingua Celtae, nostra Galli appellantur. 
        Hi omnes lingua, institutis, legibus inter se differunt. Gallos ab Aquitanis 
        Garumna flumen, a Belgis Matrona et Sequana dividit.
        """, "Caesar, De Bello Gallico, Book 1"),
        
        ("""
        Confiteantur tibi, Domine, omnia opera tua et sancti tui benedicant tibi. 
        Gloriam regni tui dicent et potentiam tuam loquentur, ut notam faciant 
        filiis hominum potentiam tuam et gloriam magnificentiae regni tui.
        """, "Psalms (Vulgate)"),
        
        ("""
        Quid est ergo tempus? Si nemo ex me quaerat, scio; si quaerenti explicare 
        velim, nescio. Fidenter tamen dico scire me quod, si nihil praeteriret, 
        non esset praeteritum tempus, et si nihil adveniret, non esset futurum 
        tempus, et si nihil esset, non esset praesens tempus.
        """, "Augustine, Confessions, Book 11"),
    ]
    
    print("Initializing Latin RAG pipeline...")
    rag = LatinRAG()
    
    print("\nIndexing sample texts...")
    rag.index_texts(sample_texts, chunk_size=300, overlap=50)
    
    # Run a sample query
    print("\n" + "="*60)
    print("Sample Query: 'What is time?'")
    print("="*60)
    
    results = rag.query("What is time?", k=2, translate=True)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {result.score:.3f}) ---")
        print(f"Source: {result.passage.source}")
        print(f"Latin: {result.passage.text}")
        print(f"English: {result.translation}")
    
    # Uncomment to run interactive mode:
    # interactive_mode(rag)