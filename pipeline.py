"""Orchestration: ingest -> segment -> embed -> translate -> store.

The Library class wires the core components together and is the single object the
UI (or a future API) talks to. It owns the SQLite store, the FAISS index, the
embedder, and the translator, and keeps them in sync.
"""

from __future__ import annotations

import os
from typing import List, Optional

from core.models import Document, Section
from core.store import Store
from core.embedder import Embedder
from core.vectorstore import VectorStore
from core.translator import Translator, NLLBTranslator
from core.stylizer import Stylizer, LocalLLMStylizer, Seq2SeqStylizer, StyleUnit, PRESETS
from core.scansion import scan_lines
from core.segmenter import segment_text
from core.normalize import embedding_text_for, strip_greek_diacritics
from core.search import SearchService, SearchHit


# Translator candidates keyed by (language, language_stage), tried in order; first
# complete model dir wins, else stock NLLB. Each entry: (model_dir, src_lang,
# strip_greek_diacritics?). A "complete" model dir has model.safetensors in its
# root (not just checkpoints). Stage None is the per-language default, used for any
# stage without its own entry (and as a fallback appended to stage-specific lists).
TRANSLATOR_MODELS = {
    ("la", None): [("models/nllb-latin", "lat_Latn", False)],
    ("grc", None): [("models/nllb-greek-v2", "ell_Grek", True),  # normalized 4-epoch
                    ("models/nllb-greek", "ell_Grek", False)],   # fallback: v1
    # Patristic / late-antique Greek: prefer the era-specific model, then the
    # general Greek models, then stock NLLB.
    ("grc", "late_antique"): [("models/nllb-greek-v3", "ell_Grek", True)],
    # Homeric / archaic Greek (Workstream B). Inert until the model is trained;
    # falls back to the general Greek default below.
    ("grc", "archaic"): [("models/nllb-greek-archaic", "ell_Grek", True)],
}
STOCK_SRC = {"la": "lat_Latn", "grc": "ell_Grek"}

# Stylizer backends: "llm" = prompted local instruct model (rich, slow, all presets);
# "t5" = the trained fast/offline Victorian model (register baked in, victorian only).
VICTORIAN_T5_DIR = "models/stylizer-victorian"


class Library:
    def __init__(
        self,
        db_path: str = "data/corpus.db",
        index_path: str = "data/index.faiss",
        embedder: Optional[Embedder] = None,
        translator: Optional[Translator] = None,
        stylizer: Optional[Stylizer] = None,
    ):
        self.db_path = db_path
        self.index_path = index_path
        self.store = Store(db_path)
        self.embedder = embedder or Embedder()
        self.vectors = VectorStore()
        self.vectors.load(index_path)
        self.translator = translator  # explicit override; else routed per language
        self._lang_translators: dict = {}   # language -> Translator (lazy)
        self.stylizer = stylizer  # explicit override; else lazily built per backend
        self._stylizers: dict = {}   # backend -> Stylizer (lazy)
        self.search_service = SearchService(self.store, self.embedder, self.vectors)

    # -- ingestion -----------------------------------------------------------

    def add_document(
        self,
        raw_text: str,
        title: str,
        *,
        author: Optional[str] = None,
        century: Optional[int] = None,
        genre: Optional[str] = None,
        language: str = "la",
        language_stage: str = "unknown",
        source: Optional[str] = None,
        has_existing_translation: bool = False,
        section_label: str = "Text",
        use_cltk: bool = False,
        verse: bool = False,
    ) -> Document:
        """Segment raw text into one section, persist, and embed.

        ``verse=True`` segments one line per segment (for poetry) instead of by
        sentence, so the side-by-side reader aligns line-for-line and scansion has
        whole verse lines to work on."""
        sentences = segment_text(raw_text, use_cltk=use_cltk, lang=language, verse=verse)
        section = Section(label=section_label, order=0)
        section.segments = [
            _segment(s, i) for i, s in enumerate(sentences)
        ]
        doc = Document(
            title=title, author=author, century=century, genre=genre,
            language=language, language_stage=language_stage, source=source,
            has_existing_translation=has_existing_translation,
            sections=[section],
        )
        self.store.add_document(doc)
        self._embed_document(doc)
        self.vectors.save(self.index_path)
        return doc

    def ingest(self, doc: Document) -> Document:
        """Persist a fully-structured Document (multi-section) and embed it."""
        self.store.add_document(doc)
        self._embed_document(doc)
        self.vectors.save(self.index_path)
        return doc

    def _embed_document(self, doc: Document) -> None:
        segments = list(doc.iter_segments())
        if not segments:
            return
        embeddings = self.embedder.embed(
            [s.text_for_embedding for s in segments], show_progress=True
        )
        self.vectors.add([s.id for s in segments], embeddings)

    def reindex(self, batch_size: int = 256) -> int:
        """Backfill embed_text and rebuild the FAISS index from the store.

        Run after changing the embedding/normalization logic. Returns the number
        of segments indexed.
        """
        segments = self.store.iter_all_segments()
        if not segments:
            return 0

        # Backfill markup-stripped embedding text for any segment missing it.
        updates = []
        for s in segments:
            if s.embed_text is None:
                et = embedding_text_for(s.latin_text)
                if et is not None:
                    s.embed_text = et
                    updates.append((s.id, et))
        if updates:
            self.store.set_embed_texts(updates)
            print(f"Backfilled embed_text for {len(updates)} segments")

        self.vectors = VectorStore()
        self.search_service = SearchService(self.store, self.embedder, self.vectors)
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            embeddings = self.embedder.embed(
                [s.text_for_embedding for s in batch], show_progress=False
            )
            self.vectors.add([s.id for s in batch], embeddings)
        self.vectors.save(self.index_path)
        print(f"Reindexed {len(segments)} segments")
        return len(segments)

    # -- translation ---------------------------------------------------------

    def _ensure_translator(self) -> Translator:
        # Backwards-compatible default (Latin / stock).
        return self.translator_for("la")

    def translator_for(self, language: str, language_stage: str = "unknown") -> Translator:
        """Return the translator for a (language, stage): an explicit override if
        set, else the best available fine-tuned model, else stock NLLB. Stage-specific
        models are preferred over the per-language default."""
        if self.translator is not None:
            return self.translator
        key = (language, language_stage)
        if key not in self._lang_translators:
            self._lang_translators[key] = self._build_translator(language, language_stage)
        return self._lang_translators[key]

    @staticmethod
    def _build_translator(language: str, language_stage: str = "unknown") -> Translator:
        # Stage-specific candidates first, then the per-language default; dedupe so a
        # model dir listed in both is only probed once.
        candidates = list(TRANSLATOR_MODELS.get((language, language_stage), []))
        for entry in TRANSLATOR_MODELS.get((language, None), []):
            if entry not in candidates:
                candidates.append(entry)
        for model_dir, src_lang, normalize in candidates:
            if os.path.isfile(os.path.join(model_dir, "model.safetensors")):
                print(f"Using fine-tuned translator for {language}/{language_stage}: {model_dir}")
                return NLLBTranslator(
                    model_name=model_dir, src_lang=src_lang,
                    preprocess=strip_greek_diacritics if normalize else None,
                )
        print(f"No fine-tuned {language}/{language_stage} model found; using stock NLLB.")
        return NLLBTranslator(src_lang=STOCK_SRC.get(language, "lat_Latn"))

    def translate_document(self, doc_id: int, batch_size: int = 8) -> int:
        """Translate every untranslated segment of a document. Returns count."""
        doc = self.store.get_document(doc_id)
        if doc is None:
            return 0
        pending = [s for s in doc.iter_segments() if not s.is_translated]
        if not pending:
            return 0
        translator = self.translator_for(doc.language, doc.language_stage)
        englishes = translator.translate_batch(
            [s.latin_text for s in pending], batch_size=batch_size
        )
        self.store.set_translations(
            [(s.id, en) for s, en in zip(pending, englishes)]
        )
        return len(pending)

    # -- stylize (post-translation register / verse) -------------------------

    def _stylizer_for(self, backend: str = "llm") -> Stylizer:
        """Return the stylizer for a backend: an explicit override if set, else the
        prompted LLM ("llm") or the trained fast Victorian model ("t5", falling back
        to the LLM if the model dir is absent)."""
        if self.stylizer is not None:
            return self.stylizer
        if backend not in self._stylizers:
            self._stylizers[backend] = self._build_stylizer(backend)
        return self._stylizers[backend]

    @staticmethod
    def _build_stylizer(backend: str) -> Stylizer:
        if backend == "t5":
            if os.path.isfile(os.path.join(VICTORIAN_T5_DIR, "model.safetensors")):
                print(f"Using trained Victorian stylizer: {VICTORIAN_T5_DIR}")
                return Seq2SeqStylizer(model_name=VICTORIAN_T5_DIR, prefix="victorianize: ")
            print(f"No trained stylizer at {VICTORIAN_T5_DIR}; falling back to LLM.")
        return LocalLLMStylizer()

    def stylize_document(self, doc_id: int, preset: str = "victorian_prose",
                         backend: str = "llm") -> int:
        """Rewrite a document's literal translations into ``preset``'s register.

        ``backend`` picks the engine: "llm" (prompted, rich, honors any preset) or
        "t5" (trained fast/offline Victorian model — register is baked in, so
        ``preset`` is ignored and treated as victorian_prose). Stylizes section by
        section over already-translated segments, storing the result in
        ``english_styled`` without touching the literal ``english_text``. Returns the
        number of segments stylized."""
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        doc = self.store.get_document(doc_id)
        if doc is None:
            return 0
        stylizer = self._stylizer_for(backend)
        if backend == "t5":
            preset = "victorian_prose"   # the trained model's register; label accordingly
        context = {
            "source_lang": doc.language_name,
            "author": doc.author,
            "era": doc.language_stage.replace("_", " ") if doc.language_stage else None,
        }
        done = 0
        for section in sorted(doc.sections, key=lambda s: s.order):
            segs = [s for s in sorted(section.segments, key=lambda x: x.order)
                    if s.is_translated]
            if not segs:
                continue
            units = [StyleUnit(literal=s.english_text, source=s.latin_text,
                               scansion=s.scansion) for s in segs]
            styled = stylizer.stylize_units(units, preset=preset, context=context)
            self.store.set_styled(
                [(s.id, text, preset) for s, text in zip(segs, styled)]
            )
            done += len(segs)
        return done

    def scan_document(self, doc_id: int, meter: str = "hexameter") -> int:
        """Scan a verse document's lines, storing each segment's metrical pattern.

        ``meter`` is 'hexameter' (default), 'pentameter', or 'elegiac' (couplets
        alternating hexameter/pentameter). Returns the count of lines that scanned."""
        doc = self.store.get_document(doc_id)
        if doc is None:
            return 0
        scanned = 0
        for section in sorted(doc.sections, key=lambda s: s.order):
            segs = sorted(section.segments, key=lambda x: x.order)
            if not segs:
                continue
            results = scan_lines([s.latin_text for s in segs],
                                 lang=doc.language, meter=meter)
            updates = [(s.id, r.scansion) for s, r in zip(segs, results) if r.scansion]
            if updates:
                self.store.set_scansions(updates)
                scanned += len(updates)
        return scanned

    # -- read / search -------------------------------------------------------

    def get_document(self, doc_id: int) -> Optional[Document]:
        return self.store.get_document(doc_id)

    def list_documents(self) -> List[Document]:
        return self.store.list_documents()

    def search(self, query: str, k: int = 5, **filters) -> List[SearchHit]:
        return self.search_service.search(query, k=k, **filters)

    def close(self) -> None:
        self.vectors.save(self.index_path)
        self.store.close()


def _segment(text: str, order: int):
    from core.models import Segment
    return Segment(latin_text=text, order=order, embed_text=embedding_text_for(text))
