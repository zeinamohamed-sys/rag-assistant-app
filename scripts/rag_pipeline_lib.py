"""
scripts/rag_pipeline_lib.py

Shared, reusable pipeline functions for the DataMind RAG Assistant:
PDF loading & inspection -> cleaning -> chunking -> embeddings -> ChromaDB.

Both scripts/build_vector_store.py and notebooks/rag_pipeline.ipynb import
this module, so the indexing logic lives in exactly one place.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Config defaults (also mirrored in data/vector_store/config.json after a build)
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE_WORDS = 800  # approx. tokens via a word-count proxy, see chunk_text()
DEFAULT_CHUNK_OVERLAP_WORDS = 120
DEFAULT_TOP_K = 5
DEFAULT_COLLECTION_NAME = "datamind_documents"


# ---------------------------------------------------------------------------
# 1) Load & inspect
# ---------------------------------------------------------------------------
@dataclass
class DocumentStats:
    filename: str
    num_pages: int
    char_count: int
    extraction_ok: bool
    likely_needs_ocr: bool


@dataclass
class PageText:
    document: str
    page: int  # 1-indexed
    text: str


def load_documents(documents_dir: Path) -> tuple[list[PageText], list[DocumentStats]]:
    """Load every PDF in `documents_dir`, extracting text page by page.

    Returns (all_pages, stats) where `all_pages` is a flat list of PageText
    across every document, and `stats` is one DocumentStats entry per file
    (for the inspection report).
    """
    all_pages: list[PageText] = []
    stats: list[DocumentStats] = []

    pdf_paths = sorted(documents_dir.glob("*.pdf"))
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(str(pdf_path))
            num_pages = len(reader.pages)
            doc_char_count = 0
            extraction_ok = True
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                doc_char_count += len(text)
                all_pages.append(PageText(document=pdf_path.name, page=i, text=text))
            likely_needs_ocr = num_pages > 0 and (doc_char_count / max(num_pages, 1)) < 20
        except Exception:  # noqa: BLE001 - a single bad PDF should not stop the pipeline
            num_pages = 0
            doc_char_count = 0
            extraction_ok = False
            likely_needs_ocr = False

        stats.append(
            DocumentStats(
                filename=pdf_path.name,
                num_pages=num_pages,
                char_count=doc_char_count,
                extraction_ok=extraction_ok,
                likely_needs_ocr=likely_needs_ocr,
            )
        )

    return all_pages, stats


# ---------------------------------------------------------------------------
# 2) Cleaning
# ---------------------------------------------------------------------------
_MULTI_BLANK_LINES = re.compile(r"\n\s*\n\s*\n+")
_MULTI_SPACES = re.compile(r"[ \t]+")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")


def clean_text(text: str) -> str:
    """Light-touch cleaning that preserves technical content.

    - Normalizes line endings.
    - Collapses runs of horizontal whitespace to a single space.
    - Collapses 3+ blank lines down to a single blank line.
    - Strips trailing whitespace on each line.
    - Does NOT lowercase, strip punctuation, or remove numbers/symbols,
      since those are meaningful in code samples, function names, and
      formulas.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE.sub("\n", text)
    text = _MULTI_SPACES.sub(" ", text)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 3) Chunking
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    text: str
    document: str
    source: str
    page: int
    chunk_id: str


def chunk_text(
    pages: list[PageText],
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """Chunk cleaned page text using a word-count approximation of tokens.

    We use *whitespace-split word count* rather than a real tokenizer as an
    approximation for chunk size: for English technical prose, word count
    and token count track each other closely enough (~1 word ~= 1.3 tokens)
    that this keeps chunks in a sane, predictable size range without adding
    a tokenizer dependency to the pipeline.

    Chunking is done per-page so that every chunk can be attributed to a
    single page for citations. Long pages are split into multiple
    overlapping chunks; short pages become a single chunk.
    """
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}

    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue

        words = cleaned.split(" ")
        step = max(chunk_size_words - chunk_overlap_words, 1)

        start = 0
        while start < len(words):
            end = min(start + chunk_size_words, len(words))
            chunk_words = words[start:end]
            chunk_str = " ".join(chunk_words).strip()

            if chunk_str:
                counters[page.document] = counters.get(page.document, 0) + 1
                chunk_id = f"{Path(page.document).stem}_{counters[page.document]:04d}"
                chunks.append(
                    Chunk(
                        text=chunk_str,
                        document=page.document,
                        source=page.document,
                        page=page.page,
                        chunk_id=chunk_id,
                    )
                )

            if end == len(words):
                break
            start += step

    return chunks


# ---------------------------------------------------------------------------
# 4) Embeddings + 5) ChromaDB
# ---------------------------------------------------------------------------
def embed_chunks(chunks: list[Chunk], model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Embed all chunk texts once, using a local Sentence Transformers model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode([c.text for c in chunks], show_progress_bar=True)
    return embeddings, model


def build_chroma_collection(
    chunks: list[Chunk],
    embeddings,
    vector_store_path: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """Create (or replace) a persistent ChromaDB collection with the given chunks."""
    import chromadb

    vector_store_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(vector_store_path))

    # Replace any existing collection so re-runs are idempotent.
    try:
        client.delete_collection(collection_name)
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass

    collection = client.create_collection(collection_name)

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c.text for c in chunks],
        metadatas=[
            {"source": c.source, "page": c.page, "chunk_id": c.chunk_id, "document": c.document}
            for c in chunks
        ],
    )
    return client, collection


def write_config(
    vector_store_path: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    num_chunks: int | None = None,
    num_documents: int | None = None,
) -> Path:
    """Save pipeline configuration/metadata alongside the vector store."""
    config = {
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "collection_name": collection_name,
        "num_chunks": num_chunks,
        "num_documents": num_documents,
    }
    config_path = vector_store_path / "config.json"
    vector_store_path.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config_path


def retrieve(collection, embed_model, question: str, top_k: int = DEFAULT_TOP_K):
    """Embed `question` and query the collection for the top_k most similar chunks."""
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results
