"""
scripts/build_vector_store.py

Reproduces the essential pipeline from notebooks/rag_pipeline.ipynb as a
single runnable script:

    PDF -> extract -> clean -> chunk -> embeddings -> ChromaDB

Run this after `python scripts/download_documents.py` (and/or
`python scripts/generate_sample_notes.py`) to (re)build the persistent
vector store the FastAPI backend loads at startup.

Usage:
    python scripts/build_vector_store.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow `import rag_pipeline_lib` whether this script is run from the repo
# root or from inside scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rag_pipeline_lib as rag  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"


def main() -> int:
    print("=" * 70)
    print("DataMind RAG Assistant - Vector Store Builder")
    print("=" * 70)

    if not DOCUMENTS_DIR.exists() or not any(DOCUMENTS_DIR.glob("*.pdf")):
        print(
            f"\nNo PDF files found in {DOCUMENTS_DIR}.\n"
            "Run `python scripts/download_documents.py` and/or "
            "`python scripts/generate_sample_notes.py` first, or place your "
            "own PDFs in that folder."
        )
        return 1

    # --- 1. Load & inspect -------------------------------------------------
    print(f"\n[1/5] Loading PDFs from {DOCUMENTS_DIR} ...")
    t0 = time.time()
    pages, stats = rag.load_documents(DOCUMENTS_DIR)

    total_pages = sum(s.num_pages for s in stats)
    ok_count = sum(1 for s in stats if s.extraction_ok)
    print(f"      Documents found : {len(stats)}")
    print(f"      Total pages     : {total_pages}")
    print(f"      Parsed OK       : {ok_count}")
    print(f"      Failed          : {len(stats) - ok_count}")
    for s in stats:
        flag = "OK" if s.extraction_ok else "FAILED"
        ocr_flag = " (may need OCR)" if s.likely_needs_ocr else ""
        print(f"        - {s.filename}: {s.num_pages} pages, {s.char_count} chars [{flag}]{ocr_flag}")

    # --- 2 & 3. Clean + chunk ----------------------------------------------
    print(f"\n[2/5] Cleaning and chunking (chunk_size={rag.DEFAULT_CHUNK_SIZE_WORDS} words, "
          f"overlap={rag.DEFAULT_CHUNK_OVERLAP_WORDS} words) ...")
    chunks = rag.chunk_text(pages)
    print(f"      Produced {len(chunks)} chunks from {len(stats)} documents.")

    if not chunks:
        print("\nNo text could be extracted/chunked from the PDFs. Aborting.")
        return 1

    # --- 4. Embeddings -------------------------------------------------------
    print(f"\n[3/5] Loading embedding model '{rag.DEFAULT_EMBEDDING_MODEL}' and encoding chunks ...")
    embeddings, _model = rag.embed_chunks(chunks)
    print(f"      Embedded {len(chunks)} chunks -> shape {embeddings.shape}")

    # --- 5. ChromaDB ---------------------------------------------------------
    print(f"\n[4/5] Writing persistent ChromaDB collection to {VECTOR_STORE_DIR} ...")
    _client, collection = rag.build_chroma_collection(chunks, embeddings, VECTOR_STORE_DIR)
    print(f"      Collection '{rag.DEFAULT_COLLECTION_NAME}' now has {collection.count()} chunks.")

    # --- Config ---------------------------------------------------------------
    print("\n[5/5] Writing config.json ...")
    config_path = rag.write_config(
        VECTOR_STORE_DIR,
        num_chunks=len(chunks),
        num_documents=len(stats),
    )
    print(f"      Wrote {config_path}")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"Done in {elapsed:.1f}s. Vector store ready at: {VECTOR_STORE_DIR}")
    print("You can now start the backend: uvicorn app.main:app --reload --port 8000 (from backend/)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
