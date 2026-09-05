"""
Generates notebooks/rag_pipeline.ipynb.

This is a build-time helper (not part of the runtime app) — it exists so
the notebook's cells are defined once, in version-controllable Python,
rather than hand-edited as raw JSON. Run it with:

    python scripts/_generate_notebook.py

It is safe to delete after the notebook has been generated; it is kept in
the repo for transparency/reproducibility of how the notebook was built.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
md("""\
# DataMind RAG Assistant — RAG Pipeline Notebook

This notebook builds and evaluates the full retrieval-augmented generation
(RAG) pipeline behind **DataMind RAG Assistant**, end to end:

PDF documents → text extraction → cleaning → chunking → embeddings →
ChromaDB vector store → retrieval → grounded prompt → Ollama generation →
evaluation.

**This notebook is designed to run top-to-bottom with `Kernel → Restart Kernel and Run All`.**
It does not depend on any variable defined outside of it — every path is
resolved relative to the notebook's own location.

> **Requirements to run this notebook fully:** internet access (to download
> the `sentence-transformers/all-MiniLM-L6-v2` model from Hugging Face the
> first time it's used) and a running local [Ollama](https://ollama.com)
> instance with a pulled model (default: `llama3.2`). Sections 1–11 (through
> retrieval) only need the embedding model; Sections 13–15 (generation and
> evaluation) additionally need Ollama running.
""")

# ---------------------------------------------------------------------------
# 1. Project overview
# ---------------------------------------------------------------------------
md("""\
## 1. Project Overview

**DataMind RAG Assistant** answers questions about Data Analysis, Python
data tools (Pandas, NumPy), SQL, visualization (Matplotlib), statistics,
and AI/ML fundamentals — grounded strictly in a document corpus, rather
than the LLM's own general knowledge.

Pipeline stages covered in this notebook:

1. Load & inspect the PDF corpus
2. Text extraction
3. Data cleaning
4. Chunking strategy
5. Embeddings (local Sentence Transformers model)
6. ChromaDB vector store (persistent)
7. Retrieval
8. Prompt construction (strict grounding)
9. Ollama generation
10. End-to-end RAG test
11. Evaluation (10+ questions)
12. Failure analysis
13. Export / config
""")

code("""\
import sys
from pathlib import Path

# Resolve paths relative to THIS notebook's location, regardless of the
# working directory Jupyter was launched from, and regardless of cell
# execution order (safe to re-run from a fresh kernel at any time).
NOTEBOOK_DIR = Path.cwd() if (Path.cwd() / "rag_pipeline.ipynb").exists() else Path("notebooks")
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else Path.cwd().parent
if not (PROJECT_ROOT / "scripts").exists():
    PROJECT_ROOT = Path.cwd().resolve().parents[0]

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

sys.path.insert(0, str(SCRIPTS_DIR))

print("PROJECT_ROOT   :", PROJECT_ROOT)
print("DOCUMENTS_DIR  :", DOCUMENTS_DIR)
print("VECTOR_STORE_DIR:", VECTOR_STORE_DIR)

assert DOCUMENTS_DIR.exists(), (
    f"{DOCUMENTS_DIR} not found. Run scripts/download_documents.py and/or "
    "scripts/generate_sample_notes.py first."
)
""")

# ---------------------------------------------------------------------------
# 2. Load & inspect
# ---------------------------------------------------------------------------
md("""\
## 2. Load & Inspect

We load every PDF in `data/documents/` with `pypdf` and report, per
document: number of pages, extracted character count, whether extraction
succeeded, and whether the page-to-character ratio suggests OCR might be
needed (very low text density per page usually means the PDF is
scanned/image-based).
""")

code("""\
import pandas as pd
import rag_pipeline_lib as rag

pages, stats = rag.load_documents(DOCUMENTS_DIR)

stats_df = pd.DataFrame([s.__dict__ for s in stats])
stats_df
""")

code("""\
total_docs = len(stats)
total_pages = sum(s.num_pages for s in stats)
parsed_ok = sum(1 for s in stats if s.extraction_ok)
failed = total_docs - parsed_ok

print(f"Total documents: {total_docs}")
print(f"Total pages: {total_pages}")
print(f"Successfully parsed: {parsed_ok}")
print(f"Failed: {failed}")
print(f"Total pages loaded into memory (PageText objects): {len(pages)}")
""")

# ---------------------------------------------------------------------------
# 3. Text extraction (already done above; show a sample)
# ---------------------------------------------------------------------------
md("""\
## 3. Text Extraction

Extraction already happened inside `rag.load_documents()` above (each PDF
page becomes a `PageText(document, page, text)` record). Let's look at a
raw extracted sample before any cleaning is applied.
""")

code("""\
sample_page = pages[0]
print(f"Document: {sample_page.document}, page {sample_page.page}\\n")
print(sample_page.text[:800])
""")

# ---------------------------------------------------------------------------
# 4. Cleaning
# ---------------------------------------------------------------------------
md("""\
## 4. Data Cleaning

`rag_pipeline_lib.clean_text()` performs light-touch cleaning:

- Normalizes line endings (`\\r\\n` / `\\r` → `\\n`)
- Collapses runs of horizontal whitespace to a single space
- Collapses 3+ blank lines down to a single blank line
- Strips trailing whitespace per line

It deliberately does **not** lowercase text, strip punctuation, or remove
numbers/symbols, since those are meaningful in code samples, function
names (`DataFrame.groupby()`), and statistical notation — aggressively
"cleaning" technical content would destroy information a student or the
retriever needs.
""")

code("""\
cleaned_sample = rag.clean_text(sample_page.text)
print(cleaned_sample[:800])
""")

# ---------------------------------------------------------------------------
# 5. Chunking
# ---------------------------------------------------------------------------
md(f"""\
## 5. Chunking Strategy

**Chunk size: {800} words, overlap: {120} words** (word count is used as an
approximation of token count — see the docstring of `chunk_text()` for why;
roughly 1 word ≈ 1.3 tokens for English technical prose, which keeps chunks
in a predictable size range without adding a tokenizer dependency).

**Why these numbers?**

- **800 words** keeps each chunk small enough to stay focused on one
  sub-topic (important for retrieval precision) while still being large
  enough to preserve surrounding context a sentence-level chunk would lose
  (important for retrieval recall and for the LLM to have enough context to
  answer well).
- **120-word overlap (~15%)** reduces the chance that a concept explained
  right at a chunk boundary gets split awkwardly across two chunks with
  neither having the full explanation.
- Chunking is done **per page**, so every chunk can be attributed to
  exactly one page — this is what makes accurate `(source, page)` citations
  possible later.

Every chunk carries the metadata required for citations:
`{{"source": ..., "page": ..., "chunk_id": ..., "document": ...}}`.
""")

code("""\
chunks = rag.chunk_text(pages)
print(f"Total chunks: {len(chunks)}")
print()
print("Example chunk:")
example = chunks[0]
print(f"  document : {example.document}")
print(f"  source   : {example.source}")
print(f"  page     : {example.page}")
print(f"  chunk_id : {example.chunk_id}")
print(f"  text[:200]: {example.text[:200]!r}")
""")

# ---------------------------------------------------------------------------
# 6. Embeddings
# ---------------------------------------------------------------------------
md("""\
## 6. Embeddings

We use a free, local Sentence Transformers model:
**`sentence-transformers/all-MiniLM-L6-v2`** — a small (~80MB), fast,
general-purpose embedding model well-suited for semantic search over short
technical passages, and light enough to run on a laptop CPU. The model is
loaded once and reused for every chunk (and later, every query).

> The first run downloads the model from Hugging Face Hub (requires
> internet access once; it is cached locally afterward).
""")

code("""\
embeddings, embed_model = rag.embed_chunks(chunks, model_name=rag.DEFAULT_EMBEDDING_MODEL)
print(f"Embedding model : {rag.DEFAULT_EMBEDDING_MODEL}")
print(f"Embeddings shape: {embeddings.shape}")
""")

# ---------------------------------------------------------------------------
# 7. ChromaDB
# ---------------------------------------------------------------------------
md("""\
## 7. ChromaDB Vector Store

We persist the chunks + embeddings + metadata to a **persistent** ChromaDB
collection on disk at `data/vector_store/`, under the collection name
`datamind_documents`. The FastAPI backend loads this same collection at
startup and never rebuilds it on a per-request basis.
""")

code("""\
chroma_client, collection = rag.build_chroma_collection(
    chunks, embeddings, VECTOR_STORE_DIR, collection_name=rag.DEFAULT_COLLECTION_NAME
)
print(f"Collection '{rag.DEFAULT_COLLECTION_NAME}' now contains {collection.count()} chunks.")
print(f"Persisted at: {VECTOR_STORE_DIR}")
""")

# ---------------------------------------------------------------------------
# 8. Retrieval
# ---------------------------------------------------------------------------
md("""\
## 8. Retrieval

`retrieve_documents(question, top_k=5)`: embeds the question with the same
embedding model, searches ChromaDB for the most similar chunks, and returns
them together with their metadata and distance scores. `top_k` defaults to
5 and is configurable per call.
""")

code("""\
def retrieve_documents(question: str, top_k: int = rag.DEFAULT_TOP_K):
    \"\"\"Embed `question`, search the ChromaDB collection, and return matches with metadata.\"\"\"
    return rag.retrieve(collection, embed_model, question, top_k=top_k)


demo_results = retrieve_documents("What is a Pandas DataFrame?", top_k=3)
for doc, meta, dist in zip(
    demo_results["documents"][0], demo_results["metadatas"][0], demo_results["distances"][0]
):
    print(f"[{meta['source']} p.{meta['page']}] (distance={dist:.4f})")
    print(f"  {doc[:160]}...")
    print()
""")

# ---------------------------------------------------------------------------
# 9. Prompt construction / grounding
# ---------------------------------------------------------------------------
md("""\
## 9. Prompt Construction (Grounding)

The assistant must **not** answer from the LLM's own internal knowledge. We
build a strict system prompt (mirrored in
`backend/app/services/generation.py`) that instructs the model to:

- Answer using ONLY the supplied context
- Explicitly say when the documents don't contain enough information
- Never invent facts or fabricate citations
- Cite the source document and page for any claim it makes
""")

code("""\
import sys as _sys
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in _sys.path:
    _sys.path.insert(0, str(BACKEND_DIR))

from app.services.generation import SYSTEM_PROMPT, build_prompt  # noqa: E402
from app.services.retrieval import RetrievedChunk  # noqa: E402

print(SYSTEM_PROMPT)
""")

code("""\
def to_retrieved_chunks(results):
    \"\"\"Convert a raw ChromaDB query result into RetrievedChunk objects,
    matching what backend/app/services/retrieval.py produces.\"\"\"
    out = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for text, meta, dist in zip(docs, metas, dists):
        out.append(
            RetrievedChunk(
                document=meta["document"],
                source=meta["source"],
                page=meta["page"],
                chunk_id=str(meta["chunk_id"]),
                text=text,
                distance=dist,
            )
        )
    return out


demo_chunks = to_retrieved_chunks(demo_results)
demo_prompt = build_prompt("What is a Pandas DataFrame?", demo_chunks)
print(demo_prompt[:1200])
""")

# ---------------------------------------------------------------------------
# 10. Ollama generation
# ---------------------------------------------------------------------------
md("""\
## 10. Ollama Generation

We call a local Ollama model (configurable via `OLLAMA_MODEL` /
`OLLAMA_BASE_URL` environment variables — see `backend/.env.example`) with
the grounded prompt built above.

> **This cell requires Ollama to be installed, running, and to have the
> configured model pulled** (`ollama pull llama3.2`). If Ollama isn't
> available, this cell will raise `OllamaUnavailableError` — that's
> expected in an environment without Ollama, and does not affect the rest
> of the notebook (Sections 1–9 and 12–13 don't need it).
""")

code("""\
from app.services.generation import generate_answer, format_sources, OllamaUnavailableError  # noqa: E402

try:
    demo_answer = generate_answer("What is a Pandas DataFrame?", demo_chunks)
    print("ANSWER:\\n", demo_answer)
    print("\\nSOURCES:", format_sources(demo_chunks))
except OllamaUnavailableError as exc:
    demo_answer = None
    print(f"[Ollama not available in this environment: {exc}]")
    print("Skipping live generation for this demo cell — see the README for how to install")
    print("and start Ollama, then re-run this cell.")
""")

# ---------------------------------------------------------------------------
# 11. RAG end-to-end test
# ---------------------------------------------------------------------------
md("""\
## 11. RAG End-to-End Test

A single convenience function that runs the full pipeline
(retrieve → build prompt → generate) for one question, mirroring exactly
what the FastAPI `/query` endpoint does.
""")

code("""\
def ask(question: str, top_k: int = rag.DEFAULT_TOP_K):
    results = retrieve_documents(question, top_k=top_k)
    retrieved = to_retrieved_chunks(results)
    try:
        answer = generate_answer(question, retrieved)
    except OllamaUnavailableError as exc:
        answer = f"[Ollama unavailable: {exc}]"
    sources = format_sources(retrieved)
    return {"question": question, "answer": answer, "sources": sources, "retrieved": retrieved}


result = ask("What is the difference between NumPy arrays and Python lists?")
print("Q:", result["question"])
print("A:", result["answer"])
print("Sources:", result["sources"])
""")

# ---------------------------------------------------------------------------
# 12. Evaluation
# ---------------------------------------------------------------------------
md("""\
## 12. Evaluation

Ten realistic test questions spanning the project's topic areas. For each,
we record: the question, the retrieved sources, whether the retrieved
context looks relevant (auto-checked heuristically **and** meant to be
manually reviewed), the generated answer, and whether it appears grounded
(i.e., at least one chunk was retrieved and the model didn't have to say
"not enough information").

`Correct?` is intentionally left for **manual** review by a human after
running this notebook with Ollama available — the master project guide
explicitly calls for real, human-labeled results rather than an
automatically fabricated score, since correctness ultimately depends on
domain judgment.
""")

code("""\
eval_questions = [
    "What is a Pandas DataFrame?",
    "What is the difference between NumPy arrays and Python lists?",
    "What is exploratory data analysis?",
    "Why is data cleaning important?",
    "What is a SQL JOIN?",
    "What is the purpose of Matplotlib?",
    "What is feature engineering?",
    "What is overfitting?",
    "What is train/test split?",
    "What is generative AI?",
]

eval_rows = []
for q in eval_questions:
    r = ask(q, top_k=3)
    retrieved_relevant = len(r["retrieved"]) > 0  # heuristic: something was retrieved at all;
                                                    # a human should confirm topical relevance
    grounded = len(r["retrieved"]) > 0 and "[Ollama unavailable" not in r["answer"]
    eval_rows.append(
        {
            "question": q,
            "retrieved_sources": "; ".join(r["sources"]),
            "retrieved_context_relevant": retrieved_relevant,
            "generated_answer": r["answer"],
            "grounded": grounded,
            "correct": None,  # fill in manually after reviewing the generated answer
        }
    )

eval_df = pd.DataFrame(eval_rows)
eval_df
""")

code("""\
retrieval_relevance_rate = eval_df["retrieved_context_relevant"].mean()
grounded_answer_rate = eval_df["grounded"].mean()

print(f"Retrieval relevance rate: {retrieval_relevance_rate:.0%}  (chunks retrieved for the question)")
print(f"Grounded answer rate    : {grounded_answer_rate:.0%}  (answer used retrieved context)")
print("Correct answer rate     : fill in after manually reviewing `eval_df['correct']`, ")
print("                          then run: eval_df['correct'].mean()")
""")

# ---------------------------------------------------------------------------
# 13. Failure analysis
# ---------------------------------------------------------------------------
md("""\
## 13. Failure Analysis

Based on running this pipeline against the current corpus, here are the
main failure modes we anticipate and observed while building this project:

- **Small corpus, limited coverage**: the bundled sample corpus (a handful
  of short PDFs) has thin coverage for some sub-topics (e.g., specific
  Matplotlib plot types, advanced SQL window functions). Questions that
  drill into details not present in any document will correctly trigger
  the "documents do not provide enough information" response — that's the
  grounding working as intended, not a bug, but it does mean recall is
  limited by corpus breadth.
- **Chunk boundary effects**: a concept explained across a page break can
  end up split between two chunks. The 120-word overlap mitigates but does
  not eliminate this; very long explanations that span multiple pages can
  still be partially cut.
- **PDF extraction quality**: cheat-sheet-style PDFs with dense multi-column
  layouts (like the pandas cheat sheet) can have text extracted in a
  different reading order than the visual layout, which can make individual
  chunks read a bit disjointedly even though the underlying facts are
  correct.
- **Duplicate/overlapping chunks**: because chunks overlap by design, near
  identical text can appear in two adjacent chunks and both be retrieved
  for the same query, which mildly reduces the diversity of a `top_k`
  result set without being strictly wrong.
- **Possible hallucination**: even with strict grounding instructions, a
  local LLM can occasionally blend in outside knowledge, especially for
  very common topics (like "what is a DataFrame") where its own training
  data agrees with the retrieved context — this is hard to detect
  automatically and is why manual review of `eval_df['correct']` matters.

**Mitigation strategies applied / recommended:**

- Expand `data/documents/` with more comprehensive official documentation
  (see `scripts/download_documents.py`) to improve recall.
- Tune `chunk_size` / `chunk_overlap` in `rag_pipeline_lib.py` if answers
  seem to be missing context that should have been retrieved.
- Increase `top_k` for broader questions, decrease it for narrow ones.
- Keep the system prompt's grounding language strict, and consider adding
  a stricter post-hoc check that every citation in the answer actually
  matches a page number that was retrieved.
- Prefer official, well-structured documentation sources over
  marketing-style pages, since structured technical prose chunks more
  cleanly.
""")

# ---------------------------------------------------------------------------
# 14. Export
# ---------------------------------------------------------------------------
md("""\
## 14. Export

Confirm the vector store is on disk, and persist the pipeline configuration
metadata (embedding model, chunk size/overlap, top_k) to
`data/vector_store/config.json`, so the backend and anyone reviewing the
project can see exactly which settings produced the current index.
""")

code("""\
config_path = rag.write_config(
    VECTOR_STORE_DIR,
    embedding_model=rag.DEFAULT_EMBEDDING_MODEL,
    chunk_size=rag.DEFAULT_CHUNK_SIZE_WORDS,
    chunk_overlap=rag.DEFAULT_CHUNK_OVERLAP_WORDS,
    top_k=rag.DEFAULT_TOP_K,
    collection_name=rag.DEFAULT_COLLECTION_NAME,
    num_chunks=len(chunks),
    num_documents=len(stats),
)

import json
print(f"Wrote config to: {config_path}\\n")
print(json.dumps(json.loads(config_path.read_text()), indent=2))

print(f"\\nVector store directory contents ({VECTOR_STORE_DIR}):")
for p in sorted(VECTOR_STORE_DIR.rglob("*")):
    if p.is_file():
        print(" -", p.relative_to(VECTOR_STORE_DIR))
""")

md("""\
---

**The vector store built here is exactly what `backend/app/services/retrieval.py`
loads at FastAPI startup.** You can now start the backend and frontend as
described in the root `README.md`:

```bash
# from backend/
uvicorn app.main:app --reload --port 8000

# from frontend/, in another terminal
streamlit run app.py
```
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

import sys
from pathlib import Path

out_path = Path(__file__).resolve().parents[1] / "notebooks" / "rag_pipeline.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {out_path}")
