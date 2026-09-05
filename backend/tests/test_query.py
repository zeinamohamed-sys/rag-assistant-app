"""
Tests for the /health and /query endpoints.

Ollama is NOT required to run these tests: the generation layer is mocked,
since unit tests should not depend on an external LLM service being up.
A real, non-mocked retriever (backed by a tiny temporary ChromaDB) is used
so retrieval logic is genuinely exercised end-to-end.

The embedding model itself is swapped for a small deterministic local
stand-in (`_FakeEmbedder`) purely so the test suite can run fully offline
in CI/sandbox environments without needing to download weights from
huggingface.co. In normal local development (with internet access), the
real app always uses `sentence-transformers/all-MiniLM-L6-v2` as
configured in app/core/config.py — this substitution is test-only.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import generation, retrieval


class _FakeEmbedder:
    """Deterministic bag-of-words style embedder, used only in tests.

    Not semantically as strong as a real sentence-transformer, but good
    enough to prove retrieval returns the more lexically-similar chunk for
    the two clearly distinct test documents below, without any network
    access.
    """

    DIM = 512

    def encode(self, texts: list[str]):
        vectors = []
        for text in texts:
            vec = np.zeros(self.DIM, dtype=float)
            for raw_word in text.lower().split():
                word = "".join(ch for ch in raw_word if ch.isalnum())
                if not word:
                    continue
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % self.DIM] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors)


@pytest.fixture(scope="module")
def temp_vector_store():
    """Build a tiny real ChromaDB collection with one document for testing."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="datamind_test_vs_"))

    import chromadb

    model = _FakeEmbedder()
    client = chromadb.PersistentClient(path=str(tmp_dir))
    collection = client.create_collection("datamind_documents")

    texts = [
        "A Pandas DataFrame is a two-dimensional, size-mutable, tabular data "
        "structure with labeled rows and columns, similar to a spreadsheet.",
        "Exploratory data analysis (EDA) is the process of visually and "
        "statistically summarizing a dataset to understand its main characteristics.",
    ]
    embeddings = model.encode(texts).tolist()
    collection.add(
        ids=["chunk_0001", "chunk_0002"],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"source": "pandas_notes.pdf", "page": 1, "chunk_id": "chunk_0001", "document": "pandas_notes.pdf"},
            {"source": "eda_notes.pdf", "page": 3, "chunk_id": "chunk_0002", "document": "eda_notes.pdf"},
        ],
    )

    test_retriever = retrieval.Retriever.__new__(retrieval.Retriever)
    test_retriever.vector_store_path = tmp_dir
    test_retriever.collection_name = "datamind_documents"
    test_retriever.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    test_retriever.embedding_model = model
    test_retriever.client = client
    test_retriever.collection = collection

    retrieval.set_retriever_for_testing(test_retriever)

    yield test_retriever

    retrieval.set_retriever_for_testing(None)
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture()
def mock_ollama(monkeypatch):
    """Mock the generation layer so tests never require a live Ollama server."""

    def fake_generate_answer(question: str, chunks):
        if not chunks:
            return "The available documents do not provide enough information to answer this question."
        return (
            f"[MOCKED ANSWER] Based on the retrieved context, here is an answer to: {question}"
        )

    monkeypatch.setattr(generation, "generate_answer", fake_generate_answer)


@pytest.fixture()
def client(temp_vector_store, mock_ollama):
    return TestClient(app)


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vector_store_loaded"] is True
    assert body["collection_count"] == 2


def test_query_happy_path(client):
    response = client.post("/query", json={"question": "What is a Pandas DataFrame?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body and isinstance(body["answer"], str) and len(body["answer"]) > 0
    assert "sources" in body and isinstance(body["sources"], list) and len(body["sources"]) > 0
    assert body["grounded"] is True


def test_query_invalid_empty_question(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_retrieves_relevant_source(client):
    response = client.post("/query", json={"question": "What is exploratory data analysis?", "top_k": 1})
    assert response.status_code == 200
    body = response.json()
    assert any("eda_notes.pdf" in s for s in body["sources"])
