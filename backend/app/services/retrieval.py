"""
Retrieval service.

Loads the persistent ChromaDB collection and the local embedding model
ONCE (via `load_retriever`) and reuses them for every query. This module
never rebuilds the vector store — that is the job of
`scripts/build_vector_store.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    document: str
    source: str
    page: int | None
    chunk_id: str
    text: str
    distance: float | None


class VectorStoreNotFoundError(RuntimeError):
    """Raised when the persistent vector store does not exist on disk."""


class Retriever:
    """Wraps a persistent ChromaDB collection + embedding model."""

    def __init__(self, vector_store_path: Path, collection_name: str, embedding_model_name: str):
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        if not vector_store_path.exists():
            raise VectorStoreNotFoundError(
                f"Vector store not found at '{vector_store_path}'. "
                "Run `python scripts/build_vector_store.py` first."
            )

        logger.info("Loading embedding model '%s'...", embedding_model_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)

        logger.info("Loading ChromaDB collection '%s' from '%s'...", collection_name, vector_store_path)
        self.client = chromadb.PersistentClient(path=str(vector_store_path))
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
            raise VectorStoreNotFoundError(
                f"Collection '{collection_name}' does not exist in the vector store at "
                f"'{vector_store_path}'. Run `python scripts/build_vector_store.py` first."
            ) from exc

        logger.info("Retriever ready. Collection has %d chunks.", self.collection.count())

    def count(self) -> int:
        return self.collection.count()

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Embed `question` and return the top_k most similar chunks with metadata."""
        k = top_k or settings.top_k
        query_embedding = self.embedding_model.encode([question]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(documents)

        for text, meta, distance in zip(documents, metadatas, distances):
            chunks.append(
                RetrievedChunk(
                    document=meta.get("document", "unknown"),
                    source=meta.get("source", meta.get("document", "unknown")),
                    page=meta.get("page"),
                    chunk_id=str(meta.get("chunk_id", "")),
                    text=text,
                    distance=distance,
                )
            )

        logger.info("Retrieved %d chunks for question: %r", len(chunks), question[:80])
        return chunks


_retriever: Retriever | None = None


def load_retriever() -> Retriever:
    """Load (or return the cached) Retriever singleton. Call once at startup."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever(
            vector_store_path=settings.resolved_vector_store_path(),
            collection_name=settings.collection_name,
            embedding_model_name=settings.embedding_model,
        )
    return _retriever


def get_retriever() -> Retriever | None:
    """Return the cached retriever without raising, for health checks etc."""
    return _retriever


def set_retriever_for_testing(retriever: Retriever | None) -> None:
    """Test hook to inject a fake/mock retriever."""
    global _retriever
    _retriever = retriever
