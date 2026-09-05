"""API routes for /health and /query."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.query import HealthResponse, QueryRequest, QueryResponse, RetrievedChunk
from app.services import generation, retrieval

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Simple liveness/readiness check, including vector store status."""
    retriever = retrieval.get_retriever()
    if retriever is None:
        return HealthResponse(status="ok", vector_store_loaded=False, collection_count=None)
    return HealthResponse(
        status="ok",
        vector_store_loaded=True,
        collection_count=retriever.count(),
    )


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query(request: QueryRequest) -> QueryResponse:
    """Answer a question using retrieval-augmented generation over the document corpus."""
    logger.info("Query received: %r", request.question[:120])

    retriever = retrieval.get_retriever()
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="The vector store is not loaded. Run scripts/build_vector_store.py and restart the API.",
        )

    try:
        chunks = retriever.retrieve(request.question, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail="Retrieval failed unexpectedly.") from exc

    try:
        answer = generation.generate_answer(request.question, chunks)
    except generation.OllamaUnavailableError as exc:
        logger.error("Generation unavailable: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    sources = generation.format_sources(chunks)
    grounded = len(chunks) > 0

    return QueryResponse(
        answer=answer,
        sources=sources,
        retrieved_chunks=[
            RetrievedChunk(
                document=c.document,
                source=c.source,
                page=c.page,
                chunk_id=c.chunk_id,
                text=c.text,
                distance=c.distance,
            )
            for c in chunks
        ],
        grounded=grounded,
    )
