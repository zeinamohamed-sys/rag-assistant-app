"""Pydantic request/response schemas for the /query endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Incoming request body for POST /query."""

    question: str = Field(..., description="The user's natural-language question.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional override for the number of chunks to retrieve.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("question must not be empty")
        return value.strip()


class RetrievedChunk(BaseModel):
    """A single retrieved chunk, returned for transparency/debugging."""

    document: str
    source: str
    page: int | None = None
    chunk_id: str
    text: str
    distance: float | None = None


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str
    sources: list[str]
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    grounded: bool = True


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    vector_store_loaded: bool
    collection_count: int | None = None
