"""
Generation service.

Builds a strict, document-grounded prompt from retrieved chunks and calls
a local Ollama model to generate the final answer. The model is never
allowed to answer purely from its own internal knowledge — the prompt
explicitly instructs it to say so when the context is insufficient.
"""

from __future__ import annotations

import logging

import ollama

from app.core.config import settings
from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are DataMind, a document-grounded AI assistant for a Data Analysis \
and AI learning platform.

Answer the user's question using ONLY the provided context below. Do not use unsupported \
outside knowledge, and do not invent facts, statistics, or citations.

If the context does not contain enough information to answer the question, say clearly \
that the available documents do not provide enough information to answer, rather than \
guessing.

When you do answer from the context, keep the answer concise and clear enough for a \
student audience. At the end of your answer, list the source document(s) and page \
number(s) you used, in the format: (source: <document>, page <page>)."""


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama service cannot be reached or errors out."""


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        page_str = f", page {chunk.page}" if chunk.page is not None else ""
        parts.append(f"[Context {i}] (source: {chunk.source}{page_str})\n{chunk.text}")
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Build the final grounded user prompt (context + question)."""
    if not chunks:
        context_block = "(No relevant context was retrieved from the documents.)"
    else:
        context_block = _build_context_block(chunks)

    return (
        f"Context from the document corpus:\n\n{context_block}\n\n"
        f"---\n\nUser question: {question}\n\nAnswer:"
    )


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Call the local Ollama model with a grounded prompt and return its answer."""
    user_prompt = build_prompt(question, chunks)

    try:
        client = ollama.Client(host=settings.ollama_base_url)
        response = client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = response["message"]["content"]
        return answer.strip()
    except Exception as exc:  # noqa: BLE001 - convert any Ollama/client error uniformly
        logger.error("Ollama generation failed: %s", exc)
        raise OllamaUnavailableError(
            f"Could not get a response from Ollama (model='{settings.ollama_model}', "
            f"base_url='{settings.ollama_base_url}'). Is Ollama running?"
        ) from exc


def format_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """Build a de-duplicated, human-readable list of 'file.pdf - page N' strings."""
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        page_str = f" - page {chunk.page}" if chunk.page is not None else ""
        label = f"{chunk.source}{page_str}"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources
