"""
Thin HTTP client for talking to the DataMind FastAPI backend.

The backend base URL is read from the API_BASE_URL environment variable
(via .env) — it is never hard-coded here.
"""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

DEFAULT_TIMEOUT = 120  # seconds; local LLM generation can be slow on CPU


class APIClientError(Exception):
    """Raised when the backend cannot be reached or returns an error."""


def check_health() -> dict | None:
    """Return the backend's /health payload, or None if unreachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def ask_question(question: str, top_k: int | None = None) -> dict:
    """Send a question to POST /query and return the parsed JSON response.

    Raises APIClientError with a friendly message on any failure.
    """
    payload: dict = {"question": question}
    if top_k is not None:
        payload["top_k"] = top_k

    try:
        response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=DEFAULT_TIMEOUT)
    except requests.ConnectionError as exc:
        raise APIClientError(
            "Could not connect to the DataMind API. Make sure the FastAPI backend "
            f"is running at {API_BASE_URL}."
        ) from exc
    except requests.Timeout as exc:
        raise APIClientError(
            "The request to the DataMind API timed out. The local LLM may be slow "
            "or unresponsive — please try again."
        ) from exc

    if response.status_code == 503:
        raise APIClientError(
            "The document vector store is not loaded on the backend yet. "
            "Run scripts/build_vector_store.py and restart the API."
        )
    if response.status_code == 502:
        raise APIClientError(
            "The AI service is currently unavailable. Please make sure Ollama "
            "and the backend are running."
        )
    if response.status_code == 422:
        raise APIClientError("Please enter a valid, non-empty question.")
    if not response.ok:
        raise APIClientError(f"Unexpected error from the API (status {response.status_code}).")

    return response.json()
