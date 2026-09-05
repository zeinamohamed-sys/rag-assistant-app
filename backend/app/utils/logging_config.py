"""Application-wide logging configuration."""

from __future__ import annotations

import logging
import sys

from app.core.config import settings


def configure_logging() -> None:
    """Configure root logging once at application startup.

    Never logs secrets (API keys, .env contents). Only logs operational
    events: startup, vector store loading, queries received, retrieval
    counts, and upstream (Ollama) errors.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if configure_logging() is called more than once
    # (e.g. during tests with --reload style workflows).
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Quiet down noisy third-party libraries a bit.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
