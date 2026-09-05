"""
Application configuration.

All configurable values are read from environment variables (with sensible
defaults for local development) using pydantic-settings. Nothing here is
hard-coded that should differ between environments (URLs, model names,
paths, etc.).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/app/core -> backend/app -> backend -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Central application settings, loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App metadata ---
    app_name: str = "DataMind RAG Assistant API"
    app_version: str = "1.0.0"

    # --- Ollama ---
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    # --- CORS ---
    frontend_origin: str = "http://localhost:8501"

    # --- Vector store / embeddings ---
    vector_store_path: str = "../data/vector_store"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "datamind_documents"
    top_k: int = 5

    # --- Logging ---
    log_level: str = "INFO"

    def resolved_vector_store_path(self) -> Path:
        """Resolve vector_store_path relative to the project root if needed."""
        p = Path(self.vector_store_path)
        if p.is_absolute():
            return p
        return (PROJECT_ROOT / "backend" / p).resolve()


settings = Settings()
