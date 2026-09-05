"""FastAPI application entrypoint for DataMind RAG Assistant."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.services import retrieval
from app.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources (embedding model + ChromaDB collection) ONCE at startup."""
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    try:
        retrieval.load_retriever()
        logger.info("Vector store and embedding model loaded successfully.")
    except retrieval.VectorStoreNotFoundError as exc:
        # Do not crash the app: allow /health to report the problem clearly,
        # and let the user build the vector store, then restart.
        logger.warning("Vector store not available at startup: %s", exc)

    yield

    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
