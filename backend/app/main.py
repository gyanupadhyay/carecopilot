"""FastAPI application factory.

Kept as a factory rather than a module-level ``app`` object so that tests
can build an instance with their own settings and so that nothing connects
to a database at import time.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.routes import api_router
from app.auth.demo import DEMO_DISCLAIMER
from app.config import settings
from app.db.session import dispose_engines
from app.llm.factory import dispose_llm
from app.observability.logging import configure_logging, get_logger
from app.observability.middleware import RequestContextMiddleware
from app.rag.embeddings import get_embedder

log = get_logger(__name__)

DESCRIPTION = f"""
{DEMO_DISCLAIMER}

CareCopilot answers questions about a synthetic patient record by routing
each question to the narrowest capable mechanism: a typed tool for known
operations, retrieval over clinical notes for unstructured questions, and
validated read-only SQL for open-ended analytics.
"""


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    log.info(
        "app.startup",
        environment=settings.environment,
        vector_backend=settings.vector_backend,
    )

    # Load the embedding model before accepting traffic. It is ~2s and, on a
    # cold machine, a download; paying it here means "the server is up" also
    # means "the server is ready", instead of the first user absorbing it.
    try:
        embedder = get_embedder()
        await embedder.embed_query("warm up")
        log.info("app.embedder_ready", model=embedder.model_name)
    except Exception:
        # A missing model must not stop the server: every non-RAG route
        # still works, and retrieval will report the failure per request.
        log.exception("app.embedder_unavailable")

    try:
        yield
    finally:
        # Pooled connections outlive the event loop otherwise, which shows
        # up as "Event loop is closed" noise on every reload.
        await dispose_llm()
        await dispose_engines()
        log.info("app.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="CareCopilot API",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
        # Interactive docs are a development convenience, not something a
        # deployed demo needs to publish.
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.environment != "production" else None,
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix="/api")
    return app


app = create_app()
