"""Embedding storage that works with or without the pgvector extension.

The production path is a native ``vector(N)`` column with an HNSW index.
Where pgvector cannot be installed (notably a stock Windows PostgreSQL
build), ``VECTOR_BACKEND=array`` falls back to ``double precision[]`` and
ranks with the ``cc_cosine_distance`` SQL function created in migration
0001. Ranking stays inside the database in both modes — the fallback is a
performance tradeoff (sequential scan, no ANN index), never a correctness
or authorization one.

Callers never branch on the backend: they use :func:`embedding_column` to
declare storage and :func:`distance_expression` to rank.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Float, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql.elements import ColumnElement

from app.config import settings

_PGVECTOR_AVAILABLE = False
try:  # pragma: no cover - depends on the installed environment
    from pgvector.sqlalchemy import Vector as _PgVector

    _PGVECTOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PgVector = None  # type: ignore[assignment]


def using_pgvector() -> bool:
    return settings.vector_backend == "pgvector" and _PGVECTOR_AVAILABLE


def embedding_type(dim: int | None = None) -> Any:
    """Return the SQLAlchemy column type for an embedding vector."""
    dim = dim or settings.embedding_dim
    if using_pgvector():
        return _PgVector(dim)
    return ARRAY(Float(precision=53))


def distance_expression(column: Any, query: list[float]) -> ColumnElement[float]:
    """Cosine *distance* between a stored embedding and a query vector.

    Lower is more similar in both backends, so ``order_by`` is identical.
    """
    if using_pgvector():
        # pgvector's <=> operator, backed by the HNSW index.
        return column.cosine_distance(query)
    return func.cc_cosine_distance(column, func.cast(query, ARRAY(Float(precision=53))))


def similarity_from_distance(distance: float) -> float:
    """Map cosine distance in [0, 2] onto a [0, 1] similarity score."""
    return max(0.0, min(1.0, 1.0 - distance))
