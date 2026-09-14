"""Embedding providers.

The default is a local ONNX model (``fastembed``): no API key, no per-token
cost, no patient text leaving the machine. For a system whose whole subject
is record confidentiality, "the embeddings never go anywhere" is worth more
than a few points of retrieval quality.

Three things this module is careful about.

*Queries and passages are embedded differently.* The BGE family is trained
with an instruction prefix on the query side; embedding a question the same
way as a document measurably degrades retrieval. ``embed_query`` and
``embed_documents`` are therefore separate methods, not one method with a
flag that callers forget to set.

*The model is CPU-bound and synchronous.* Every call runs in a worker thread
so a single embedding request cannot stall the event loop and every other
in-flight request with it.

*Dimension is checked, not assumed.* A model that returns 768 dimensions
into a ``vector(384)`` column fails at insert time with an opaque error;
checking at load time says which setting is wrong.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import struct
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache

from app.config import settings
from app.observability.logging import get_logger

log = get_logger(__name__)


class EmbeddingError(RuntimeError):
    """Embedding could not be produced or is the wrong shape."""


class EmbeddingProvider(ABC):
    name: str = "unknown"

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for storage."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""

    def _check(self, vectors: Sequence[Sequence[float]]) -> None:
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingError(
                    f"{self.model_name} returned {len(vector)} dimensions, but "
                    f"EMBEDDING_DIM is {self.dimension}. Update the setting and "
                    "re-run ingestion — stored vectors of the old size cannot "
                    "be compared against the new ones."
                )


class LocalEmbedder(EmbeddingProvider):
    """ONNX sentence-transformer via fastembed. The production default."""

    name = "local"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        dimension: int | None = None,
        cache_size: int | None = None,
    ):
        self._model_name = model_name or settings.embedding_model
        self._dimension = dimension or settings.embedding_dim
        self._model = None  # loaded lazily; see _ensure_model
        self._cache_size = (
            cache_size if cache_size is not None else settings.embedding_cache_size
        )
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_model(self):  # type: ignore[no-untyped-def]
        """Load on first use.

        Loading costs seconds and, on a cold machine, a model download. Doing
        it at import time would make every process that merely imports the
        app — including ones that never embed anything — pay for it.
        """
        if self._model is None:
            from fastembed import TextEmbedding

            log.info("embeddings.loading", model=self._model_name)
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = await asyncio.to_thread(self._embed_passages_sync, list(texts))
        self._check(vectors)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query, reusing a recent identical one (PRD §36 P3).

        **The cache is keyed on the query text and nothing else, and that is
        safe for a specific reason**: the value is a vector *of the question*,
        not an answer and not a record. Authorization happens after this, in
        the retrieval SQL, which is patient-scoped and RLS-backed — so two
        patients asking "what were my last results?" share one embedding and
        still read entirely different rows. A cache keyed on text that
        returned *results* would be a cross-patient leak; one that returns a
        query vector cannot be (§40 P13).

        It does hold question text in process memory, bounded by
        ``EMBEDDING_CACHE_SIZE`` and lost on restart. Nothing is written to
        disk, which is the property that keeps it out of §26's "do not log
        raw clinical data" territory.
        """
        cached = self._query_cache.get(text)
        if cached is not None:
            # Refresh recency so a repeatedly-asked question stays resident.
            self._query_cache.move_to_end(text)
            return list(cached)

        vectors = await asyncio.to_thread(self._embed_query_sync, text)
        self._check(vectors)
        vector = vectors[0]

        if self._cache_size > 0:
            self._query_cache[text] = list(vector)
            while len(self._query_cache) > self._cache_size:
                self._query_cache.popitem(last=False)
        return vector

    def _embed_passages_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.passage_embed(
            texts, batch_size=settings.embedding_batch_size
        )
        return [[float(value) for value in vector] for vector in vectors]

    def _embed_query_sync(self, text: str) -> list[list[float]]:
        model = self._ensure_model()
        return [
            [float(value) for value in vector] for vector in model.query_embed([text])
        ]


class HashingEmbedder(EmbeddingProvider):
    """A deterministic, dependency-free embedder for tests.

    It is **not semantic**: vectors come from hashed token features, so it
    can tell "the same text" from "different text" and nothing more. That is
    exactly what a test of the *plumbing* — chunking, storage, scoping,
    ranking SQL — needs, and it runs in microseconds without downloading a
    model. Retrieval *quality* is measured by the evaluation set against the
    real model, never here.
    """

    name = "hashing"

    def __init__(self, *, dimension: int | None = None) -> None:
        self._dimension = dimension or settings.embedding_dim

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return f"hashing-{self._dimension}"

    def _vector(self, text: str) -> list[float]:
        buckets = [0.0] * self._dimension
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            position = struct.unpack("<Q", digest)[0] % self._dimension
            # Sign from the low bit, so unrelated tokens can cancel rather
            # than all pushing in the same direction.
            buckets[position] += 1.0 if digest[0] & 1 else -1.0

        norm = math.sqrt(sum(value * value for value in buckets))
        if norm == 0:
            # An empty or unsplittable string. A zero vector has no direction;
            # cc_cosine_distance and pgvector both treat it as maximally far.
            return buckets
        return [value / norm for value in buckets]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_embedder(*, provider: str | None = None) -> EmbeddingProvider:
    name = (provider or settings.embedding_provider).lower()
    if name == "local":
        return LocalEmbedder()
    if name == "hashing":
        return HashingEmbedder()
    if name == "voyage":
        raise EmbeddingError(
            "EMBEDDING_PROVIDER=voyage is not implemented. The local ONNX "
            "model is the supported path; add a VoyageEmbedder here if a "
            "hosted model is needed."
        )
    raise EmbeddingError(f"Unknown EMBEDDING_PROVIDER: {name!r}")


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    """The process-wide embedder — the model is expensive to load twice."""
    return build_embedder()
