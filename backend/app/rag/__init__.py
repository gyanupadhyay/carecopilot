"""Retrieval-augmented generation over clinical documentation (PRD §19, §20)."""

from app.rag.chunking import Chunk, chunk_document, contextualize, split_sections
from app.rag.context import BuiltContext, build_context
from app.rag.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    HashingEmbedder,
    LocalEmbedder,
    build_embedder,
    get_embedder,
)
from app.rag.fusion import FusionResult, deduplicate, fuse, reciprocal_rank_fusion
from app.rag.ingestion import IngestionReport, ingest_document, ingest_documents
from app.rag.pipeline import RagResult, normalize_query, retrieve
from app.rag.reranking import (
    HeuristicReranker,
    LLMReranker,
    NoopReranker,
    Reranker,
    build_reranker,
)
from app.rag.retrieval import RetrievedChunk, keyword_search, vector_search

__all__ = [
    "BuiltContext",
    "Chunk",
    "EmbeddingError",
    "EmbeddingProvider",
    "FusionResult",
    "HashingEmbedder",
    "HeuristicReranker",
    "IngestionReport",
    "LLMReranker",
    "LocalEmbedder",
    "NoopReranker",
    "RagResult",
    "Reranker",
    "RetrievedChunk",
    "build_context",
    "build_embedder",
    "build_reranker",
    "chunk_document",
    "contextualize",
    "deduplicate",
    "fuse",
    "get_embedder",
    "ingest_document",
    "ingest_documents",
    "keyword_search",
    "normalize_query",
    "reciprocal_rank_fusion",
    "retrieve",
    "split_sections",
    "vector_search",
]
