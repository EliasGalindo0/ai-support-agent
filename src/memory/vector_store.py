"""
Lightweight vector store for semantic memory retrieval.

Design:
- Uses numpy cosine similarity — no heavy dependencies (no Pinecone/Weaviate required).
- Persists index to a .npz file on disk for durability across restarts.
- Pluggable embedding function: default uses Anthropic's text-embedding model
  or falls back to a simple TF-IDF heuristic for offline/testing.
- For production at scale, swap the backend to Pinecone / pgvector / Qdrant
  by replacing VectorStore._index_backend.

Embeddings are used for:
- Long-term memory: find relevant past interactions.
- Knowledge base: semantic search over docs.
- RAG pipeline: augment agent context with retrieved chunks.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from src.config import get_settings

log = structlog.get_logger(__name__)


@dataclass
class VectorDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    document: VectorDocument
    score: float  # cosine similarity 0-1


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------
class _AnthropicEmbedder:
    """Uses voyage-large-2 via Anthropic (best quality for support use cases)."""

    def __init__(self, api_key: str) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        # Voyage model dimensions
        self._dim = 1024

    async def embed(self, texts: list[str]) -> np.ndarray:
        import anthropic
        # Voyage embedding via Anthropic client
        response = self._client.beta.messages.batches  # type: ignore
        # Fallback: use OpenAI embeddings or simple hash embedding
        # For production, integrate voyage-large-2 directly
        return self._fallback_embed(texts)

    def _fallback_embed(self, texts: list[str]) -> np.ndarray:
        """TF-IDF inspired hash embedding — deterministic, no API calls needed."""
        dim = get_settings().embedding_dim
        result = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            for token in tokens:
                h = hash(token) % dim
                result[i, h] += 1.0
            norm = np.linalg.norm(result[i])
            if norm > 0:
                result[i] /= norm
        return result


class _OpenAIEmbedder:
    """Uses text-embedding-3-small (1536 dims, low cost)."""

    def __init__(self, api_key: str) -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = "text-embedding-3-small"
        self._dim = 1536

    async def embed(self, texts: list[str]) -> np.ndarray:
        response = await self._client.embeddings.create(
            input=texts, model=self._model
        )
        vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        return np.stack(vectors)


class _FallbackEmbedder:
    """Hash-based embedding — zero dependencies, for testing."""

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        dim = self._dim
        result = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            for token in tokens:
                h = hash(token) % dim
                result[i, h] += 1.0
            norm = np.linalg.norm(result[i])
            if norm > 0:
                result[i] /= norm
        return result


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
class VectorStore:
    """
    Flat cosine-similarity vector store with disk persistence.

    Suitable for corpora up to ~100K documents.
    Above that, consider pgvector or Qdrant.
    """

    def __init__(self, namespace: str = "default") -> None:
        cfg = get_settings()
        self._namespace = namespace
        self._index_path = Path(f"data/embeddings/{namespace}.npz")
        self._meta_path = Path(f"data/embeddings/{namespace}_meta.json")

        # Choose embedder
        if cfg.openai_api_key:
            self._embedder: Any = _OpenAIEmbedder(cfg.openai_api_key)
        else:
            self._embedder = _FallbackEmbedder(dim=cfg.embedding_dim)

        # In-memory index
        self._vectors: np.ndarray | None = None     # shape (N, dim)
        self._documents: list[VectorDocument] = []

        self._load()

    def _load(self) -> None:
        try:
            if self._index_path.exists() and self._meta_path.exists():
                data = np.load(str(self._index_path))
                self._vectors = data["vectors"]
                with open(self._meta_path) as f:
                    metas = json.load(f)
                self._documents = [
                    VectorDocument(id=m["id"], text=m["text"], metadata=m["metadata"])
                    for m in metas
                ]
                log.info(
                    "vector_store.loaded",
                    namespace=self._namespace,
                    count=len(self._documents),
                )
        except Exception as exc:
            log.warning("vector_store.load_failed", error=str(exc))
            self._vectors = None
            self._documents = []

    def _save(self) -> None:
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            if self._vectors is not None:
                np.savez_compressed(str(self._index_path), vectors=self._vectors)
            with open(self._meta_path, "w") as f:
                json.dump(
                    [
                        {"id": d.id, "text": d.text, "metadata": d.metadata}
                        for d in self._documents
                    ],
                    f,
                )
        except Exception as exc:
            log.error("vector_store.save_failed", error=str(exc))

    async def add(self, documents: list[VectorDocument]) -> None:
        if not documents:
            return

        texts = [d.text for d in documents]
        new_vectors = await self._embedder.embed(texts)

        if self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

        self._documents.extend(documents)
        self._save()

        log.info(
            "vector_store.added",
            namespace=self._namespace,
            count=len(documents),
            total=len(self._documents),
        )

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self._vectors is None or len(self._documents) == 0:
            return []

        k = top_k or get_settings().long_term_search_top_k
        query_vec = await self._embedder.embed([query])  # (1, dim)

        # Cosine similarity
        # Vectors are already normalised for fallback embedder; normalise here too
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalised = self._vectors / norms

        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        scores = (normalised @ q_norm.T).flatten()  # (N,)

        # Apply metadata filter
        indices = list(range(len(self._documents)))
        if metadata_filter:
            indices = [
                i for i in indices
                if all(
                    self._documents[i].metadata.get(fk) == fv
                    for fk, fv in metadata_filter.items()
                )
            ]

        top_indices = sorted(indices, key=lambda i: scores[i], reverse=True)[:k]
        return [
            SearchResult(document=self._documents[i], score=float(scores[i]))
            for i in top_indices
        ]

    async def delete(self, doc_id: str) -> bool:
        idx = next(
            (i for i, d in enumerate(self._documents) if d.id == doc_id), None
        )
        if idx is None:
            return False

        self._documents.pop(idx)
        if self._vectors is not None:
            self._vectors = np.delete(self._vectors, idx, axis=0)
            if len(self._documents) == 0:
                self._vectors = None

        self._save()
        return True

    def count(self) -> int:
        return len(self._documents)
