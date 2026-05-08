"""Embedding function for the memory stores.

Calls OpenRouter's /v1/embeddings endpoint directly via httpx — no torch,
no sentence-transformers, no openai SDK. Default model is
`google/gemini-embedding-001`.

Env vars:
  OPENROUTER_API_KEY  -- required
  MEMORY_EMBED_MODEL  -- optional override (default: google/gemini-embedding-001)
"""

from __future__ import annotations

import os
from typing import Sequence

import httpx
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


_DEFAULT_MODEL = os.getenv("MEMORY_EMBED_MODEL", "google/gemini-embedding-001")
_ENDPOINT = "https://openrouter.ai/api/v1/embeddings"


class OpenRouterEmbeddingFunction(EmbeddingFunction):
    """Chroma EmbeddingFunction backed by OpenRouter's /v1/embeddings."""

    def __init__(self, model: str = _DEFAULT_MODEL, *, api_key: str | None = None, timeout: float = 60.0):
        self.model = model
        self._api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self._client = httpx.Client(timeout=timeout)

    @staticmethod
    def name() -> str:
        return "openrouter"

    def __call__(self, input: Documents) -> Embeddings:
        texts: Sequence[str] = list(input)
        if not texts:
            return []
        resp = self._client.post(
            _ENDPOINT,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": list(texts)},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [list(item["embedding"]) for item in data]


def default_embedding_function() -> EmbeddingFunction:
    return OpenRouterEmbeddingFunction()
