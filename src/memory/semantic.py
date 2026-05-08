"""Semantic memory — natural-language memories with embedding retrieval.

A semantic memory is a short, free-form sentence the agent chooses to store
about the user/world (e.g. ``"the user loves dogs"``). Memories are stored in
a Chroma collection so the agent can recall by similarity at any later turn.

Design rules:

- Memories are NL strings, not (subject, predicate, object) triples.
- Writes happen only via explicit agent tool calls (``semantic_write``),
  not as an automatic per-turn side effect.
- Retrieval is similarity-based via :class:`SemanticMemory.search`.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from .embedding import default_embedding_function


@dataclass
class SemanticMemoryRecord:
    """One natural-language semantic memory."""

    id: str = ""
    text: str = ""
    thread_id: str = ""
    created_at: str = ""
    score: float = 0.0  # similarity score for search results


class SemanticMemory:
    """Chroma-backed natural-language semantic memory.

    Storage is intentionally free-form: each record is a single concise
    sentence in natural language. Retrieval uses Chroma's default sentence
    transformer embeddings so the store is self-contained.
    """

    def __init__(
        self,
        path: str | Path = "data/memory/semantic_chroma",
        *,
        collection: str = "semantic",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._coll = self._client.get_or_create_collection(
            name=collection, embedding_function=default_embedding_function(),
        )

    def write(self, record: SemanticMemoryRecord | str, **kwargs: Any) -> str:
        """Insert one NL memory.

        Accepts either a fully built :class:`SemanticMemoryRecord` or a string
        plus keyword metadata (the convenient form for tool calls).
        """
        if isinstance(record, str):
            record = SemanticMemoryRecord(text=record, **kwargs)
        if not record.text or not record.text.strip():
            raise ValueError("semantic memory text must be a non-empty string")
        if not record.id:
            record.id = f"sm-{uuid.uuid4().hex[:12]}"
        if not record.created_at:
            record.created_at = datetime.now(timezone.utc).isoformat()

        self._coll.add(
            ids=[record.id],
            documents=[record.text.strip()],
            metadatas=[
                {
                    "thread_id": record.thread_id,
                    "created_at": record.created_at,
                }
            ],
        )
        return record.id

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        min_score: float | None = None,
    ) -> list[SemanticMemoryRecord]:
        if not query or not query.strip():
            return []
        if self._coll.count() == 0:
            return []
        res = self._coll.query(
            query_texts=[query],
            n_results=min(k, self._coll.count()),
        )
        out: list[SemanticMemoryRecord] = []
        ids_row = (res.get("ids") or [[]])[0]
        docs_row = (res.get("documents") or [[]])[0]
        metas_row = (res.get("metadatas") or [[]])[0]
        dists_row = (res.get("distances") or [[]])[0] or [0.0] * len(docs_row)
        for sm_id, doc, meta, dist in zip(ids_row, docs_row, metas_row, dists_row):
            similarity = max(0.0, 1.0 - float(dist))
            if min_score is not None and similarity < min_score:
                continue
            meta = meta or {}
            out.append(
                SemanticMemoryRecord(
                    id=sm_id,
                    text=doc,
                    thread_id=meta.get("thread_id", ""),
                    created_at=meta.get("created_at", ""),
                    score=similarity,
                )
            )
        return out

    def get(self, record_id: str) -> SemanticMemoryRecord | None:
        if not record_id:
            return None
        try:
            res = self._coll.get(ids=[record_id])
        except Exception:  # noqa: BLE001
            return None
        ids_row = res.get("ids") or []
        if not ids_row:
            return None
        docs = res.get("documents") or [""]
        metas = res.get("metadatas") or [{}]
        meta = metas[0] or {}
        return SemanticMemoryRecord(
            id=ids_row[0],
            text=docs[0] if docs else "",
            thread_id=meta.get("thread_id", ""),
            created_at=meta.get("created_at", ""),
        )

    def delete(self, record_id: str) -> bool:
        if not record_id:
            return False
        try:
            self._coll.delete(ids=[record_id])
            return True
        except Exception:  # noqa: BLE001
            return False

    def all(self, *, limit: int = 1000) -> list[SemanticMemoryRecord]:
        if self._coll.count() == 0:
            return []
        res = self._coll.get(limit=limit)
        ids_row = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        out: list[SemanticMemoryRecord] = []
        for sm_id, doc, meta in zip(ids_row, docs, metas):
            meta = meta or {}
            out.append(
                SemanticMemoryRecord(
                    id=sm_id,
                    text=doc,
                    thread_id=meta.get("thread_id", ""),
                    created_at=meta.get("created_at", ""),
                )
            )
        return out

    def count(self) -> int:
        return self._coll.count()
