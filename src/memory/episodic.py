"""Episodic memory — past thread summaries, written via reflection.

Each entry is a self-contained "what happened in that thread" summary plus
the metadata needed to attribute it (thread id, timestamp, rubric score).
Stored in Chroma so we can do similarity search at conversation start
("have I seen a thread like this before?").

Why a vector store and not SQLite? Episodic recall is fuzzy by nature —
the cue at retrieval time ("user is asking about cold-outreach to a fintech
CTO") will rarely match exact words from the past summary. Embeddings
buy us that fuzzy match for free.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from .embedding import default_embedding_function


@dataclass
class EpisodicEntry:
    summary: str
    thread_id: str
    score: float = 0.0             # rubric overall, 0-5, if available
    created_at: str = ""
    id: str = ""                   # chroma id when listing via :meth:`all`


class EpisodicMemory:
    """Chroma-backed episodic store.

    Uses Chroma's default sentence-transformers embedder so the store is
    self-contained — no LLM call needed at write or read time.
    """

    def __init__(self, path: str | Path = "data/memory/episodic_chroma",
                 collection: str = "episodic"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._coll = self._client.get_or_create_collection(
            name=collection, embedding_function=default_embedding_function(),
        )

    def write(self, entry: EpisodicEntry) -> str:
        """Insert one summary. Returns the entry id."""
        eid = f"ep-{uuid.uuid4().hex[:12]}"
        if not entry.created_at:
            entry.created_at = datetime.now(timezone.utc).isoformat()
        self._coll.add(
            ids=[eid],
            documents=[entry.summary],
            metadatas=[{
                "thread_id": entry.thread_id,
                "score": entry.score,
                "created_at": entry.created_at,
            }],
        )
        return eid

    def search(self, cue: str, k: int = 3) -> list[EpisodicEntry]:
        if self._coll.count() == 0:
            return []
        res = self._coll.query(query_texts=[cue], n_results=min(k, self._coll.count()))
        out: list[EpisodicEntry] = []
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            out.append(EpisodicEntry(
                summary=doc,
                thread_id=meta.get("thread_id", ""),
                score=float(meta.get("score", 0.0) or 0.0),
                created_at=meta.get("created_at", ""),
            ))
        return out

    def all(self, *, limit: int = 500) -> list[EpisodicEntry]:
        """List entries (most recently added last in Chroma get order)."""
        n = self._coll.count()
        if n == 0:
            return []
        cap = min(limit, n)
        res = self._coll.get(limit=cap)
        ids_row = res.get("ids") or []
        docs_row = res.get("documents") or []
        metas_row = res.get("metadatas") or []
        out: list[EpisodicEntry] = []
        for eid, doc, meta in zip(ids_row, docs_row, metas_row):
            meta = meta or {}
            out.append(
                EpisodicEntry(
                    summary=doc or "",
                    thread_id=meta.get("thread_id", ""),
                    score=float(meta.get("score", 0.0) or 0.0),
                    created_at=meta.get("created_at", ""),
                    id=eid,
                )
            )
        return out

    def count(self) -> int:
        return self._coll.count()
