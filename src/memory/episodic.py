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
    updated_at: str = ""           # bumped on upsert_by_thread; falls back to
                                   # ``created_at`` for legacy rows
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
        """Insert one summary. Returns the entry id.

        Use :meth:`upsert_by_thread` instead when reflection should keep
        a single rolling summary per thread (Forge's current default —
        the legacy ``write`` is kept for the week-2 notebook and any
        caller that intentionally wants multiple rows per thread).
        """
        eid = f"ep-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        if not entry.created_at:
            entry.created_at = now
        if not entry.updated_at:
            entry.updated_at = entry.created_at
        self._coll.add(
            ids=[eid],
            documents=[entry.summary],
            metadatas=[{
                "thread_id": entry.thread_id,
                "score": entry.score,
                "created_at": entry.created_at,
                "updated_at": entry.updated_at,
            }],
        )
        return eid

    def upsert_by_thread(self, entry: EpisodicEntry) -> str:
        """Insert or overwrite the episode for ``entry.thread_id``.

        Reflection now runs once per chat turn but should produce a
        single rolling summary per thread, refined as the conversation
        grows. This method enforces that invariant by keying on
        ``thread_id`` instead of a fresh UUID per call.

        Returns the row id (existing if found, new otherwise).
        ``created_at`` is preserved across updates; ``updated_at`` is
        always set to "now".
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            existing = self._coll.get(where={"thread_id": entry.thread_id})
        except Exception:  # noqa: BLE001
            existing = {"ids": [], "metadatas": []}
        ids_row = list(existing.get("ids") or [])
        metas_row = list(existing.get("metadatas") or [])
        if ids_row:
            eid = ids_row[0]
            prior_meta = metas_row[0] or {}
            created_at = (
                entry.created_at
                or prior_meta.get("created_at")
                or now
            )
            self._coll.update(
                ids=[eid],
                documents=[entry.summary],
                metadatas=[{
                    "thread_id": entry.thread_id,
                    "score": entry.score,
                    "created_at": created_at,
                    "updated_at": now,
                }],
            )
            # Defensive: if older code wrote duplicate rows for the same
            # thread, collapse them down to the one we just updated.
            if len(ids_row) > 1:
                try:
                    self._coll.delete(ids=list(ids_row[1:]))
                except Exception:  # noqa: BLE001
                    pass
            return eid
        eid = f"ep-{uuid.uuid4().hex[:12]}"
        created_at = entry.created_at or now
        self._coll.add(
            ids=[eid],
            documents=[entry.summary],
            metadatas=[{
                "thread_id": entry.thread_id,
                "score": entry.score,
                "created_at": created_at,
                "updated_at": now,
            }],
        )
        return eid

    def get_by_thread(self, thread_id: str) -> EpisodicEntry | None:
        """Return the existing episode for ``thread_id``, or ``None``."""
        try:
            res = self._coll.get(where={"thread_id": thread_id})
        except Exception:  # noqa: BLE001
            return None
        ids_row = res.get("ids") or []
        if not ids_row:
            return None
        docs_row = res.get("documents") or []
        metas_row = res.get("metadatas") or []
        meta = (metas_row[0] or {}) if metas_row else {}
        return EpisodicEntry(
            summary=(docs_row[0] if docs_row else "") or "",
            thread_id=thread_id,
            score=float(meta.get("score", 0.0) or 0.0),
            created_at=meta.get("created_at", "") or "",
            updated_at=meta.get("updated_at", "") or "",
            id=ids_row[0],
        )

    def search(self, cue: str, k: int = 3) -> list[EpisodicEntry]:
        if self._coll.count() == 0:
            return []
        res = self._coll.query(query_texts=[cue], n_results=min(k, self._coll.count()))
        out: list[EpisodicEntry] = []
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            created_at = meta.get("created_at", "") or ""
            out.append(EpisodicEntry(
                summary=doc,
                thread_id=meta.get("thread_id", ""),
                score=float(meta.get("score", 0.0) or 0.0),
                created_at=created_at,
                updated_at=meta.get("updated_at", "") or created_at,
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
            created_at = meta.get("created_at", "") or ""
            out.append(
                EpisodicEntry(
                    summary=doc or "",
                    thread_id=meta.get("thread_id", ""),
                    score=float(meta.get("score", 0.0) or 0.0),
                    created_at=created_at,
                    updated_at=meta.get("updated_at", "") or created_at,
                    id=eid,
                )
            )
        return out

    def count(self) -> int:
        return self._coll.count()
