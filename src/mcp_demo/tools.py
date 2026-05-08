"""Three primitive retrieval tools shared by all four clients in notebook 1.

The whole point of segment 1 is that the *same three primitives* can be
reached four different ways (direct in-process, MCP over stdio, programmatic
code-in-turn, or a coding agent). This module is the single source of truth
for what those primitives do.

Each function returns a list of plain dicts so the same payloads survive
JSON serialization across MCP, exec'd into a sandboxed namespace by the
programmatic / coding-agent clients, etc.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from shared import CrossEncoderReranker, load_bm25, load_chroma
from retrievers import bm25_search as _bm25_search_docs, dense_search as _dense_search_docs

# Lazy singletons. Importing this module is cheap; the heavy artifacts
# (BM25 index, MiniLM embeddings, cross-encoder weights) load on first use.
_CHROMA: Any = None
_BM25: Any = None
_RERANKER: CrossEncoderReranker | None = None


def _chroma() -> Any:
    global _CHROMA
    if _CHROMA is None:
        _CHROMA = load_chroma()
    return _CHROMA


def _bm25() -> Any:
    global _BM25
    if _BM25 is None:
        _BM25 = load_bm25()
    return _BM25


def _reranker() -> CrossEncoderReranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoderReranker()
    return _RERANKER


def _doc_to_hit(doc: Document, score: float | None = None) -> dict:
    meta = doc.metadata or {}
    return {
        "id": str(meta.get("chunk_id") or f"{meta.get('source','?')}::{hash(doc.page_content) & 0xffffffff:x}"),
        "source": meta.get("source", "?"),
        "title": meta.get("title", "?"),
        "text": doc.page_content,
        "score": float(score) if score is not None else None,
    }


def bm25_search(query: str, k: int = 10) -> list[dict]:
    """Lexical BM25 search. Returns up to k hits, each as a plain dict."""
    docs = _bm25_search_docs(_bm25(), query, k=k)
    return [_doc_to_hit(d) for d in docs]


def dense_search(query: str, k: int = 10) -> list[dict]:
    """Dense (MiniLM-embedded) search. Returns up to k hits, each as a plain dict."""
    docs = _dense_search_docs(_chroma(), query, k=k)
    return [_doc_to_hit(d) for d in docs]


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Cross-encoder rerank a candidate list (deduped by id), returning top_k."""
    if not candidates:
        return []
    seen: set[str] = set()
    deduped: list[dict] = []
    for c in candidates:
        cid = str(c.get("id"))
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(c)

    docs = [Document(page_content=c["text"], metadata={"source": c.get("source"), "title": c.get("title"), "chunk_id": c.get("id")}) for c in deduped]
    pairs = [(query, d.page_content) for d in docs]
    model = _reranker()._load()
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, deduped), key=lambda p: float(p[0]), reverse=True)
    out: list[dict] = []
    for s, c in ranked[:top_k]:
        c2 = dict(c)
        c2["score"] = float(s)
        out.append(c2)
    return out


__all__ = ["bm25_search", "dense_search", "rerank"]
