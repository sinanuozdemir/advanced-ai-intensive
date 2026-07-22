"""Forge codebase-RAG MCP server.

One tool: ``hybrid_retrieve(query, k)``. Backed by BM25 + Chroma (dense)
fused with RRF, then cross-encoder reranked — the W1 stack, re-applied to
the user's repo.

The agent's tool loop is the agentic RAG. This server does NOT iterate
internally.

Loads its index from ``<repo>/.forge/rag_index/`` (built by ``forge index``).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from _common import repo_root

# Ensure the ``forge`` package is importable when launched as a script.
_FORGE_PKG_PARENT = Path(__file__).resolve().parent.parent
if str(_FORGE_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_FORGE_PKG_PARENT))

from mcp.server.fastmcp import FastMCP

from forge.repo_rag import load_index
from forge.paths import ForgePaths

mcp = FastMCP("forge-repo-rag")


@lru_cache(maxsize=1)
def _index():
    paths = ForgePaths.for_repo(repo_root())
    bm25_payload, coll, cfg = load_index(paths)
    # Cross-encoder is heavy; load lazily on first rerank().
    return bm25_payload, coll, cfg


def _bm25_search(payload: dict, query: str, k: int) -> list[dict]:
    bm25 = payload.get("bm25")
    documents = payload.get("documents") or []
    if bm25 is None or not documents:
        return []
    # Mirror notebooks/retrievers._tokenize
    tokens = [t for t in query.lower().split() if t.isalnum() or any(c.isalnum() for c in t)]
    scores = bm25.get_scores(tokens)
    ranked = sorted(zip(scores, documents), key=lambda kv: float(kv[0]), reverse=True)
    return [
        {
            "chunk_id": d.metadata["chunk_id"],
            "rel_path": d.metadata["rel_path"],
            "start_line": d.metadata["start_line"],
            "end_line": d.metadata["end_line"],
            "text": d.page_content,
            "score": float(s),
        }
        for s, d in ranked[:k]
    ]


def _dense_search(coll, query: str, k: int) -> list[dict]:
    if coll.count() == 0:
        return []
    res = coll.query(query_texts=[query], n_results=min(k, coll.count()))
    out = []
    for cid, doc, meta, dist in zip(
        (res.get("ids") or [[]])[0],
        (res.get("documents") or [[]])[0],
        (res.get("metadatas") or [[]])[0],
        (res.get("distances") or [[]])[0],
    ):
        meta = meta or {}
        out.append({
            "chunk_id": cid,
            "rel_path": meta.get("rel_path", ""),
            "start_line": meta.get("start_line", 0),
            "end_line": meta.get("end_line", 0),
            "text": doc,
            "score": float(1.0 - (dist or 0.0)),  # to similarity
        })
    return out


def _rrf_fuse(rank_lists, k: int, rrf_k: int) -> list[dict]:
    scores: dict[str, float] = {}
    keep: dict[str, dict] = {}
    for ranks in rank_lists:
        for rank, hit in enumerate(ranks):
            key = hit["chunk_id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            keep.setdefault(key, hit)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{**keep[c], "fused_score": float(s)} for c, s in ordered[:k]]


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    if not hits:
        return []
    pairs = [(query, h["text"]) for h in hits]
    try:
        model = _reranker()
        scores = model.predict(pairs)
    except Exception as exc:  # noqa: BLE001
        # If the cross-encoder fails to load, fall back to RRF order.
        for h in hits:
            h["rerank_score"] = None
        return hits[:top_k]
    ranked = sorted(zip(scores, hits), key=lambda p: float(p[0]), reverse=True)
    out = []
    for s, h in ranked[:top_k]:
        h = {**h, "rerank_score": float(s)}
        out.append(h)
    return out


@mcp.tool()
def hybrid_retrieve(query: str, k: int = 5) -> list[dict]:
    """Hybrid retrieve over the indexed repo.

    BM25 + dense (Chroma) candidates -> RRF fusion -> cross-encoder rerank.
    Returns up to ``k`` chunks with text and (rel_path, start_line, end_line)
    so the agent can quote / open the source.

    Args:
        query: A focused, self-contained search query.
        k: How many reranked chunks to return (default 5).
    """
    if not query or not query.strip():
        return []
    try:
        bm25_payload, coll, cfg = _index()
    except FileNotFoundError as exc:
        return [{"error": str(exc)}]
    pool = max(k * 3, 15)
    bm25_hits = _bm25_search(bm25_payload, query, min(pool, cfg.bm25_k * 2))
    dense_hits = _dense_search(coll, query, min(pool, cfg.dense_k * 2))
    fused = _rrf_fuse([bm25_hits, dense_hits], k=max(k * 3, 10), rrf_k=cfg.rrf_k)
    return _rerank(query, fused, top_k=k)




if __name__ == "__main__":
    mcp.run()
