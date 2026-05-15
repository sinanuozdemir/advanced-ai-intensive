"""Forge code-execution MCP server.

One agent-facing tool — ``execute_python(code)`` — backed by a process-local,
persistent Python namespace. Pre-bound primitives:

  bm25_search(query: str, k: int = 10) -> list[dict]
  dense_search(query: str, k: int = 10) -> list[dict]
  rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]
  hybrid_retrieve(query: str, k: int = 5) -> list[dict]

This is the W2 NB1 "via_programmatic" pattern, ported into Forge as a real
MCP server so any agent (solo, supervisor, ephemeral) can opt in via the
same permission gate as every other tool.

**Risk surface**: ``execute_python`` runs arbitrary Python in the server
subprocess (no sandbox). It is the single highest-risk tool Forge ships,
so the recommended default is ``permissions.tools.code_execute_python = "ask"``
(the bundled ``config.toml`` does this). The subprocess is launched by the
permission-gated tool loader, so the agent cannot bypass the broker.

Stdout is captured and returned (truncated at 4 kB). The namespace persists
across calls so the agent can build up state — call ``reset_namespace`` if
the agent corners itself.
"""
from __future__ import annotations

import contextlib
import io
import sys
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any

from _common import repo_root

# Make ``forge.*`` importable when launched as a script (mirrors repo_rag).
_FORGE_PKG_PARENT = Path(__file__).resolve().parent.parent
if str(_FORGE_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_FORGE_PKG_PARENT))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("forge-code")


# ---------------------------------------------------------------------------
# Pre-bound retrieval primitives — share the W1 stack from repo_rag_server.
# We import the helpers directly so the agent sees the same hit shape.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _rag_state():
    """Load the indexed repo's BM25 + Chroma exactly like repo_rag_server.

    Lifted via direct import so we don't re-spawn or re-index. Returns
    (bm25_payload, chroma_coll, repo_rag_cfg).
    """
    from forge.repo_rag import load_index
    from forge.paths import ForgePaths

    paths = ForgePaths.for_repo(repo_root())
    return load_index(paths)


def _bm25_search_impl(query: str, k: int = 10) -> list[dict]:
    if not query.strip():
        return []
    bm25_payload, _coll, _cfg = _rag_state()
    bm25 = bm25_payload.get("bm25")
    documents = bm25_payload.get("documents") or []
    if bm25 is None or not documents:
        return []
    tokens = [t for t in query.lower().split() if any(c.isalnum() for c in t)]
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


def _dense_search_impl(query: str, k: int = 10) -> list[dict]:
    if not query.strip():
        return []
    _bm25, coll, _cfg = _rag_state()
    if coll.count() == 0:
        return []
    res = coll.query(query_texts=[query], n_results=min(k, coll.count()))
    out: list[dict] = []
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
            "score": float(1.0 - (dist or 0.0)),
        })
    return out


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _rerank_impl(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []
    pairs = [(query, c.get("text", "")) for c in candidates]
    try:
        scores = _reranker().predict(pairs)
    except Exception:
        # If the cross-encoder fails to load, return the input order.
        return candidates[:top_k]
    ranked = sorted(zip(scores, candidates), key=lambda p: float(p[0]), reverse=True)
    return [{**h, "rerank_score": float(s)} for s, h in ranked[:top_k]]


def _hybrid_retrieve_impl(query: str, k: int = 5) -> list[dict]:
    """RRF fuse BM25 + dense, then cross-encoder rerank — same logic the
    ``repo_rag.hybrid_retrieve`` tool uses, exposed here so the agent doesn't
    have to roll its own RRF in Python."""
    _bm25, _coll, cfg = _rag_state()
    pool = max(k * 3, 15)
    bm25_hits = _bm25_search_impl(query, min(pool, cfg.bm25_k * 2))
    dense_hits = _dense_search_impl(query, min(pool, cfg.dense_k * 2))
    scores: dict[str, float] = {}
    keep: dict[str, dict] = {}
    for ranks in (bm25_hits, dense_hits):
        for rank, hit in enumerate(ranks):
            key = hit["chunk_id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (cfg.rrf_k + rank + 1)
            keep.setdefault(key, hit)
    fused = [
        {**keep[c], "fused_score": float(s)}
        for c, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: max(k * 3, 10)]
    ]
    return _rerank_impl(query, fused, top_k=k)


# ---------------------------------------------------------------------------
# Persistent namespace
# ---------------------------------------------------------------------------

_STATE: dict[str, Any] = {
    "namespace": None,
    "execs": 0,
}


def _build_namespace() -> dict[str, Any]:
    """Fresh namespace with the four primitives pre-bound."""
    return {
        "__name__": "__forge_code__",
        "bm25_search": _bm25_search_impl,
        "dense_search": _dense_search_impl,
        "rerank": _rerank_impl,
        "hybrid_retrieve": _hybrid_retrieve_impl,
    }


def _get_namespace() -> dict[str, Any]:
    if _STATE["namespace"] is None:
        _STATE["namespace"] = _build_namespace()
    return _STATE["namespace"]


_MAX_STDOUT = 4_000


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def execute_python(code: str) -> dict:
    """Run a Python snippet in a persistent namespace and return what it printed.

    The namespace already has these helpers pre-bound:

      bm25_search(query, k=10)           -> list[dict]
      dense_search(query, k=10)          -> list[dict]
      rerank(query, candidates, top_k=5) -> list[dict]
      hybrid_retrieve(query, k=5)        -> list[dict]

    Each hit has ``chunk_id``, ``rel_path``, ``start_line``, ``end_line``,
    ``text``, and a score.

    Workflow:
      1. Use ``print()`` to surface anything you want to inspect.
      2. State persists across calls — variables you assign stay available
         until ``reset_namespace`` is invoked.
      3. Stdout is truncated at 4000 chars; consider summarizing large
         outputs in code rather than printing entire structures.

    Returns:
        ``{"ok": bool, "stdout": str, "error": str | None, "exec_count": int}``.
        ``ok=False`` means an exception was raised; the traceback is in
        ``error`` and any partial stdout is still in ``stdout``.
    """
    _STATE["execs"] += 1
    ns = _get_namespace()
    buf = io.StringIO()
    err: str | None = None
    ok = True
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<forge-code>", "exec"), ns)
    except Exception:
        ok = False
        err = traceback.format_exc(limit=8)
    out = buf.getvalue()
    if len(out) > _MAX_STDOUT:
        out = out[:_MAX_STDOUT] + f"\n...[truncated {len(out) - _MAX_STDOUT} chars]"
    return {
        "ok": ok,
        "stdout": out,
        "error": err,
        "exec_count": _STATE["execs"],
    }


@mcp.tool()
def reset_namespace() -> dict:
    """Wipe the persistent namespace (re-binding the four primitives).

    Useful when the agent corners itself with stale state or shadows a
    builtin. The exec counter is preserved so the tracer can still tell
    when the agent reset vs. just stopped using the tool.
    """
    _STATE["namespace"] = _build_namespace()
    return {"ok": True, "exec_count": _STATE["execs"]}


if __name__ == "__main__":
    mcp.run()
