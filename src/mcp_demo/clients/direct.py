"""Direct (in-process) client.

The agent reaches the three retrieval primitives as plain Python @tool
functions in the same process. No transport overhead. This is the
"Direct" column in the four-quadrant slide.
"""

from __future__ import annotations

import time

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from shared import estimate_cost, get_llm
from .. import tools as _tools
from .common import ClientResult


def _build_agent(model_slug: str):
    @tool
    def bm25_search(query: str, k: int = 10) -> list[dict]:
        """Lexical BM25 search over the corpus. Returns up to k hits."""
        return _tools.bm25_search(query, k=k)

    @tool
    def dense_search(query: str, k: int = 10) -> list[dict]:
        """Dense vector search over the corpus. Returns up to k hits."""
        return _tools.dense_search(query, k=k)

    @tool
    def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """Cross-encoder rerank a candidate list (deduped by id)."""
        return _tools.rerank(query, candidates, top_k=top_k)

    return create_agent(
        model=get_llm(model_slug),
        tools=[bm25_search, dense_search, rerank],
        system_prompt=(
            "You are a research assistant. You have three tools:\n"
            "  - bm25_search(query, k): lexical search\n"
            "  - dense_search(query, k): vector search\n"
            "  - rerank(query, candidates, top_k): cross-encoder rerank\n"
            "Standard pipeline: call bm25_search and dense_search for the same query, "
            "concatenate their hits, then call rerank to pick the best top_k. "
            "Use the reranked text to write a concise, cited answer."
        ),
    )


def run(question: str, model_slug: str = "openai/gpt-5.4-nano") -> ClientResult:
    agent = _build_agent(model_slug)
    t0 = time.time()
    out = agent.invoke({"messages": [HumanMessage(content=question)]})
    elapsed = time.time() - t0

    msgs = out["messages"]
    final = msgs[-1].content if msgs else ""
    n_tool = sum(1 for m in msgs if getattr(m, "tool_calls", None))
    in_t = out_t = 0
    for m in msgs:
        um = getattr(m, "usage_metadata", None) or {}
        in_t += int(um.get("input_tokens", 0) or 0)
        out_t += int(um.get("output_tokens", 0) or 0)
    return ClientResult(
        client="direct",
        answer=final,
        n_tool_calls=n_tool,
        tool_latency_total_s=0.0,
        total_latency_s=elapsed,
        input_tokens=in_t,
        output_tokens=out_t,
        cost_usd=estimate_cost(model_slug, in_t, out_t),
        raw_messages=msgs,
    )
