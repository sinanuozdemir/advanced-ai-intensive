"""Same agent code, but `retrieve` is reached over MCP stdio.

Spawns `python -m mcp_demo.server` as a subprocess, opens an MCP session
against its stdio, and uses LangChain's MCP adapter to expose the server's
tools as `@tool`s the agent can call.

Per-call latency: ~50-150ms (stdio framing + JSON-RPC roundtrip).
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from shared import estimate_cost, get_llm
from .common import ClientResult


_SRC = Path(__file__).resolve().parents[2]


async def _run_async(question: str, model_slug: str) -> ClientResult:
    # `langchain-mcp-adapters` is the tiny shim that turns MCP tools into LangChain @tools.
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "week2-retriever": {
                "command": sys.executable,
                "args": ["-m", "mcp_demo.server"],
                "transport": "stdio",
                "env": {"PYTHONPATH": str(_SRC)},
            }
        }
    )
    tools = await client.get_tools()
    agent = create_agent(
        model=get_llm(model_slug),
        tools=tools,
        system_prompt=(
            "You are a research assistant. You have three MCP tools:\n"
            "  - bm25_search(query, k): lexical search\n"
            "  - dense_search(query, k): vector search\n"
            "  - rerank(query, candidates, top_k): cross-encoder rerank\n"
            "Standard pipeline: call bm25_search and dense_search for the same query, "
            "concatenate their hits, then call rerank to pick the best top_k. "
            "Use the reranked text to write a concise, cited answer."
        ),
    )

    t0 = time.time()
    out = await agent.ainvoke({"messages": [HumanMessage(content=question)]})
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
        client="via_mcp",
        answer=final,
        n_tool_calls=n_tool,
        # We don't intercept per-call MCP latency individually; total minus
        # ~LLM-time is a good enough proxy for the segment's lesson.
        tool_latency_total_s=0.0,
        total_latency_s=elapsed,
        input_tokens=in_t,
        output_tokens=out_t,
        cost_usd=estimate_cost(model_slug, in_t, out_t),
        raw_messages=msgs,
    )


def run(question: str, model_slug: str = "openai/gpt-5.4-nano") -> ClientResult:
    """Sync wrapper over the async MCP client. Notebook-friendly.

    `asyncio.run()` can't be called from a thread that already has a running
    event loop (e.g. inside a Jupyter cell). Detect that and farm the work
    out to a one-shot worker thread that owns its own loop. Outside Jupyter
    the fast path is the same as before.
    """
    def _go():
        return asyncio.run(_run_async(question, model_slug))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _go()

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_go).result()
