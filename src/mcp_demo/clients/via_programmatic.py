"""Programmatic client.

Multi-turn LangGraph ReAct agent whose only tool is `execute_python(code)`.
The exec namespace has the three primitives pre-bound (bm25_search,
dense_search, rerank) — the model writes code, observes printed output,
iterates, then states a final answer.

This is the "Programmatic" column in segment 1's four-quadrant slide:
the model composes pre-made tools as code in successive turns. Contrast
with the coding-agent client, which gets NO pre-bound tools and must
write its own from scratch using a skill doc.
"""

from __future__ import annotations

import io
import time
import contextlib
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

from shared import estimate_cost, get_llm
from .. import tools as _tools
from .common import ClientResult


SYSTEM_PROMPT = """You are a programmatic agent with one tool: `execute_python(code)`.

The execution namespace persists across calls and has these functions \
already imported:

  bm25_search(query: str, k: int = 10) -> list[dict]
  dense_search(query: str, k: int = 10) -> list[dict]
  rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]

Each hit dict has keys: id, source, title, text, score.

Workflow:
1. Write a small snippet that retrieves candidates with bm25_search and \
dense_search.
2. Inspect what came back (use print()).
3. Concatenate hits and call rerank to keep the top 5.
4. Print the reranked hit titles/snippets so you can see them.
5. Once you have good evidence, reply directly to the user with a \
concise, cited answer. DO NOT call execute_python again after that.

Use AT MOST 5 execute_python calls."""


def _build_namespace() -> tuple[dict[str, Any], dict[str, int]]:
    counter: dict[str, int] = {}

    def wrap(fn, name: str):
        def wrapped(*args, **kwargs):
            counter[name] = counter.get(name, 0) + 1
            return fn(*args, **kwargs)
        return wrapped

    ns: dict[str, Any] = {
        "bm25_search": wrap(_tools.bm25_search, "bm25_search"),
        "dense_search": wrap(_tools.dense_search, "dense_search"),
        "rerank": wrap(_tools.rerank, "rerank"),
    }
    return ns, counter


def run(question: str, model_slug: str = "openai/gpt-5.4-nano") -> ClientResult:
    namespace, _prim_counter = _build_namespace()
    exec_calls = {"n": 0}

    @tool
    def execute_python(code: str) -> str:
        """Execute Python code in a persistent namespace.

        The namespace already has `bm25_search`, `dense_search`, `rerank`
        bound. Use print() to surface intermediate results back to yourself.
        Returns captured stdout (truncated to 4000 chars) plus any error.
        """
        exec_calls["n"] += 1
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<programmatic>", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001
            out = buf.getvalue()
            return f"{out}\n[ERROR] {type(exc).__name__}: {exc}"
        out = buf.getvalue()
        if len(out) > 4000:
            out = out[:4000] + "\n...[truncated]"
        return out or "(no output)"

    agent = create_agent(
        model=get_llm(model_slug),
        tools=[execute_python],
        system_prompt=SYSTEM_PROMPT,
    )

    t0 = time.time()
    out = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 16},
    )
    elapsed = time.time() - t0

    msgs = out["messages"]
    final = msgs[-1].content if msgs else ""
    if isinstance(final, list):
        final = "\n".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in final)

    in_t = out_t = 0
    for m in msgs:
        um = getattr(m, "usage_metadata", None) or {}
        in_t += int(um.get("input_tokens", 0) or 0)
        out_t += int(um.get("output_tokens", 0) or 0)

    return ClientResult(
        client="via_programmatic",
        answer=str(final),
        n_tool_calls=exec_calls["n"],
        tool_latency_total_s=0.0,
        total_latency_s=elapsed,
        input_tokens=in_t,
        output_tokens=out_t,
        cost_usd=estimate_cost(model_slug, in_t, out_t),
        raw_messages=msgs,
    )
