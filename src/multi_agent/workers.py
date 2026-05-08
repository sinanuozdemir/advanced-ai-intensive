"""Specialist worker agents used by every topology in `topologies.py`.

Each worker is a `(name, system_prompt, tools, model_slug)` quartet wrapped as
a `WorkerSpec`. Topologies turn specs into LangChain `create_agent` instances.

The three workers exercise *different* skills so a heterogeneous task mix can
discriminate between flat and supervised topologies:
- `researcher` -> retrieval + synthesis (week 1's hybrid retriever)
- `summarizer` -> long-text compression (no tools, big context)
- `code_runner` -> arithmetic + light Python (a sandboxed `python` tool)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import tool


@dataclass
class WorkerSpec:
    """Declarative description of a specialist agent.

    Topologies materialise these into agents with `langchain.agents.create_agent`.
    Keeping the spec as data (not a closure over an LLM) means topologies can
    swap the model per-worker without re-defining the worker.
    """

    name: str
    description: str           # used by the supervisor to pick a worker
    system_prompt: str
    tools: list[Any] = field(default_factory=list)
    model_slug: str = "openai/gpt-5.4-nano"


# ---------------------------------------------------------------------------
# Tool builders. Each returns a list of @tool callables ready to bind.
# ---------------------------------------------------------------------------


def _retrieve_tool_factory(hybrid, reranker):
    """Build the `retrieve` tool used by the researcher worker."""

    @tool
    def retrieve(query: str) -> str:
        """Search the heterogeneous corpus and return up to 5 reranked chunks.

        Args:
            query: A focused, self-contained sub-query.
        """
        pool = hybrid.search(query, k=10)
        ranked = reranker.rerank(query, pool, top_k=5)
        return "\n\n---\n\n".join(
            f"[{d.metadata.get('source')}: {d.metadata.get('title','?')[:50]}] "
            f"{d.page_content[:600]}"
            for d in ranked
        )

    return [retrieve]


def _python_tool() -> list[Callable]:
    """A *minimal* Python execution tool for arithmetic and string munging.

    Intentionally narrow: `eval` of a single expression, no statements.
    Strict whitelist of names so it's safe enough for a teaching notebook.
    """

    import math

    _SAFE_NAMES = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "round": round, "sorted": sorted, "set": set, "list": list,
        "tuple": tuple, "dict": dict, "range": range, "math": math,
    }

    @tool
    def python_eval(expression: str) -> str:
        """Evaluate a single Python *expression* and return its repr.

        Use for arithmetic, list comprehensions, and small data wrangling.
        Cannot run statements, imports, or I/O.

        Args:
            expression: A single Python expression, e.g. `sum(x**2 for x in range(10))`.
        """
        try:
            value = eval(expression, {"__builtins__": {}}, _SAFE_NAMES)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {type(exc).__name__}: {exc}"
        return repr(value)[:1000]

    return [python_eval]


# ---------------------------------------------------------------------------
# Worker factories. Notebooks call these to assemble a topology.
# ---------------------------------------------------------------------------


def make_researcher(hybrid, reranker, *, model_slug: str = "openai/gpt-5.4-nano") -> WorkerSpec:
    return WorkerSpec(
        name="researcher",
        description=(
            "Use for questions that need facts from the heterogeneous corpus "
            "(Beehiiv newsletters + Wikipedia + HotpotQA). Has a `retrieve` tool."
        ),
        system_prompt=(
            "You are a research specialist. Your job is to call the `retrieve` "
            "tool with focused, self-contained queries and return a concise, "
            "evidence-cited answer. Cite sources inline like [wikipedia: Title]."
        ),
        tools=_retrieve_tool_factory(hybrid, reranker),
        model_slug=model_slug,
    )


def make_summarizer(*, model_slug: str = "openai/gpt-5.4-nano") -> WorkerSpec:
    return WorkerSpec(
        name="summarizer",
        description=(
            "Use for tasks that require condensing or restructuring long text "
            "the user supplies in the question. No tools — pure transformation."
        ),
        system_prompt=(
            "You are a summarization specialist. Take the input text and "
            "produce a faithful, dense summary. Do NOT add facts that aren't "
            "in the source. Default to 5 bullet points unless the task says otherwise."
        ),
        tools=[],
        model_slug=model_slug,
    )


def make_code_runner(*, model_slug: str = "openai/gpt-5.4-nano") -> WorkerSpec:
    return WorkerSpec(
        name="code_runner",
        description=(
            "Use for arithmetic, counting, list/set operations, or anything "
            "that benefits from running a Python expression. Has a `python_eval` tool."
        ),
        system_prompt=(
            "You are a numeric/code specialist. For any computation, call the "
            "`python_eval` tool with a single expression rather than guessing. "
            "Then return the answer with one sentence of explanation."
        ),
        tools=_python_tool(),
        model_slug=model_slug,
    )


# ---------------------------------------------------------------------------
# Bundle helpers
# ---------------------------------------------------------------------------


def default_workers(
    hybrid, reranker, *, model_slug: str = "openai/gpt-5.4-nano"
) -> list[WorkerSpec]:
    """Convenience: the standard three-worker bundle Segment 1 ships with."""
    return [
        make_researcher(hybrid, reranker, model_slug=model_slug),
        make_summarizer(model_slug=model_slug),
        make_code_runner(model_slug=model_slug),
    ]
