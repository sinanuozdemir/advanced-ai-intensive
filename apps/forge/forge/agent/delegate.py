"""Worker-as-tool delegation.

Forge's main agent runs in a single ``langchain.agents.create_agent`` loop
with a flat tool list. Specialists (the built-in planner / coder / critic
plus every persistent agent in ``.forge/agents/*.toml``) appear in that
list as one ``delegate_to_<name>`` tool each. Calling the tool spins up
the specialist's own ``create_agent`` on the supplied sub-task, runs it
to completion in a fresh context window, and returns the answer string.

This replaces the old supervisor-with-router topology: the main LLM owns
both routing decisions (via tool calls) and the final reply (via its
normal completion). No separate router or synthesis LLM.

Identity & permission semantics:

* Built-in workers (planner / coder / critic) receive the main-identity
  tool list narrowed by name set (``READ_ONLY``, ``READ_ONLY|WRITE``,
  etc.). The broker treats their calls as if the main agent made them.
* Persistent agents receive tools wrapped with their own ``agent_name``
  via ``LoadedTools.wrap_for_agent`` so the broker's per-agent allowlist
  (declared in the TOML) is what actually gates execution.

Memory tools are never bundled into delegate tool lists — only the main
agent reads/writes semantic memory. That's the locked design decision in
the README.
"""
from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from shared import get_llm
from multi_agent.workers import WorkerSpec

from ..config import ForgeConfig
from ..paths import ForgePaths
from ..trace import Tracer
from .agents_registry import load_persistent_agents
from .compaction import build_compaction_with_summarizer
from .workers import default_workers


class _DelegateArgs(BaseModel):
    sub_task: str = Field(
        description=(
            "A self-contained instruction for the specialist. Include all "
            "context the specialist needs — it does NOT see the main "
            "conversation history. Be specific about expected output format."
        ),
    )


def _make_delegate_tool(
    spec: WorkerSpec,
    *,
    cfg: ForgeConfig,
    tracer: Tracer,
) -> StructuredTool:
    """Wrap a ``WorkerSpec`` as a ``delegate_to_<name>`` LangChain tool."""
    middleware = build_compaction_with_summarizer(
        cfg.compaction, cfg.models.summarizer,
    )
    tool_name = f"delegate_to_{spec.name}"

    async def _run(sub_task: str) -> str:
        tracer.emit(
            "agent_spawn", agent_name=spec.name, kind="worker", parent="main",
        )
        agent = create_agent(
            model=get_llm(spec.model_slug),
            tools=list(spec.tools),
            system_prompt=spec.system_prompt,
            middleware=middleware,
        )
        try:
            out = await agent.ainvoke(
                {"messages": [HumanMessage(content=sub_task)]},
            )
        except Exception as exc:  # noqa: BLE001
            err = f"ERROR: {type(exc).__name__}: {exc}"
            tracer.emit("agent_done", agent_name=spec.name, result=err[:240])
            return err
        msgs = out.get("messages", []) if isinstance(out, dict) else []
        answer = msgs[-1].content if msgs else ""
        tracer.emit(
            "agent_done", agent_name=spec.name, result=str(answer)[:240],
        )
        return str(answer)

    return StructuredTool.from_function(
        coroutine=_run,
        name=tool_name,
        description=(
            f"Hand a sub-task to the {spec.name!r} specialist and return "
            f"its concise answer. {spec.description.strip()} "
            "The specialist runs in a fresh context (no shared history); "
            "include any needed context in `sub_task`."
        ),
        args_schema=_DelegateArgs,
    )


def collect_specialists(
    *,
    paths: ForgePaths,
    cfg: ForgeConfig,
    tools: list[Any],
    loaded_tools: Any,
) -> list[WorkerSpec]:
    """Return the ordered ``WorkerSpec`` list the main agent can delegate to.

    Built-ins come first (planner, coder, critic) so the LLM's default
    routing instinct lands on the predictable canonical flow when nothing
    else fits; persistent agents follow in load order.
    """
    specs: list[WorkerSpec] = list(default_workers(tools, cfg.models))
    for entry in load_persistent_agents(paths):
        scoped = (
            loaded_tools.wrap_for_agent(entry.spec.name)
            if loaded_tools is not None else tools
        )
        specs.append(entry.to_worker(scoped))
    return specs


def build_delegate_tools(
    specs: list[WorkerSpec],
    *,
    cfg: ForgeConfig,
    tracer: Tracer,
) -> list[StructuredTool]:
    """Materialize one delegate tool per spec."""
    return [_make_delegate_tool(s, cfg=cfg, tracer=tracer) for s in specs]


def specialist_menu(specs: list[WorkerSpec]) -> str:
    """Format a markdown bullet menu of available specialists for the
    main agent's system prompt. One line per spec; trimmed descriptions."""
    if not specs:
        return ""
    lines = []
    for s in specs:
        desc = (s.description or "").strip().splitlines()[0]
        lines.append(f"- `delegate_to_{s.name}` — {desc}")
    return "\n".join(lines)


__all__ = [
    "build_delegate_tools",
    "collect_specialists",
    "specialist_menu",
]
