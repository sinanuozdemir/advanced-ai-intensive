"""Forge's main agent.

Single ``langchain.agents.create_agent`` that owns the entire turn. Its
flat tool list is:

* every permission-gated MCP tool (``fs_*``, ``shell_*``, ``git_*``,
  ``repo_rag_*``, ``code_*``, and any user-installed ``user_<slug>_*``);
* the memory tools (``semantic_read`` / ``semantic_write``) — these live
  ONLY on the main agent;
* one ``delegate_to_<name>`` tool per built-in worker (planner / coder /
  critic) and per persistent agent (``.forge/agents/*.toml``);
* a ``spawn_ephemeral`` tool for short-lived specialists assembled on the
  fly with a custom tool subset.

The main agent decides routing by calling those delegate tools when a
specialist fits the sub-task; otherwise it just answers. No separate
router LLM, no supervisor synthesis step — trivial chat is one call,
complex tasks are N+1 (one per delegate).

The dataclass keeps its historical name (``ForgeSoloAgent``) for the
handful of external imports; conceptually it's the one and only Forge
topology now.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from shared import get_llm
from shared.checkpointer import make_async_sqlite_checkpointer

from ..config import ForgeConfig
from ..paths import ForgePaths
from ..trace import Tracer
from .compaction import build_compaction_with_summarizer
from .delegate import (
    build_delegate_tools,
    collect_specialists,
    specialist_menu,
)
from .prompts import MAIN_SYSTEM, PLAN_MODE_ADDENDUM
from .spawn import build_spawn_tool


@dataclass
class ForgeSoloAgent:
    """Historical name retained for import compatibility — this is now the
    sole Forge agent topology, not just a fallback."""

    agent: Any
    checkpointer: Any
    tools: list[Any]
    tracer: Tracer

    async def ainvoke(
        self, task: str, *, thread_id: str
    ) -> tuple[str, list[Any]]:
        """Run one turn / task. State is keyed by ``thread_id`` for resume."""
        self.tracer.emit(
            "agent_spawn", agent_name="main", kind="main", parent=None,
        )
        out = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=task)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        msgs = out["messages"]
        answer = msgs[-1].content if msgs else ""
        self.tracer.emit(
            "agent_done", agent_name="main", result=_preview(answer),
        )
        return answer, msgs


# Public alias for the new architecture (the name actually reflects what
# the class IS now). The old name is kept above for callers that haven't
# been updated yet.
ForgeMainAgent = ForgeSoloAgent


async def build_forge_solo(
    *,
    paths: ForgePaths,
    cfg: ForgeConfig,
    tools: list[Any],
    memory_tools: list[Any] | None,
    tracer: Tracer,
    loaded_tools: Any = None,
    plan_mode: bool = False,
) -> ForgeSoloAgent:
    """Build the main agent.

    Args:
        tools: Main-identity gated tool list (the same one given to
            built-in workers; persistent agents get their own identity
            inside ``delegate.collect_specialists``).
        memory_tools: ``semantic_read`` / ``semantic_write``; ``None`` or
            empty when memory is disabled in config.
        loaded_tools: ``forge.mcp.LoadedTools`` — required for persistent
            agents so they get their own broker identity. ``None`` falls
            back to using ``tools`` directly (acceptable for tests).
        plan_mode: When True, prepends an addendum to the system prompt
            that nudges the agent to call ``delegate_to_planner`` before
            any write-class tool. Maps to today's ``mode="plan"`` flag.
    """
    middleware = build_compaction_with_summarizer(
        cfg.compaction, cfg.models.summarizer,
    )
    checkpointer = await make_async_sqlite_checkpointer(paths.checkpoints_sqlite)

    specs = collect_specialists(
        paths=paths, cfg=cfg, tools=tools, loaded_tools=loaded_tools,
    )
    delegate_tools = build_delegate_tools(specs, cfg=cfg, tracer=tracer)
    spawn_tool = build_spawn_tool(
        all_tools=tools, default_model=cfg.models.default_agent, tracer=tracer,
    )

    menu = specialist_menu(specs)
    system_prompt = MAIN_SYSTEM
    if menu:
        system_prompt += (
            "\n\n### Available specialists\n\n"
            "When a sub-task fits one of these, call its delegate tool with "
            "a self-contained `sub_task` string. The specialist runs in a "
            "fresh context (no shared history) and returns a concise answer "
            "you can incorporate into your reply.\n\n" + menu
        )
    if plan_mode:
        system_prompt += "\n\n" + PLAN_MODE_ADDENDUM

    all_tools = (
        list(tools)
        + list(memory_tools or [])
        + list(delegate_tools)
        + [spawn_tool]
    )
    agent = create_agent(
        model=get_llm(cfg.models.default_agent),
        tools=all_tools,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer,
    )
    return ForgeSoloAgent(
        agent=agent, checkpointer=checkpointer, tools=all_tools, tracer=tracer,
    )


# Friendlier public name. ``build_forge_solo`` is retained as an alias for
# the (handful of) external callers that haven't been migrated yet.
build_forge_main = build_forge_solo


def _preview(s: str, n: int = 240) -> str:
    s = s if isinstance(s, str) else str(s)
    return s if len(s) <= n else s[: n - 3] + "..."


__all__ = [
    "ForgeMainAgent",
    "ForgeSoloAgent",
    "build_forge_main",
    "build_forge_solo",
]
