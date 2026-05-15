"""Ephemeral sub-agent spawning.

The supervisor gets a ``spawn`` tool that opens a fresh thread, runs a
short-lived specialist agent, and collapses its result back into the
supervisor's transcript. The ephemeral agent shares the supervisor's
permission identity (``agent_name == "ephemeral"`` for the permission
broker) and has no memory store of its own.

Implementation: build a ``langchain.agents.create_agent`` per spawn (cheap
— no checkpointer, no middleware), ``ainvoke`` once so MCP-gated tools work
(sync ``invoke`` would hit StructuredTools that disallow sync execution).
"""
from __future__ import annotations

import uuid
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from shared import get_llm

from ..trace import Tracer


def build_spawn_tool(
    *,
    all_tools: list[Any],
    default_model: str,
    tracer: Tracer | None = None,
) -> Any:
    """Return a LangChain ``spawn`` tool the supervisor can call.

    Args (when the LLM calls the tool):
        role: A short name for the specialist (e.g. "test_writer"). Used
            for trace + audit attribution.
        sub_task: The exact instruction the ephemeral specialist should
            execute. Self-contained — the ephemeral agent does not see
            the supervisor's history.
        tools: Comma-separated tool names to restrict the specialist to.
            Empty string means "all tools the supervisor sees".
    """
    tool_by_name = {t.name: t for t in all_tools}

    @tool
    async def spawn(role: str, sub_task: str, tools: str = "") -> str:
        """Spawn a short-lived specialist agent on a self-contained sub-task.

        The specialist runs in a fresh context window (no inherited
        history), executes the sub-task, and returns its final answer.
        Use this when you want a focused detour without polluting your
        own working memory.

        Args:
            role: A short name for trace attribution, e.g. "test_writer".
            sub_task: A self-contained instruction. The specialist sees
                only this string, not the supervisor's history.
            tools: Comma-separated tool names to grant the specialist.
                Empty string grants all tools the supervisor has.
        """
        wanted = (
            {t.strip() for t in tools.split(",") if t.strip()}
            if tools else set(tool_by_name.keys())
        )
        narrowed = [tool_by_name[n] for n in wanted if n in tool_by_name]
        eph_id = f"eph-{uuid.uuid4().hex[:8]}"
        if tracer is not None:
            tracer.emit(
                "agent_spawn", agent_name=f"{role}::{eph_id}",
                kind="ephemeral", parent="main",
            )
        agent = create_agent(
            model=get_llm(default_model),
            tools=narrowed,
            system_prompt=(
                f"You are an ephemeral specialist with role={role!r}. "
                "You run in a fresh context, execute the given sub-task, "
                "and return a single concise answer."
            ),
        )
        try:
            out = await agent.ainvoke(
                {"messages": [HumanMessage(content=sub_task)]},
            )
        except Exception as exc:  # noqa: BLE001
            err = f"ERROR: {type(exc).__name__}: {exc}"
            if tracer is not None:
                tracer.emit(
                    "agent_done", agent_name=f"{role}::{eph_id}",
                    result=err[:200],
                )
            return err
        msgs = out.get("messages", [])
        answer = msgs[-1].content if msgs else ""
        if tracer is not None:
            tracer.emit(
                "agent_done", agent_name=f"{role}::{eph_id}",
                result=str(answer)[:200],
            )
        return str(answer)

    return spawn


__all__ = ["build_spawn_tool"]
