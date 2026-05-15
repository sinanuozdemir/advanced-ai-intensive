"""Forge memory layer.

A thin wrapper around ``src/memory/*`` that pins paths to the per-repo
``.forge/memory/`` directory and exposes:

* ``MemoryStores`` — bundle of semantic / episodic / procedural stores.
* ``build_memory_tools`` — LangChain tools (``semantic_write``,
  ``semantic_read``) wired to the tracer for live updates in the Memory
  panel.
* ``reflect_main_thread`` — end-of-thread reflection that persists
  episodic summaries and procedural skills.

Only the main chat agent receives these tools in the MVP (per
``forge_brief.md`` + the agreed plan).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.tools import tool

# Re-exports from src/memory/*. ``src/`` is on PYTHONPATH for this repo.
from memory.episodic import EpisodicMemory
from memory.procedural import ProceduralMemory
from memory.semantic import SemanticMemory, SemanticMemoryRecord

from .paths import ForgePaths
from .trace import Tracer


@dataclass
class MemoryStores:
    semantic: SemanticMemory
    episodic: EpisodicMemory
    procedural: ProceduralMemory

    @classmethod
    def for_paths(cls, paths: ForgePaths) -> "MemoryStores":
        paths.ensure()
        return cls(
            semantic=SemanticMemory(path=paths.semantic_chroma),
            episodic=EpisodicMemory(path=paths.episodic_chroma),
            procedural=ProceduralMemory(
                path=paths.procedural_sqlite,
                when_chroma_path=paths.procedural_when_chroma,
            ),
        )


def build_memory_tools(
    stores: MemoryStores,
    *,
    tracer: Tracer | None = None,
    thread_id: str = "",
    semantic_k: int = 5,
    agent_name: str = "main",
) -> list[Any]:
    """Return ``[semantic_write, semantic_read]`` as LangChain tools.

    The ``thread_id`` is captured in each write so we can attribute memories
    back to the originating conversation when browsing the Memory panel.

    Both tools emit the same ``tool_call`` / ``tool_result`` pair that
    MCP-backed tools do (via ``mcp._wrap_with_gate``) so that the Chat
    view's tool cards, ``_load_thread_transcript``, and
    ``thread_eval.reconstruct_trajectory`` all surface memory operations
    uniformly. The ``memory_write`` / ``memory_read`` events are kept on
    top for the Memory panel, which keys off ``store=``.
    """

    @tool
    def semantic_write(text: str) -> str:
        """Record one durable, free-form natural-language memory about the
        user, their preferences, workspace, or long-lived project facts.

        Call this ONLY when you've learned something the user is likely to
        want you to remember across conversations — preferences, invariants,
        recurring patterns. Do NOT use this as a scratchpad for intra-task
        notes.

        Args:
            text: A single concise sentence. Examples:
              - "User prefers four-space indents in Python."
              - "Project tests live in tests/, not test/."
        """
        if tracer is not None:
            tracer.emit(
                "tool_call", agent_name=agent_name,
                tool="semantic_write", args={"text": text},
            )
        try:
            rec = SemanticMemoryRecord(text=text, thread_id=thread_id)
            sid = stores.semantic.write(rec)
        except Exception as exc:  # noqa: BLE001
            if tracer is not None:
                tracer.emit(
                    "tool_result", agent_name=agent_name,
                    tool="semantic_write", ok=False,
                    preview=f"error: {exc!r}"[:240],
                )
            raise
        if tracer is not None:
            tracer.emit("memory_write", store="semantic", id=sid, text=text)
            tracer.emit(
                "tool_result", agent_name=agent_name,
                tool="semantic_write", ok=True,
                preview=f"stored {sid}",
            )
        return f"stored {sid}"

    @tool
    def semantic_read(query: str, k: int = semantic_k) -> str:
        """Recall up to ``k`` semantic memories most similar to ``query``.

        The harness only injects a small one-time "thread seed" on the first turn;
        after that, **you** must call this when recall matters: user preferences,
        past stable facts, workspace invariants, anything the user might assume
        you remember, or narrower details than the seed provided. Prefer a
        targeted phrase or question as ``query``; lower ``k`` when you only need
        a quick check.

        Args:
            query: A short, focused recall cue.
            k: Max number of memories to return.
        """
        if tracer is not None:
            tracer.emit(
                "tool_call", agent_name=agent_name,
                tool="semantic_read", args={"query": query, "k": k},
            )
        try:
            hits = stores.semantic.search(
                query, k=max(1, min(int(k or 1), 20)),
            )
        except Exception as exc:  # noqa: BLE001
            if tracer is not None:
                tracer.emit(
                    "tool_result", agent_name=agent_name,
                    tool="semantic_read", ok=False,
                    preview=f"error: {exc!r}"[:240],
                )
            raise
        if tracer is not None:
            tracer.emit(
                "memory_read", store="semantic", query=query, hits=len(hits),
            )
        if not hits:
            result = "(no relevant memories)"
        else:
            result = "\n".join(
                f"- {h.text} (score={h.score:.2f})" for h in hits
            )
        if tracer is not None:
            tracer.emit(
                "tool_result", agent_name=agent_name,
                tool="semantic_read", ok=True,
                preview=result[:240],
            )
        return result

    return [semantic_write, semantic_read]


# ---------------------------------------------------------------------------
# End-of-thread reflection
# ---------------------------------------------------------------------------


def _messages_to_dicts(messages: list[Any]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m.__class__.__name__.replace("Message", "").lower()
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def reflect_main_thread(
    *,
    stores: MemoryStores,
    tracer: Tracer | None,
    thread_id: str,
    messages: list[Any],
    rubric_score: float = 0.0,
    model_slug: str = "openai/gpt-5.4-nano",
) -> dict:
    """Reflect on the main chat agent's transcript and persist learnings.

    Delegates to ``memory_reflect_agent.reflect_with_agent`` which runs
    reflection as a proper agent loop (tools for searching existing
    skills before writing). The agent emits its own ``memory_write``
    events from inside its save tools, so no extra tracing is needed
    here.
    """
    from .memory_reflect_agent import reflect_with_agent

    return reflect_with_agent(
        stores=stores,
        tracer=tracer,
        thread_id=thread_id,
        messages=messages,
        rubric_score=rubric_score,
        model_slug=model_slug,
    )


__all__ = [
    "MemoryStores",
    "build_memory_tools",
    "reflect_main_thread",
]
