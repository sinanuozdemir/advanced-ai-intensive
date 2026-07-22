"""MemoryAgent — GenericAgent + persistent checkpointing + memory.

Bolts on three things to the base loop:

  1. Persistent checkpointing (SQLite locally / Postgres in prod) replacing
     the in-memory MemorySaver when not using DATABASE_URL.
  2. Semantic memory: optional per-turn recall prefix, plus `semantic_write`
     / `semantic_search` tools (notebook-aligned).
  3. End-of-thread reflection via `end_thread(...)` (episodic + procedural).

Stores are partitioned under ``AGENT_DATA_ROOT / <memory_scope> /`` so supervisor
and sub-agents do not collide.

Activate by setting ``USE_MEMORY_AGENT=1`` and using :func:`make_agent` or
constructing :class:`MemoryAgent` with the right ``memory_scope``.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

# Make the repo's src/ importable regardless of where the flask app is launched from.
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return here.parent


_REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(_REPO_ROOT / "src"))

from agent_builder import GenericAgent  # noqa: E402

from shared.checkpointer import (  # noqa: E402
    make_async_sqlite_checkpointer,
    make_async_postgres_checkpointer,
)
from memory import (  # noqa: E402
    SemanticMemory,
    EpisodicMemory,
    ProceduralMemory,
    reflect_on_thread,
)


_DATA_ROOT = Path(os.environ.get("AGENT_DATA_ROOT", _REPO_ROOT / "data" / "sdr_runtime"))
_DATA_ROOT.mkdir(parents=True, exist_ok=True)

CONFIG_TO_SCOPE: dict[str, str] = {
    "lead_gen_config.json": "lead_gen",
    "qualifying_agent.json": "qualifier",
    "email_agent.json": "email",
}

KNOWN_MEMORY_SCOPES: tuple[str, ...] = ("supervisor", "lead_gen", "qualifier", "email")

MEM_SYSTEM_PROMPT_SUFFIX = """
---
Long-term memory tools:
- semantic_write(text): store a short natural-language fact worth recalling later (preferences, stable business facts, commitments). Do not store chitchat.
- semantic_search(query): retrieve similar prior memories by meaning when continuity matters.

BEFORE answering questions about prior conversations, prospects, commitments, or anything that sounds like "do you remember", you MUST first call semantic_search with a short query. Only say you have no memory of something if semantic_search returns nothing relevant. The "Recent past conversations" block below is also memory — use it.

Chitchat and ephemeral task detail usually do not belong in semantic memory.
""".strip()


def _render_recent_episodic(episodic: EpisodicMemory, n: int = 5) -> str:
    """Render the last n episodic entries as a system-prompt block."""
    try:
        rows = episodic.all(limit=max(n * 4, 20))
    except Exception:  # noqa: BLE001
        return ""
    if not rows:
        return ""
    rows = sorted(rows, key=lambda e: e.created_at or "", reverse=True)[:n]
    lines = ["# Recent past conversations (episodic memory, most recent first)"]
    for e in rows:
        when = (e.created_at or "")[:19]
        lines.append(
            f"- [{when}] thread={e.thread_id or '?'} score={e.score:g}\n  {e.summary}"
        )
    return "\n".join(lines)


def agent_data_root() -> Path:
    return _DATA_ROOT


def memory_scope_for_config(config_path: str) -> str:
    return CONFIG_TO_SCOPE.get(Path(config_path).name, "default")


def build_semantic_tools(
    semantic: SemanticMemory,
    get_thread_id: Callable[[], str],
) -> list:
    """LangChain tools aligned with ``notebooks/memory_systems.ipynb``."""

    @tool
    def semantic_write(text: str) -> str:
        """Persist a concise natural-language fact for later recall (user prefs, business facts, commitments)."""
        tid = get_thread_id()
        mid = semantic.write(text, thread_id=tid)
        return f"stored semantic memory id={mid}"

    @tool
    def semantic_search(query: str) -> str:
        """Search prior semantic memories by similarity. Use when answers may depend on remembered context."""
        hits = semantic.search(query, k=5, min_score=0.25)
        if not hits:
            return "(no matching memories)"
        lines = [f"- {h.text} (similarity={h.score:.2f})" for h in hits]
        return "\n".join(lines)

    return [semantic_write, semantic_search]


def memory_stores_snapshot(scope: str, *, limit: int = 500) -> dict[str, Any]:
    """JSON-serializable view of one scope's stores (for inspector API)."""
    base = _DATA_ROOT / scope
    semantic = SemanticMemory(base / "semantic_chroma")
    episodic = EpisodicMemory(base / "episodic_chroma")
    procedural = ProceduralMemory(base / "procedural.sqlite")
    sem_rows = semantic.all(limit=limit)
    epi_rows = episodic.all(limit=limit)
    proc_rows = procedural.all()
    return {
        "scope": scope,
        "counts": {
            "semantic": semantic.count(),
            "episodic": episodic.count(),
            "procedural": procedural.count(),
        },
        "semantic": [
            {
                "id": r.id,
                "text": r.text,
                "thread_id": r.thread_id,
                "created_at": r.created_at,
            }
            for r in sem_rows
        ],
        "episodic": [
            {
                "id": e.id,
                "summary": e.summary,
                "thread_id": e.thread_id,
                "score": e.score,
                "created_at": e.created_at,
            }
            for e in epi_rows
        ],
        "procedural": [
            {
                "name": s.name,
                "when_to_use": s.when_to_use,
                "fragment_preview": (s.fragment[:240] + "…") if len(s.fragment) > 240 else s.fragment,
                "score": s.score,
                "usage_count": s.usage_count,
                "created_at": s.created_at,
            }
            for s in proc_rows
        ],
    }


class MemoryAgent(GenericAgent):
    """GenericAgent + checkpointing + scoped memory stores + semantic tools."""

    def __init__(self, config_path: str = "agent_config.json", *, memory_scope: str = "default"):
        super().__init__(config_path=config_path)
        self.memory_scope = memory_scope
        self._scope_root = _DATA_ROOT / memory_scope
        self._scope_root.mkdir(parents=True, exist_ok=True)
        self._sqlite_checkpoint_path = self._scope_root / "sdr_checkpoints.sqlite"
        # Postgres / sqlite saver is created lazily inside the asyncio loop
        # via ``_ensure_checkpointer`` — the LangGraph savers are bound to the
        # loop they are created on.
        self.checkpointer = None
        self.semantic = SemanticMemory(self._scope_root / "semantic_chroma")
        self.episodic = EpisodicMemory(self._scope_root / "episodic_chroma")
        self.procedural = ProceduralMemory(self._scope_root / "procedural.sqlite")
        self._turn_counts: dict[str, int] = {}
        self._active_memory_thread_id: str | None = None

    async def _ensure_checkpointer(self) -> None:
        if self.checkpointer is not None:
            return
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            self.checkpointer = await make_async_postgres_checkpointer(db_url)
        else:
            self.checkpointer = await make_async_sqlite_checkpointer(self._sqlite_checkpoint_path)

    async def _merge_tools(self, tools: list[Any]) -> list[Any]:
        mem_tools = build_semantic_tools(
            self.semantic,
            lambda: (self._active_memory_thread_id or "default"),
        )
        return [*tools, *mem_tools]

    async def initialize(self) -> None:
        await self._ensure_checkpointer()
        await super().initialize()

    def _get_system_prompt(self) -> str:
        base = super()._get_system_prompt()
        skills_block = self.procedural.render_for_system_prompt(n=5)
        episodic_block = _render_recent_episodic(self.episodic, n=5)
        parts: list[str] = [base]
        if skills_block:
            parts.append(skills_block)
        if episodic_block:
            parts.append(episodic_block)
        parts.append(MEM_SYSTEM_PROMPT_SUFFIX)
        return "\n\n".join(parts)

    def _use_memory_autoprefix(self) -> bool:
        return os.environ.get("USE_MEMORY_AUTOPREFIX", "1").lower() in {"1", "true", "yes"}

    async def chat(self, message: str, conversation_id: str = None) -> dict[str, Any]:
        thread_id = conversation_id or "default"
        self._active_memory_thread_id = thread_id
        try:
            hits: list = []
            if self._use_memory_autoprefix():
                try:
                    hits = list(self.semantic.search(message, k=5, min_score=0.25))
                except Exception:  # noqa: BLE001
                    hits = []
                if hits:
                    recall = "\n".join(f"- {h.text}" for h in hits)
                    message = "[memory recall]\n" + recall + "\n\n[user message]\n" + message

            result = await super().chat(message=message, conversation_id=conversation_id)
            if not result.get("success"):
                return result

            turn_idx = self._turn_counts.get(thread_id, 0)
            self._turn_counts[thread_id] = turn_idx + 1
            result["semantic_recall_hits"] = len(hits)
            return result
        finally:
            self._active_memory_thread_id = None

    async def end_thread(self, conversation_id: str, rubric_score: float = 0.0) -> dict[str, Any]:
        """Run end-of-thread reflection (episodic + procedural)."""
        history = await self.get_conversation_history(conversation_id)
        messages = [
            {
                "role": "user" if h["type"] == "human" else "assistant",
                "content": h["content"],
            }
            for h in history
        ]
        return reflect_on_thread(
            thread_id=conversation_id,
            messages=messages,
            episodic=self.episodic,
            procedural=self.procedural,
            rubric_score=rubric_score,
        )

    def memory_stats(self) -> dict[str, int]:
        return {
            "semantic_memories": self.semantic.count(),
            "episodic_entries": self.episodic.count(),
            "procedural_skills": self.procedural.count(),
        }


def make_agent(config_path: str = "agent_config.json") -> GenericAgent:
    """Factory: MemoryAgent with scoped stores if the feature flag is on."""
    if os.environ.get("USE_MEMORY_AGENT", "").lower() in {"1", "true", "yes"}:
        scope = memory_scope_for_config(config_path)
        print(
            f"SDR running with MemoryAgent scope={scope} "
            f"({datetime.now().isoformat()})"
        )
        return MemoryAgent(config_path=config_path, memory_scope=scope)
    return GenericAgent(config_path=config_path)
