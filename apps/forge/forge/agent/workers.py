"""Forge's three core workers: planner, coder, critic.

These are ``WorkerSpec`` instances (reusing the dataclass from
``src/multi_agent/workers.py``) so they plug straight into ``build_supervisor``
and ``build_solo`` from ``src/multi_agent/topologies.py``.

Each worker receives the shared, permission-gated tool list (plus, in the
coder's case, write tools). The supervisor wrapper in ``supervisor.py``
adds these as default workers.
"""
from __future__ import annotations

from typing import Any

from multi_agent.workers import WorkerSpec

from ..config import ModelsConfig
from .prompts import CODER_SYSTEM, CRITIC_SYSTEM, PLANNER_SYSTEM


# Tools every worker sees by name. The supervisor passes the full gated tool
# list; the worker's system prompt narrows which to use. Names use single
# underscores (OpenAI tool-name regex rejects dots).
READ_ONLY = {"fs_read", "fs_list", "repo_rag_hybrid_retrieve",
             "git_status", "git_diff", "git_log", "git_branch"}
WRITE = {"fs_write", "fs_edit", "fs_mkdir", "shell_exec", "shell_allowlist",
         "git_add", "git_commit", "git_reset"}


def _filter_tools(tools: list[Any], names: set[str]) -> list[Any]:
    return [t for t in tools if t.name in names]


def make_planner(tools: list[Any], cfg: ModelsConfig) -> WorkerSpec:
    """Planner: read-only worker that drafts a numbered plan."""
    return WorkerSpec(
        name="planner",
        description=(
            "Use to draft a numbered plan before any risky / multi-step edit. "
            "Has read tools (fs.read, fs.list, repo_rag.hybrid_retrieve, git.diff). "
            "Cannot modify files."
        ),
        system_prompt=PLANNER_SYSTEM,
        tools=_filter_tools(tools, READ_ONLY),
        model_slug=cfg.planner,
    )


def make_coder(tools: list[Any], cfg: ModelsConfig) -> WorkerSpec:
    """Coder: read+write+shell+git. The workhorse."""
    return WorkerSpec(
        name="coder",
        description=(
            "Use to execute concrete edit/test/refactor work. Has all read+write tools."
        ),
        system_prompt=CODER_SYSTEM,
        tools=_filter_tools(tools, READ_ONLY | WRITE),
        model_slug=cfg.coder,
    )


def make_critic(tools: list[Any], cfg: ModelsConfig) -> WorkerSpec:
    """Critic: read-only verdict on the coder's recent work."""
    return WorkerSpec(
        name="critic",
        description=(
            "Use at the end of a task to verify success and surface remaining "
            "issues. Read-only (fs.read, git.diff, shell.exec for tests)."
        ),
        system_prompt=CRITIC_SYSTEM,
        # Critic gets shell_exec so it can run the tests itself.
        tools=_filter_tools(tools, READ_ONLY | {"shell_exec"}),
        model_slug=cfg.critic,
    )


def default_workers(tools: list[Any], cfg: ModelsConfig) -> list[WorkerSpec]:
    return [
        make_planner(tools, cfg),
        make_coder(tools, cfg),
        make_critic(tools, cfg),
    ]


__all__ = ["make_planner", "make_coder", "make_critic", "default_workers",
           "READ_ONLY", "WRITE"]
