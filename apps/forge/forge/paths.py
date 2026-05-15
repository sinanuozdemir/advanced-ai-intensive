"""Per-repo state convention.

Forge stores everything under ``<repo>/.forge/`` when launched inside a repo,
and falls back to ``apps/forge/data/`` when launched outside any repo (e.g.
during testing of the package itself).

We detect "the repo" as: the closest ancestor directory containing either a
``.git`` directory or an existing ``.forge`` directory. If nothing matches we
treat the CWD as the repo root.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Resolve the repo root.

    Resolution order:

    1. The ``FORGE_REPO`` env var (the MCP subprocesses use this — it's the
       most reliable signal we have).
    2. Walk up from ``start`` (or CWD) looking for ``.git`` or ``.forge``.
    3. Fall back to ``start`` / CWD itself.
    """
    env = os.environ.get("FORGE_REPO")
    if env:
        return Path(env).resolve()
    cur = (start or Path.cwd()).resolve()
    for parent in (cur, *cur.parents):
        if (parent / ".git").exists() or (parent / ".forge").exists():
            return parent
    return cur


@dataclass(frozen=True)
class ForgePaths:
    """Resolved per-repo paths used everywhere in Forge."""

    repo_root: Path
    forge_dir: Path
    config_toml: Path
    agents_dir: Path
    memory_dir: Path
    semantic_chroma: Path
    episodic_chroma: Path
    procedural_sqlite: Path
    procedural_when_chroma: Path
    checkpoints_sqlite: Path
    audit_jsonl: Path
    trace_jsonl: Path
    permissions_toml: Path
    eval_results_dir: Path
    rag_index_dir: Path

    @classmethod
    def for_repo(cls, start: Path | None = None) -> "ForgePaths":
        repo = find_repo_root(start)
        forge_dir = repo / ".forge"
        memory_dir = forge_dir / "memory"
        return cls(
            repo_root=repo,
            forge_dir=forge_dir,
            config_toml=forge_dir / "config.toml",
            agents_dir=forge_dir / "agents",
            memory_dir=memory_dir,
            semantic_chroma=memory_dir / "semantic_chroma",
            episodic_chroma=memory_dir / "episodic_chroma",
            procedural_sqlite=memory_dir / "procedural.sqlite",
            procedural_when_chroma=memory_dir / "procedural_when_chroma",
            checkpoints_sqlite=forge_dir / "checkpoints.sqlite",
            audit_jsonl=forge_dir / "audit.jsonl",
            trace_jsonl=forge_dir / "trace.jsonl",
            permissions_toml=forge_dir / "permissions.toml",
            eval_results_dir=forge_dir / "eval_results",
            rag_index_dir=forge_dir / "rag_index",
        )

    def ensure(self) -> "ForgePaths":
        """Create every directory we own, idempotently. Returns self."""
        for d in (
            self.forge_dir,
            self.agents_dir,
            self.memory_dir,
            self.eval_results_dir,
            self.rag_index_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self


def env_overrides_root() -> Path | None:
    """``FORGE_REPO=/some/path`` overrides the detected root. Handy for tests."""
    env = os.environ.get("FORGE_REPO")
    return Path(env).resolve() if env else None
