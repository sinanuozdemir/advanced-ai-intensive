"""Persistent agents registry.

A persistent agent is a user-configured named worker (e.g. ``researcher``,
``test_writer``) that the supervisor can delegate to. Configuration lives at
``<repo>/.forge/agents/<name>.toml``:

    name = "researcher"
    description = "Search the repo & web for context. Read-only."
    model = "openai/gpt-5.4-nano"
    system_prompt = '''You are the researcher specialist...'''
    tools = ["fs_read", "fs_list", "repo_rag_hybrid_retrieve", "git_diff"]

The registry exposes one ``WorkerSpec`` per agent + helpers to upsert /
delete a config (used by the Electron Agents tab via the FastAPI server).
No memory store of its own (deferred per the locked decisions in the plan).
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w
from pydantic import BaseModel, Field, ValidationError

from multi_agent.workers import WorkerSpec

from ..paths import ForgePaths


class PersistentAgentSpec(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    model: str = "openai/gpt-5.4-nano"
    system_prompt: str = Field(min_length=1)
    tools: list[str] = Field(default_factory=list)


@dataclass
class LoadedPersistentAgent:
    spec: PersistentAgentSpec
    toml_path: Path

    def to_worker(self, all_tools: list[Any]) -> WorkerSpec:
        """Build a ``WorkerSpec`` whose ``tools`` are the subset of ``all_tools``
        named in ``spec.tools``."""
        allowed = set(self.spec.tools)
        narrowed = [t for t in all_tools if t.name in allowed]
        return WorkerSpec(
            name=self.spec.name,
            description=self.spec.description,
            system_prompt=self.spec.system_prompt,
            tools=narrowed,
            model_slug=self.spec.model,
        )


def load_persistent_agents(paths: ForgePaths) -> list[LoadedPersistentAgent]:
    """Walk ``.forge/agents/*.toml`` and return validated specs.

    Files that fail validation are skipped with a warning printed to stderr;
    we don't want one broken agent to crash the TUI.
    """
    out: list[LoadedPersistentAgent] = []
    if not paths.agents_dir.is_dir():
        return out
    for toml_path in sorted(paths.agents_dir.glob("*.toml")):
        try:
            with toml_path.open("rb") as fh:
                raw = tomllib.load(fh)
            # Default ``name`` from filename when missing.
            raw.setdefault("name", toml_path.stem)
            spec = PersistentAgentSpec.model_validate(raw)
        except (ValidationError, Exception) as exc:  # noqa: BLE001
            import sys
            print(
                f"forge: skipping invalid persistent agent {toml_path}: {exc}",
                file=sys.stderr,
            )
            continue
        out.append(LoadedPersistentAgent(spec=spec, toml_path=toml_path))
    return out


def write_persistent_agent(
    paths: ForgePaths, spec: PersistentAgentSpec
) -> Path:
    """Atomically write ``.forge/agents/<spec.name>.toml``."""
    paths.ensure()
    target = paths.agents_dir / f"{spec.name}.toml"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{spec.name}.toml.", dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            tomli_w.dump(spec.model_dump(mode="python"), fh)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def delete_persistent_agent(paths: ForgePaths, name: str) -> bool:
    """Remove ``.forge/agents/<name>.toml``. Returns True if it existed."""
    target = paths.agents_dir / f"{name}.toml"
    if not target.is_file():
        return False
    target.unlink()
    return True


__all__ = [
    "PersistentAgentSpec",
    "LoadedPersistentAgent",
    "load_persistent_agents",
    "write_persistent_agent",
    "delete_persistent_agent",
]
