"""Shared helpers for Forge's MCP servers.

Each server is launched as its own subprocess over stdio. It learns which
repo to operate on from the ``FORGE_REPO`` env var (set by the agent loader
in ``forge.mcp.tool_loader``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the forge package is importable when these servers are launched
# as scripts (not modules) — i.e. `python apps/forge/mcp_servers/fs_server.py`.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


def repo_root() -> Path:
    """Return the repo Forge is operating on, from ``FORGE_REPO`` or CWD."""
    env = os.environ.get("FORGE_REPO")
    return Path(env).resolve() if env else Path.cwd().resolve()


def ensure_under_root(p: str | Path) -> Path:
    """Resolve ``p`` (relative to repo root if not absolute) and refuse if it
    escapes the repo root."""
    root = repo_root()
    cand = Path(p)
    if not cand.is_absolute():
        cand = root / cand
    cand = cand.resolve()
    try:
        cand.relative_to(root)
    except ValueError as exc:
        raise PermissionError(
            f"path {cand} escapes repo root {root}"
        ) from exc
    return cand
