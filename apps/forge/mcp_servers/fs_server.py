"""Forge filesystem MCP server.

Tools: read, list, write, edit, mkdir. Every path is constrained to the
repo root (see ``ensure_under_root``). Permission gating is the agent
loader's job — this server just executes.

Run standalone over stdio (the default MCP transport):

    FORGE_REPO=/path/to/repo python apps/forge/mcp_servers/fs_server.py
"""
from __future__ import annotations

import os
from pathlib import Path

from _common import ensure_under_root, repo_root

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("forge-fs")


@mcp.tool()
def read(path: str, max_bytes: int = 1_000_000) -> str:
    """Read a UTF-8 text file from the repo and return its contents.

    Args:
        path: Repo-relative or absolute path. Must resolve inside the repo.
        max_bytes: Cap to keep the LLM context manageable.
    """
    p = ensure_under_root(path)
    if not p.is_file():
        return f"ERROR: not a file: {p}"
    data = p.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
        suffix = f"\n... [truncated at {max_bytes} bytes; file is {p.stat().st_size} bytes]"
    else:
        suffix = ""
    try:
        return data.decode("utf-8") + suffix
    except UnicodeDecodeError:
        return f"ERROR: file at {p} is not utf-8 text"


@mcp.tool()
def list(path: str = ".", recursive: bool = False, limit: int = 200) -> list[str]:
    """List directory contents (repo-relative paths). Hidden files included.

    Args:
        path: Repo-relative directory (default repo root).
        recursive: If True, walk subdirectories.
        limit: Cap the number of entries returned.
    """
    p = ensure_under_root(path)
    if not p.is_dir():
        return [f"ERROR: not a directory: {p}"]
    root = repo_root()
    out: list[str] = []
    iterator = p.rglob("*") if recursive else p.iterdir()
    for entry in iterator:
        rel = entry.relative_to(root).as_posix()
        suffix = "/" if entry.is_dir() else ""
        out.append(rel + suffix)
        if len(out) >= limit:
            out.append(f"... [truncated at {limit} entries]")
            break
    return out


@mcp.tool()
def write(path: str, content: str, create_parents: bool = True) -> str:
    """Overwrite (or create) a UTF-8 file with ``content``.

    Args:
        path: Repo-relative or absolute path inside the repo.
        content: The full new file contents.
        create_parents: If True, create missing parent directories.
    """
    p = ensure_under_root(path)
    if create_parents:
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {p.relative_to(repo_root()).as_posix()}"


@mcp.tool()
def edit(path: str, old: str, new: str, expect_unique: bool = True) -> str:
    """Replace ``old`` with ``new`` in the file at ``path``.

    Args:
        path: Repo-relative path.
        old: Exact substring to replace. Must appear exactly once when
            ``expect_unique`` is True (the default — protects against
            accidental multi-site edits).
        new: Replacement text.
        expect_unique: If True (default), refuse when ``old`` matches 0 or >1
            times. If False, replace all occurrences.
    """
    p = ensure_under_root(path)
    if not p.is_file():
        return f"ERROR: not a file: {p}"
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        return f"ERROR: pattern not found in {p.relative_to(repo_root()).as_posix()}"
    if expect_unique and count > 1:
        return (
            f"ERROR: pattern found {count} times in {p.relative_to(repo_root()).as_posix()}; "
            "pass expect_unique=False to apply to all, or pass a longer pattern."
        )
    p.write_text(text.replace(old, new), encoding="utf-8")
    return f"replaced {count if not expect_unique else 1} occurrence(s) in {p.relative_to(repo_root()).as_posix()}"


@mcp.tool()
def mkdir(path: str) -> str:
    """Create a directory (including parents). Idempotent."""
    p = ensure_under_root(path)
    p.mkdir(parents=True, exist_ok=True)
    return f"mkdir {p.relative_to(repo_root()).as_posix()}"


if __name__ == "__main__":
    mcp.run()
