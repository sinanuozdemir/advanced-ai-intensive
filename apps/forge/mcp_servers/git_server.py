"""Forge git MCP server.

Tools: status, diff, log, branch, add, commit, reset.

These are thin wrappers around ``git`` invoked via subprocess at the repo
root. ``commit`` / ``reset`` / push-class operations are still gated by the
agent-side PermissionBroker before being called.
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from _common import ensure_under_root, repo_root

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("forge-git")


def _git(args: list[str], timeout: int = 30) -> dict:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return {"exit_code": 127, "stdout": "", "stderr": "git not on PATH"}
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": f"git timed out after {timeout}s"}
    return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


@mcp.tool()
def status() -> dict:
    """``git status --porcelain --branch`` against the repo root."""
    return _git(["status", "--porcelain=v1", "--branch"])


@mcp.tool()
def diff(path: str | None = None, staged: bool = False) -> dict:
    """``git diff`` (or ``git diff --staged``) optionally scoped to ``path``."""
    args = ["diff"]
    if staged:
        args.append("--staged")
    if path:
        args.append("--")
        args.append(str(ensure_under_root(path).relative_to(repo_root())))
    return _git(args)


@mcp.tool()
def log(n: int = 10, oneline: bool = True) -> dict:
    """``git log`` — last ``n`` commits."""
    n = max(1, min(int(n or 10), 200))
    args = ["log", f"-n{n}"]
    if oneline:
        args.append("--oneline")
    return _git(args)


@mcp.tool()
def branch() -> dict:
    """``git branch --show-current``."""
    return _git(["branch", "--show-current"])


@mcp.tool()
def add(paths: list[str]) -> dict:
    """Stage paths. Each path must resolve inside the repo."""
    if not paths:
        return {"exit_code": 1, "stdout": "", "stderr": "no paths"}
    rels = []
    for p in paths:
        try:
            rels.append(str(ensure_under_root(p).relative_to(repo_root())))
        except PermissionError as exc:
            return {"exit_code": 1, "stdout": "", "stderr": str(exc)}
    return _git(["add", *rels])


@mcp.tool()
def commit(message: str, paths: list[str] | None = None) -> dict:
    """``git commit -m <message>`` optionally restricted to ``paths``."""
    if not message or not message.strip():
        return {"exit_code": 1, "stdout": "", "stderr": "empty commit message"}
    args = ["commit", "-m", message]
    if paths:
        rels: list[str] = []
        for p in paths:
            try:
                rels.append(str(ensure_under_root(p).relative_to(repo_root())))
            except PermissionError as exc:
                return {"exit_code": 1, "stdout": "", "stderr": str(exc)}
        args.append("--")
        args.extend(rels)
    return _git(args)


@mcp.tool()
def reset(target: str = "HEAD", hard: bool = False) -> dict:
    """``git reset`` to ``target``. ``hard=True`` discards working-tree changes
    — gated by the permission broker."""
    args = ["reset"]
    if hard:
        args.append("--hard")
    args.append(target)
    return _git(args)


@mcp.tool()
def push(remote: str = "origin", branch_name: str = "") -> dict:
    """``git push`` to ``remote`` / ``branch_name``. Almost always denied by
    the default permissions; included for completeness."""
    args = ["push", remote]
    if branch_name:
        args.append(branch_name)
    return _git(args, timeout=60)


if __name__ == "__main__":
    mcp.run()
