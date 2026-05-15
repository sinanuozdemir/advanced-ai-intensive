"""Forge shell MCP server.

One tool: ``exec``. Runs a shell command in the repo root with a hard
timeout. Permission gating happens at the agent-loader layer, not here; the
server's only safety net is the timeout and the ``FORGE_REPO`` chroot to cwd.

A second tool ``allowlist`` returns the configured shell allowlist (read
from ``.forge/config.toml``) so the agent can self-check before calling
``exec``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from _common import repo_root

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("forge-shell")

DEFAULT_TIMEOUT = 30


def _load_allowlist() -> list[str]:
    """Read shell_allowlist from ``.forge/config.toml`` if present."""
    cfg_path = repo_root() / ".forge" / "config.toml"
    if not cfg_path.is_file():
        return []
    try:
        try:
            import tomllib  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]
        with cfg_path.open("rb") as fh:
            data = tomllib.load(fh)
        return list(data.get("permissions", {}).get("shell_allowlist", []) or [])
    except Exception:  # noqa: BLE001
        return []


@mcp.tool()
def exec(command: str, cwd: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Run a shell command in the repo root (or an explicit cwd inside it).

    Returns ``{exit_code, stdout, stderr, truncated}``. Output is capped at
    16k chars per stream to keep the LLM context manageable.

    Args:
        command: A shell command string. Executed via /bin/sh -c.
        cwd: Optional working directory (must be inside the repo).
        timeout: Seconds before the command is killed (default 30, max 120).
    """
    timeout = max(1, min(int(timeout or DEFAULT_TIMEOUT), 120))
    wd = Path(cwd).resolve() if cwd else repo_root()
    try:
        wd.relative_to(repo_root())
    except ValueError:
        return {
            "exit_code": -1, "stdout": "", "truncated": False,
            "stderr": f"refused: cwd {wd} escapes repo root",
        }
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(wd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": -1,
            "stdout": (exc.stdout or "")[:16_000],
            "stderr": f"TIMEOUT after {timeout}s",
            "truncated": False,
        }
    cap = 16_000
    out = proc.stdout or ""
    err = proc.stderr or ""
    truncated = len(out) > cap or len(err) > cap
    return {
        "exit_code": proc.returncode,
        "stdout": out[:cap],
        "stderr": err[:cap],
        "truncated": truncated,
    }


@mcp.tool()
def allowlist() -> list[str]:
    """Return the configured shell allowlist (command prefixes the agent can
    run without an extra approval prompt)."""
    return _load_allowlist()


if __name__ == "__main__":
    mcp.run()
