"""Coding-agent client.

Claude-Code-shaped: the agent has a shell and a filesystem, no
pre-imported helpers, no pre-bound primitives. It writes Python files
to a scratch directory and runs them via `run_shell`, observing
stdout / stderr the way a human would.

Tools:
  - write_file(path, content) -> str
  - read_file(path) -> str
  - run_shell(cmd) -> str

System prompt embeds SKILL.md, which describes Chroma / BM25 / the
cross-encoder conceptually and shows the imports the agent will need.

This is the "Coding agent" column in segment 1's four-quadrant slide.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from shared import estimate_cost, get_llm
from .common import ClientResult


_PKG_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PKG_DIR.parent.parent
_SKILL_PATH = _PKG_DIR / "SKILL.md"

# Budget knobs — kept together so the prompt, the server-side cap, and the
# LangGraph recursion limit can't drift apart. Each shell call costs ~2
# messages (AIMessage + ToolMessage), plus inspection turns and the final
# answer; recursion_limit must comfortably exceed 2 * MAX_SHELL_CALLS.
MAX_SHELL_CALLS = 12
RECURSION_LIMIT = 40


def _load_skill() -> str:
    try:
        return _SKILL_PATH.read_text()
    except FileNotFoundError:
        return "(SKILL.md missing)"


SYSTEM_PROMPT_TEMPLATE = """You are a coding agent with a shell and a filesystem.

You have three tools:
  - write_file(path, content): write a file inside your scratch directory
  - read_file(path): read a file (relative paths resolve inside scratch)
  - run_shell(cmd): run a shell command inside scratch (60s timeout)

Nothing is pre-imported for you. Read the skill below to learn what's
on disk and which Python packages to use. Then write a script, run it,
read the output, iterate, and finally answer the user.

================ SKILL.md ================
{skill}
================ end SKILL.md ================

Use AT MOST {max_shell} run_shell calls."""


def _safe_path(scratch: Path, p: str) -> Path:
    """Resolve `p` against scratch, refusing escapes outside scratch.

    Compares the *joined-but-not-resolved* path against scratch so that
    symlinks placed inside scratch (e.g. `notebooks/data` -> repo
    data dir) traverse normally; absolute paths and `..` escapes still
    fail because the joined path lands outside scratch.
    """
    if Path(p).is_absolute():
        raise ValueError(f"path {p!r} must be relative")
    joined = (scratch / p)
    # Normalize without following symlinks; reject if it climbs above scratch.
    norm = Path(os.path.normpath(joined))
    try:
        norm.relative_to(scratch)
    except ValueError as exc:
        raise ValueError(f"path {p!r} escapes scratch dir") from exc
    return norm


def run(question: str, model_slug: str = "openai/gpt-5.4-nano") -> ClientResult:
    scratch = Path(tempfile.mkdtemp(prefix="coding_agent_"))
    # Make repo-relative paths in SKILL.md (e.g. `notebooks/data/...`)
    # resolve from inside scratch by symlinking the data tree. Imports of
    # `corpus` / `retrievers` work via PYTHONPATH below; only filesystem
    # reads need this.
    (scratch / "notebooks").mkdir(parents=True, exist_ok=True)
    (scratch / "notebooks" / "data").symlink_to(
        _REPO_ROOT / "notebooks" / "data"
    )
    shell_calls = {"n": 0}
    env = os.environ.copy()
    src_dir = str(_REPO_ROOT / "src")
    notebooks_dir = str(_REPO_ROOT / "notebooks")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join([p for p in [src_dir, notebooks_dir, existing] if p])

    @tool
    def write_file(path: str, content: str) -> str:
        """Write `content` to `path` inside the scratch directory.

        Relative paths only. Returns the absolute path written, or an
        error string starting with [ERROR].
        """
        try:
            target = _safe_path(scratch, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return f"wrote {len(content)} bytes to {target}"
        except Exception as exc:  # noqa: BLE001
            return f"[ERROR] {type(exc).__name__}: {exc}"

    @tool
    def read_file(path: str) -> str:
        """Read a file. Returns its text (truncated to 4000 chars) or [ERROR]."""
        try:
            target = _safe_path(scratch, path)
            text = target.read_text()
            if len(text) > 4000:
                text = text[:4000] + "\n...[truncated]"
            return text
        except Exception as exc:  # noqa: BLE001
            return f"[ERROR] {type(exc).__name__}: {exc}"

    @tool
    def run_shell(cmd: str) -> str:
        """Run a shell command in the scratch directory.

        Returns combined stdout+stderr (truncated to 4000 chars).
        60-second timeout. PYTHONPATH includes the repo's src/ and
        notebooks/ so imports of `corpus`, `retrievers`, etc.
        resolve.
        """
        if shell_calls["n"] >= MAX_SHELL_CALLS:
            return (
                f"[ERROR] shell budget exhausted ({MAX_SHELL_CALLS} calls); "
                "answer the user now from what you have."
            )
        shell_calls["n"] += 1
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(scratch),
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "[ERROR] command timed out after 60s"
        out = (proc.stdout or "") + (("\n[stderr] " + proc.stderr) if proc.stderr else "")
        if len(out) > 4000:
            out = out[:4000] + "\n...[truncated]"
        return out or f"(no output, exit={proc.returncode})"

    agent = create_react_agent(
        model=get_llm(model_slug),
        tools=[write_file, read_file, run_shell],
        prompt=SYSTEM_PROMPT_TEMPLATE.format(
            skill=_load_skill(), max_shell=MAX_SHELL_CALLS
        ),
    )

    t0 = time.time()
    try:
        out = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": RECURSION_LIMIT},
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    elapsed = time.time() - t0

    msgs = out["messages"]
    final = msgs[-1].content if msgs else ""
    if isinstance(final, list):
        final = "\n".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in final)

    in_t = out_t = 0
    for m in msgs:
        um = getattr(m, "usage_metadata", None) or {}
        in_t += int(um.get("input_tokens", 0) or 0)
        out_t += int(um.get("output_tokens", 0) or 0)

    return ClientResult(
        client="via_coding_agent",
        answer=str(final),
        n_tool_calls=shell_calls["n"],
        tool_latency_total_s=0.0,
        total_latency_s=elapsed,
        input_tokens=in_t,
        output_tokens=out_t,
        cost_usd=estimate_cost(model_slug, in_t, out_t),
        raw_messages=msgs,
    )
