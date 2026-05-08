"""Subprocess wrapper that hands the same task to a CLI coding agent.

Supported CLIs (pinned per the plan):
- `claude`     - Claude Code (Anthropic). Polished, single-vendor, opinionated.
- `opencode`   - opencode (open-source, multi-model). Configurable, hackable.

Usage:
    result = via_cli.run("explain the RAG architecture", cli="claude")
    print(result.answer, result.total_latency_s)

We run with `-p` (Claude Code) / `--non-interactive` (opencode) so the
CLI emits its final answer to stdout instead of opening an interactive REPL.
Either binary must be installed and on PATH; we degrade gracefully (the
returned `ClientResult.answer` is `"CLI not installed: ..."`) so notebook 1
still renders without forcing every student to install both tools.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import json

from .common import ClientResult

_CLI_COMMANDS: dict[str, list[list[str]]] = {
    # Claude CLI variants observed in the wild. Try in order.
    "claude": [
        # Robust subprocess mode: feed prompt on stdin instead of argv.
        ["claude", "-p", "--input-format", "text", "--tools", "", "--disable-slash-commands", "--no-session-persistence", "--output-format", "json"],
        ["claude", "--print", "--input-format", "text", "--tools", "", "--disable-slash-commands", "--no-session-persistence", "--output-format", "json"],
        # Compatibility fallbacks.
        ["claude", "-p", "--input-format", "text"],
        ["claude", "--print", "--input-format", "text"],
    ],
    # opencode's non-interactive mode; `run` is the most stable.
    "opencode": [
        ["opencode", "run"],
    ],
}


def run(
    question: str,
    *,
    cli: str = "claude",
    cwd: str | None = None,
    timeout_s: int = 120,
) -> ClientResult:
    """Hand `question` to a CLI agent and capture its stdout as the answer.

    Parameters
    ----------
    question : str
        The user task. Forwarded as the trailing positional argument.
    cli : {"claude", "opencode"}
        Which CLI to invoke. Must be installed on PATH.
    cwd : str | None
        Working directory for the CLI. Defaults to the current process cwd.
        Important for coding tasks: the CLI will see this as its project root.
    timeout_s : int
        Hard kill after this many seconds.

    Returns
    -------
    ClientResult with `client="via_cli"`. Token counts are 0 because CLI
    transcripts don't expose per-call token usage; cost is 0 for the same
    reason. The plan calls this out explicitly in the trade-off matrix.
    """
    if cli not in _CLI_COMMANDS:
        raise ValueError(f"Unsupported cli={cli!r}; choose from {list(_CLI_COMMANDS)}")
    cmd_variants = _CLI_COMMANDS[cli]
    binary = cmd_variants[0][0]
    if shutil.which(binary) is None:
        return ClientResult(
            client=f"via_cli:{cli}",
            answer=f"CLI not installed: `{binary}` is not on PATH. "
                   f"See README for install instructions.",
            total_latency_s=0.0,
        )

    last_timeout = False
    start = time.time()
    for i, cmd_base in enumerate(cmd_variants):
        # Claude is more reliable in subprocess mode when prompt is fed on stdin.
        if cli == "claude":
            cmd = cmd_base
            input_text = question
            stdin_stream = None
        else:
            cmd = cmd_base + [question]
            input_text = None
            stdin_stream = subprocess.DEVNULL
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                stdin=stdin_stream,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            elapsed = time.time() - start
            answer = (proc.stdout or "").strip()
            # When Claude is run with `--output-format json`, normalize to plain text.
            # Also surface CLI-side failures as explicit error strings.
            if answer.startswith("{"):
                try:
                    payload = json.loads(answer)
                    if isinstance(payload, dict):
                        if payload.get("is_error") is True:
                            subtype = payload.get("subtype") or "unknown_error"
                            answer = f"CLI ERROR ({subtype})"
                        else:
                            # Try the common result-bearing keys.
                            answer = (
                                str(payload.get("result"))
                                if payload.get("result") is not None
                                else str(payload.get("content", answer))
                            )
                except Exception:
                    pass
            if not answer and proc.stderr:
                answer = f"(no stdout) stderr: {proc.stderr.strip()[:500]}"
            n_tool_calls = _count_tool_calls(proc.stdout)
            return ClientResult(
                client=f"via_cli:{cli}",
                answer=answer,
                n_tool_calls=n_tool_calls,
                tool_latency_total_s=0.0,    # opaque from the outside
                total_latency_s=elapsed,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                raw_messages=[],
            )
        except subprocess.TimeoutExpired:
            last_timeout = True
            # Try fallback variant (e.g. `-p`) before giving up.
            continue

    elapsed = time.time() - start
    if last_timeout:
        return ClientResult(
            client=f"via_cli:{cli}",
            answer=f"TIMEOUT after {timeout_s}s",
            total_latency_s=elapsed,
        )
    return ClientResult(
        client=f"via_cli:{cli}",
        answer=f"CLI invocation failed for {cli}. Tried variants: {cmd_variants}",
        total_latency_s=elapsed,
    )


def _count_tool_calls(stdout: str | None) -> int:
    """Best-effort tool-call count by scraping common transcript markers.

    Both `claude --print` and `opencode run` emit human-readable lines like
    `> Read(...)` or `Tool: read_file(...)`. We count occurrences. Imperfect
    by design — the trade-off matrix in the plan lists "transcript you have
    to parse" as the CLI client's defining limitation.
    """
    if not stdout:
        return 0
    count = 0
    for marker in ("Tool:", "→ Tool", "> Read(", "> Write(", "> Bash(", "🔧"):
        count += stdout.count(marker)
    return count
