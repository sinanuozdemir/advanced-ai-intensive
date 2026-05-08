"""Smoke-test Claude CLI in subprocess/non-interactive mode.

Run:
    PYTHONPATH=src python -m mcp_demo.clients.claude_cli_smoke
"""

from __future__ import annotations

import shutil
import subprocess
import time


PROMPT = "Reply with exactly: CLI_SMOKE_OK"
TIMEOUT_S = 25


VARIANTS: list[list[str]] = [
    ["claude", "-p", PROMPT],
    ["claude", "--print", PROMPT],
    ["claude", "-p", "--output-format", "json", PROMPT],
    ["claude", "--print", "--output-format", "json", PROMPT],
    ["claude", "-p", "--permission-mode", "dontAsk", PROMPT],
    ["claude", "-p", "--no-session-persistence", PROMPT],
    ["claude", "-p", "--tools", "", PROMPT],
    ["claude", "-p", "--tools", "", "--output-format", "json", PROMPT],
]


def _fmt(text: str, limit: int = 220) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def main() -> None:
    if shutil.which("claude") is None:
        print("claude binary not found on PATH")
        return

    print("Claude CLI smoke test")
    print(f"Prompt: {PROMPT!r}")
    print(f"Timeout per variant: {TIMEOUT_S}s")
    print("=" * 70)

    for cmd in VARIANTS:
        print("\n$ " + " ".join(repr(tok) if tok == "" else tok for tok in cmd))
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
                check=False,
            )
            elapsed = time.time() - t0
            print(f"status: EXIT {proc.returncode} in {elapsed:.2f}s")
            print("stdout:", _fmt(proc.stdout))
            print("stderr:", _fmt(proc.stderr))
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"status: TIMEOUT after {elapsed:.2f}s")


if __name__ == "__main__":
    main()
