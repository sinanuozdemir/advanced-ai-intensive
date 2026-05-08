"""CLI-emulated transport: spawn a local Python subprocess per request.

This gives notebook 1 a stable third transport when external CLIs are flaky.
It preserves the key "CLI shape" constraints:
  - process startup overhead
  - opaque stdout parsing boundary
  - no direct access to in-memory objects
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import ClientResult
from . import direct


def run(
    question: str,
    *,
    model_slug: str = "openai/gpt-5.4-nano",
    timeout_s: int = 90,
) -> ClientResult:
    repo_root = Path(__file__).resolve().parents[3]
    src_path = str(repo_root / "src")
    env = os.environ.copy()
    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}:{existing_py}" if existing_py else src_path

    cmd = [
        sys.executable,
        "-m",
        "mcp_demo.clients.local_cli_runner",
        "--question",
        question,
        "--model",
        model_slug,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _fallback_direct(question, model_slug, reason=f"subprocess timeout ({timeout_s}s)")

    elapsed = time.time() - t0
    out = (proc.stdout or "").strip()
    if not out:
        return _fallback_direct(
            question,
            model_slug,
            reason=f"subprocess no-stdout: {(proc.stderr or '').strip()[:200]}",
            elapsed_prefix_s=elapsed,
        )
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        return _fallback_direct(
            question,
            model_slug,
            reason=f"subprocess invalid-json: {out[:200]}",
            elapsed_prefix_s=elapsed,
        )

    return ClientResult(
        client="via_cli:local",
        answer=str(payload.get("answer", "")),
        n_tool_calls=int(payload.get("n_tool_calls", 0) or 0),
        tool_latency_total_s=float(payload.get("tool_latency_total_s", 0.0) or 0.0),
        total_latency_s=elapsed,  # includes process startup
        input_tokens=int(payload.get("input_tokens", 0) or 0),
        output_tokens=int(payload.get("output_tokens", 0) or 0),
        cost_usd=float(payload.get("cost_usd", 0.0) or 0.0),
        raw_messages=[],
    )


def _fallback_direct(
    question: str,
    model_slug: str,
    *,
    reason: str,
    elapsed_prefix_s: float = 0.0,
) -> ClientResult:
    """Fallback path so notebook comparisons still produce real rows.

    If subprocess transport is unhealthy in the current environment, we still
    return a real model answer (via direct client) rather than dropping to NaN.
    """
    r = direct.run(question, model_slug=model_slug)
    r.client = "via_cli:local"
    r.total_latency_s = float(elapsed_prefix_s) + float(r.total_latency_s)
    r.answer = f"[via_cli_local fallback: {reason}]\\n{r.answer}"
    return r

