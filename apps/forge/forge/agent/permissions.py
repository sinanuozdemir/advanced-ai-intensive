"""Permission broker.

Every MCP tool call goes through ``PermissionBroker.gate(tool_name, args, agent_name)``
before the underlying tool function runs. The broker consults three layers:

1. **Per-agent allowlist** — persistent agents declare their permitted tools in
   ``.forge/agents/<name>.toml``. A tool *not* listed is denied outright. Main
   and ephemeral agents skip this gate (they inherit the user's trust).
2. **Per-tool override** — ``permissions.tools[<tool_name>]`` in the main
   config: "allow" / "ask" / "deny".
3. **Default** — ``permissions.default`` (defaults to "ask").

Result is one of:

- ``"allow"``  — proceed.
- ``"deny"``   — refuse with ``PermissionDenied``.
- ``"ask"``    — invoke the registered approver coroutine. If none registered
  (e.g. forge running headless without a TUI), fall back to deny.

Every decision is appended to ``<repo>/.forge/audit.jsonl`` for after-the-fact
review (and for the Electron app's audit panel).
"""
from __future__ import annotations

import asyncio
import json
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Protocol

from ..config import PermissionsConfig, PermissionDecision
from ..paths import ForgePaths


GateOutcome = Literal["allow", "deny"]


class PermissionDenied(PermissionError):
    """Raised when the broker denies a tool call."""

    def __init__(self, tool_name: str, agent_name: str, reason: str) -> None:
        super().__init__(f"denied {tool_name} for agent {agent_name}: {reason}")
        self.tool_name = tool_name
        self.agent_name = agent_name
        self.reason = reason


class ApproverProtocol(Protocol):
    """Anything that can satisfy an 'ask' decision asynchronously."""

    async def __call__(
        self, *, tool_name: str, args: dict, agent_name: str, reason: str
    ) -> bool: ...


@dataclass
class PermissionBroker:
    """Permission broker. Construct once at TUI startup; share across agents."""

    paths: ForgePaths
    cfg: PermissionsConfig
    persistent_allowlists: dict[str, set[str]] = field(default_factory=dict)
    approver: ApproverProtocol | None = None
    audit_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ---------------------------------------------------------------- registry

    def set_persistent_allowlist(self, agent_name: str, tools: list[str]) -> None:
        """Persistent agents declare their tools explicitly in their TOML."""
        self.persistent_allowlists[agent_name] = set(tools)

    def set_approver(self, approver: ApproverProtocol | None) -> None:
        """Wire (or unwire) the TUI / Electron modal that resolves 'ask'."""
        self.approver = approver

    # ---------------------------------------------------------------- decide

    def _static_decision(self, tool_name: str, agent_name: str) -> PermissionDecision:
        # Persistent agents: deny unless explicitly listed.
        if agent_name not in {"main", "ephemeral"}:
            allowlist = self.persistent_allowlists.get(agent_name)
            if allowlist is None or tool_name not in allowlist:
                return "deny"
        # Per-tool overrides win over default.
        override = self.cfg.tools.get(tool_name)
        if override is not None:
            return override
        return self.cfg.default

    def _shell_allowlisted(self, args: dict) -> bool:
        if not args:
            return False
        cmd = args.get("command") or ""
        if not isinstance(cmd, str) or not cmd.strip():
            return False
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()
        if not tokens:
            return False
        for allowed in self.cfg.shell_allowlist:
            allowed_tokens = allowed.split()
            if tokens[: len(allowed_tokens)] == allowed_tokens:
                return True
        return False

    async def gate(
        self,
        *,
        tool_name: str,
        args: dict | None = None,
        agent_name: str = "main",
    ) -> GateOutcome:
        """Decide whether to allow the tool call. Raises ``PermissionDenied``
        on deny. Returns ``"allow"`` on allow."""
        args = args or {}
        decision: PermissionDecision = self._static_decision(tool_name, agent_name)
        reason = "config default"
        # shell allowlist promotes an "ask" to "allow" for safe-prefix commands.
        if (
            decision == "ask"
            and tool_name in {"shell_exec", "shell.exec"}
            and self._shell_allowlisted(args)
        ):
            decision = "allow"
            reason = "shell allowlist match"

        if decision == "allow":
            await self._audit(
                tool_name=tool_name, agent_name=agent_name,
                args=args, decision="allow", source=reason,
            )
            return "allow"

        if decision == "deny":
            await self._audit(
                tool_name=tool_name, agent_name=agent_name,
                args=args, decision="deny", source=reason,
            )
            raise PermissionDenied(tool_name, agent_name, reason)

        # ask → defer to the approver
        approved = False
        ask_reason = "no approver registered (headless)"
        if self.approver is not None:
            try:
                approved = bool(await self.approver(
                    tool_name=tool_name, args=args,
                    agent_name=agent_name, reason="config says ask",
                ))
                ask_reason = "human approved" if approved else "human denied"
            except Exception as exc:  # noqa: BLE001
                approved = False
                ask_reason = f"approver failed: {exc!r}"
        await self._audit(
            tool_name=tool_name, agent_name=agent_name,
            args=args, decision="allow" if approved else "deny",
            source=ask_reason,
        )
        if not approved:
            raise PermissionDenied(tool_name, agent_name, ask_reason)
        return "allow"

    # ---------------------------------------------------------------- audit

    async def _audit(
        self,
        *,
        tool_name: str,
        agent_name: str,
        args: dict,
        decision: GateOutcome,
        source: str,
    ) -> None:
        if not self.paths.audit_jsonl.parent.exists():
            self.paths.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "agent": agent_name,
            "decision": decision,
            "source": source,
            "args": _truncate_args(args),
        }
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        async with self.audit_lock:
            await asyncio.to_thread(
                self._append_line, self.paths.audit_jsonl, line,
            )

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def _truncate_args(args: dict, limit: int = 1024) -> dict:
    """Shrink any string arg longer than ``limit`` chars so the audit log
    doesn't bloat into MBs when a write tool dumps a whole file."""
    out: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > limit:
            out[k] = v[:limit] + f"... [truncated {len(v) - limit} chars]"
        else:
            out[k] = v
    return out


__all__ = [
    "PermissionBroker",
    "PermissionDenied",
    "ApproverProtocol",
    "GateOutcome",
]
