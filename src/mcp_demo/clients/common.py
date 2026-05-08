"""Shared shape returned by every client in this package."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClientResult:
    """Uniform return shape from `direct.run`, `via_mcp.run`, `via_cli.run`."""

    client: str             # "direct" | "via_mcp" | "via_cli"
    answer: str
    n_tool_calls: int = 0
    tool_latency_total_s: float = 0.0
    total_latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    raw_messages: list = field(default_factory=list)
