"""Three clients that all answer the same question via different transport.

- `direct.run(question, model_slug)` -> answer
- `via_mcp.run(question, model_slug)` -> answer (talks to mcp_demo.server via stdio)
- `via_cli.run(question, cli="claude" | "opencode")` -> answer (subprocess wrapper)

All three return the same `ClientResult` dataclass so notebook 1 can
compare them apples-to-apples in the eval harness.
"""

from .common import ClientResult

__all__ = ["ClientResult"]
