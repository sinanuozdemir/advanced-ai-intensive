"""src/mcp_demo — teaching MCP server + three clients (direct, mcp, cli).

Named `mcp_demo` (not `mcp`) because the official MCP Python SDK ships as
the top-level `mcp` package and we cannot shadow it.

Distinct from the production HubSpot/email/research MCP servers in
`apps/sdr_multi_agent/mcp_servers/` — this one is a pedagogical example that
exposes the Week 1 hybrid retriever so we can compare three ways of letting
an agent reach a tool.
"""
