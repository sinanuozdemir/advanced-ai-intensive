"""FastMCP wrapper around the agent_api HTTP service.

This server is a **thin HTTP client**, not a re-implementation of the
workflow. It is the production pattern: the workflow runs in one process
(with its own dependencies, observability, and scaling), and any agent
that speaks MCP can drive it through a stable tool surface.

Configuration via env vars:

    AGENT_API_BASE_URL   default http://localhost:8090
    AGENT_API_TIMEOUT_S  default 600 (long because the loop can run minutes)

Install into Forge by dropping this file into ``<repo>/.forge/mcp_servers/``
and adding a ``UserMcpServer`` entry to ``<repo>/.forge/mcp_servers.json``.
See the README next to this file for the exact JSON shape.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("research-workflow")


def _base_url() -> str:
    return os.environ.get("AGENT_API_BASE_URL", "http://localhost:8090").rstrip("/")


def _timeout_s() -> float:
    try:
        return float(os.environ.get("AGENT_API_TIMEOUT_S", "600"))
    except ValueError:
        return 600.0


def _client() -> httpx.Client:
    return httpx.Client(base_url=_base_url(), timeout=_timeout_s())


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text}


@mcp.tool()
def research(
    topic: str,
    max_iterations: int = 3,
) -> dict[str, Any]:
    """Run a deep-research workflow against the agent_api service.

    The workflow plans focused sub-queries, runs a real-internet research
    agent (SerpAPI + Firecrawl) per sub-query, reflects on whether the
    findings answer the topic, and loops up to ``max_iterations`` times
    before writing the final cited report.

    SYNCHRONOUS — blocks until the workflow finishes (typically 30-180s).

    Args:
        topic: The research question or topic to brief.
        max_iterations: Cap on plan -> agent -> reflect rounds.
    """
    body = {
        "topic": topic,
        "max_iterations": int(max_iterations),
    }
    try:
        with _client() as client:
            resp = client.post("/research", json=body)
    except httpx.HTTPError as exc:
        return {"error": f"agent_api unreachable: {type(exc).__name__}: {exc}"}
    if resp.status_code >= 400:
        return {"error": f"HTTP {resp.status_code}", "detail": _safe_json(resp)}
    data = _safe_json(resp)
    artifact = data.get("artifact") or {}
    return {
        "artifact_id": data.get("artifact_id"),
        "request_id": data.get("request_id"),
        "outcome": data.get("outcome"),
        "rounds": data.get("rounds"),
        "findings_count": data.get("findings_count"),
        "summary": (artifact.get("final_draft") or "")[:1500],
        "full_artifact_endpoint": f"/artifacts/{data.get('artifact_id')}",
    }


@mcp.tool()
def get_artifact(artifact_id: str) -> dict[str, Any]:
    """Fetch a previously-saved artifact and its full provenance.

    Use this after ``research`` returns an artifact_id, or to retrieve any
    artifact from a previous workflow run. Returns the final draft text,
    the shape it was written against, every iteration's grade and critique,
    and metadata.

    Args:
        artifact_id: The UUID returned by ``research`` or ``list_artifacts``.
    """
    try:
        with _client() as client:
            resp = client.get(f"/artifacts/{artifact_id}")
    except httpx.HTTPError as exc:
        return {"error": f"agent_api unreachable: {type(exc).__name__}: {exc}"}
    if resp.status_code == 404:
        return {"error": f"artifact {artifact_id} not found"}
    if resp.status_code >= 400:
        return {"error": f"HTTP {resp.status_code}", "detail": _safe_json(resp)}
    return _safe_json(resp)


@mcp.tool()
def list_artifacts(limit: int = 20, offset: int = 0) -> dict[str, Any]:
    """List saved artifacts, newest first.

    Args:
        limit: Max items to return (1-100).
        offset: Skip this many items for pagination.
    """
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    try:
        with _client() as client:
            resp = client.get("/artifacts", params={"limit": limit, "offset": offset})
    except httpx.HTTPError as exc:
        return {"error": f"agent_api unreachable: {type(exc).__name__}: {exc}"}
    if resp.status_code >= 400:
        return {"error": f"HTTP {resp.status_code}", "detail": _safe_json(resp)}
    return _safe_json(resp)


if __name__ == "__main__":
    mcp.run()
