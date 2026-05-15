"""Tests for the FastMCP wrapper.

We exercise the underlying tool functions directly (FastMCP keeps them as
plain callables on the server's tool registry) and patch ``httpx.Client``
with ``respx`` so we can assert each tool calls the right URL with the
right body shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
import respx


_HERE = Path(__file__).resolve().parent
_MCP_DIR = _HERE.parent / "mcp_server"
sys.path.insert(0, str(_MCP_DIR))


@pytest.fixture(autouse=True)
def _base_url(monkeypatch):
    monkeypatch.setenv("AGENT_API_BASE_URL", "http://upstream.test:8090")
    monkeypatch.setenv("AGENT_API_TIMEOUT_S", "30")
    # Re-import the module fresh so the env vars take effect.
    sys.modules.pop("research_workflow_server", None)


def _import_module():
    import importlib
    mod = importlib.import_module("research_workflow_server")
    return mod


def _call_tool(server_mod, tool_name: str, **kwargs):
    """FastMCP keeps the underlying function on the wrapped tool object."""
    tool = next(
        t for t in server_mod.mcp._tool_manager._tools.values()
        if t.name == tool_name
    )
    return tool.fn(**kwargs)


@respx.mock
def test_research_tool_posts_correct_body():
    mod = _import_module()
    route = respx.post("http://upstream.test:8090/research").mock(
        return_value=httpx.Response(
            200,
            json={
                "artifact_id": "abc-123",
                "request_id": "rid-1",
                "outcome": "complete",
                "rounds": 2,
                "findings_count": 4,
                "artifact": {"final_draft": "BODY " * 600},
            },
        )
    )
    out = _call_tool(mod, "research", topic="hello", max_iterations=2)
    assert route.called
    body = route.calls[0].request.read().decode("utf-8")
    assert '"topic":"hello"' in body
    assert '"max_iterations":2' in body
    assert out["artifact_id"] == "abc-123"
    assert out["outcome"] == "complete"
    assert out["rounds"] == 2
    assert out["findings_count"] == 4
    # Summary truncated to the documented 1500 chars.
    assert len(out["summary"]) <= 1500


@respx.mock
def test_research_tool_returns_error_on_upstream_5xx():
    mod = _import_module()
    respx.post("http://upstream.test:8090/research").mock(
        return_value=httpx.Response(503, json={"error": "upstream down"})
    )
    out = _call_tool(mod, "research", topic="hello")
    assert "error" in out
    assert "503" in out["error"]


@respx.mock
def test_research_tool_returns_error_on_network_failure():
    mod = _import_module()
    respx.post("http://upstream.test:8090/research").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    out = _call_tool(mod, "research", topic="hello")
    assert "error" in out
    assert "unreachable" in out["error"]


@respx.mock
def test_get_artifact_tool_hits_correct_path():
    mod = _import_module()
    route = respx.get("http://upstream.test:8090/artifacts/xyz").mock(
        return_value=httpx.Response(200, json={"artifact_id": "xyz"})
    )
    out = _call_tool(mod, "get_artifact", artifact_id="xyz")
    assert route.called
    assert out == {"artifact_id": "xyz"}


@respx.mock
def test_get_artifact_404_returns_explicit_error():
    mod = _import_module()
    respx.get("http://upstream.test:8090/artifacts/missing").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    out = _call_tool(mod, "get_artifact", artifact_id="missing")
    assert "error" in out
    assert "missing" in out["error"]


@respx.mock
def test_list_artifacts_passes_pagination_params():
    mod = _import_module()
    route = respx.get("http://upstream.test:8090/artifacts").mock(
        return_value=httpx.Response(200, json={"items": [], "total": 0, "limit": 5, "offset": 10})
    )
    out = _call_tool(mod, "list_artifacts", limit=5, offset=10)
    assert route.called
    req_url = str(route.calls[0].request.url)
    assert "limit=5" in req_url
    assert "offset=10" in req_url
    assert out["limit"] == 5


