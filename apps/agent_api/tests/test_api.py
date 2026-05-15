"""HTTP-layer tests using FastAPI's TestClient.

The workflow itself is exercised in ``test_workflow.py``. Here we focus on
the shape of the API surface:

* ``/healthz`` is always 200.
* ``/readyz`` is 503 when the LLM smoke-check fails or search tools
  aren't configured; 200 when everything passes.
* ``/research`` end-to-end with stubbed deps lands an artifact in the
  store and the response shape matches ``ResearchResponse``.
* ``/artifacts`` and ``/artifacts/{id}`` round-trip a saved artifact.
* ``/metrics`` returns Prometheus text including our counter names.
* ``/trace`` filters log lines by request_id and returns plain text.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.schemas import PlanStep, ReflectVerdict, ResearchPlan


PLAN = ResearchPlan(
    steps=[
        PlanStep(query="what is the topic?", intent="define the topic"),
        PlanStep(query="why does it matter?", intent="impact"),
    ],
    rationale="test plan",
)
VERDICT_DONE = ReflectVerdict(
    done=True, reasoning="passed", missing_questions=[],
)


class _StubPlan:
    def invoke(self, messages):  # noqa: ARG002
        return PLAN


class _StubAgent:
    def invoke(self, state):  # noqa: ARG002
        class R:
            answer = "stub answer [https://example.org/x]"
        return R()


class _StubReflect:
    def invoke(self, messages):  # noqa: ARG002
        return VERDICT_DONE


class _StubArtifact:
    def invoke(self, messages):  # noqa: ARG002
        class R:
            content = "# test artifact\n\nbody [https://example.org/x]"
        return R()


@pytest.fixture
def client(tmp_settings, monkeypatch):
    """Boot the FastAPI app with stub LLMs + agent so /research runs offline."""

    monkeypatch.setenv("SERPAPI_API_KEY", "sk-test-fake")
    tmp_settings.serpapi_api_key = "sk-test-fake"

    from app import deps as deps_mod

    monkeypatch.setattr(deps_mod, "llm_smoke_check", lambda s: (True, "stubbed"))
    monkeypatch.setattr(deps_mod, "search_tools_check", lambda s: (True, "stubbed"))
    monkeypatch.setattr(deps_mod, "build_plan_llm", lambda s: _StubPlan())
    monkeypatch.setattr(deps_mod, "build_research_agent", lambda s: _StubAgent())
    monkeypatch.setattr(deps_mod, "build_reflect_llm", lambda s: _StubReflect())
    monkeypatch.setattr(deps_mod, "build_artifact_llm", lambda s: _StubArtifact())

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as c:
        yield c


def test_healthz_always_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz_green_when_all_checks_pass(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["checks"]["sqlite"]["ok"] is True
    assert payload["checks"]["llm"]["ok"] is True
    assert payload["checks"]["search_tools"]["ok"] is True
    assert payload["checks"]["workflow"]["ok"] is True


def test_readyz_503_when_llm_unavailable(tmp_settings, monkeypatch):
    """If the LLM key is missing the graph won't compile; readiness 503s."""
    from app import deps as deps_mod
    monkeypatch.setattr(
        deps_mod, "llm_smoke_check",
        lambda s: (False, "OPENROUTER_API_KEY is not set"),
    )
    from app.main import app as fastapi_app
    with TestClient(fastapi_app) as c:
        r = c.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["llm"]["ok"] is False
    assert body["checks"]["workflow"]["ok"] is False


def test_readyz_503_when_serpapi_key_missing(tmp_settings, monkeypatch):
    """SerpAPI key is required for real web search; missing => not ready."""
    from app import deps as deps_mod
    monkeypatch.setattr(deps_mod, "llm_smoke_check", lambda s: (True, "ok"))
    monkeypatch.setattr(
        deps_mod, "search_tools_check",
        lambda s: (False, "SERPAPI_API_KEY is not set"),
    )
    from app.main import app as fastapi_app
    with TestClient(fastapi_app) as c:
        r = c.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["checks"]["search_tools"]["ok"] is False


def test_research_runs_end_to_end(client):
    r = client.post("/research", json={"topic": "Test topic for the workflow"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outcome"] == "complete"
    assert body["rounds"] == 1
    assert body["findings_count"] == 2  # 2 plan steps -> 2 findings
    assert body["artifact"]["topic"] == "Test topic for the workflow"
    assert body["artifact"]["final_draft"].startswith("# test artifact")
    assert body["request_id"]


def test_research_request_id_threads_through_response_header(client):
    r = client.post("/research", json={"topic": "header-thread test"})
    assert "x-request-id" in r.headers
    assert r.headers["x-request-id"] == r.json()["request_id"]


def test_get_and_list_artifacts(client):
    posted = client.post("/research", json={"topic": "list me"})
    art_id = posted.json()["artifact_id"]

    r = client.get(f"/artifacts/{art_id}")
    assert r.status_code == 200
    assert r.json()["artifact_id"] == art_id

    r = client.get("/artifacts")
    assert r.status_code == 200
    listing = r.json()
    assert listing["total"] >= 1
    assert any(item["artifact_id"] == art_id for item in listing["items"])


def test_get_artifact_404(client):
    r = client.get("/artifacts/does-not-exist")
    assert r.status_code == 404


def test_metrics_exposes_workflow_counters(client):
    client.post("/research", json={"topic": "for metrics"})
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "agent_workflow_runs_total" in body
    assert "agent_reflect_rounds_total" in body
    assert "agent_workflow_latency_seconds" in body
    assert 'agent_workflow_runs_total{outcome="complete"}' in body


def test_trace_filters_by_request_id(client, tmp_settings):
    posted = client.post("/research", json={"topic": "trace me"})
    rid = posted.json()["request_id"]
    r = client.get("/trace", params={"request_id": rid})
    assert r.status_code == 200
    text = r.text
    assert rid in text
    assert "workflow.start" in text or "node.start" in text


def test_trace_empty_for_unknown_request_id(client):
    r = client.get("/trace", params={"request_id": "no-such-id-12345"})
    assert r.status_code == 200
    assert "no log lines" in r.text.lower()
