"""Workflow-level tests with stubbed LLMs.

We never touch the network. We build:

* a ``StubPlanLLM`` that returns a scripted ``ResearchPlan`` per call,
* a ``StubResearchAgent`` that returns a different cited answer per call,
* a ``StubReflectLLM`` driven by a scripted list of ``done`` flags,
* a ``StubArtifactLLM`` that just echoes back a deterministic markdown blob.

That's enough to assert: (a) the workflow finalizes when reflect says
``done``, (b) it caps when reflect keeps saying not-done, (c) the saved
artifact records every round in ``provenance``, (d) follow-up questions
from reflect flow into the next plan round.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.schemas import (
    Artifact,
    PlanStep,
    ReflectVerdict,
    ResearchPlan,
)
from app.workflow import build_workflow


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _plan(*queries: str) -> ResearchPlan:
    # Pad short test queries so they pass the min_length=4 validator.
    steps = [
        PlanStep(query=f"query: {q}", intent=f"learn about: {q}") for q in queries
    ]
    return ResearchPlan(steps=steps, rationale="stub plan")


@dataclass
class StubPlanLLM:
    """Scripted plans; the i-th call returns plans[i]."""

    plans: list[ResearchPlan]
    calls: int = 0
    seen_focus: list[list[str]] = field(default_factory=list)

    def invoke(self, messages):
        # Capture the follow-up questions surfaced to the plan node so a
        # test can assert reflect-to-plan handoff.
        for m in messages:
            content = getattr(m, "content", "")
            if "FOLLOW-UP QUESTIONS" in content:
                lines = [
                    line[4:].strip()
                    for line in content.splitlines()
                    if line.startswith("  - ")
                ]
                self.seen_focus.append(lines)
                break
        else:
            self.seen_focus.append([])
        idx = min(self.calls, len(self.plans) - 1)
        p = self.plans[idx]
        self.calls += 1
        return p


@dataclass
class _Result:
    answer: str


@dataclass
class StubResearchAgent:
    """Returns a different cited answer on each call. The agent_node calls
    this once per plan step, so a 2-step plan = 2 agent invocations."""

    answers: list[str] = field(default_factory=list)
    calls: int = 0

    def invoke(self, state: dict[str, Any]) -> _Result:
        idx = min(self.calls, len(self.answers) - 1) if self.answers else 0
        text = (
            self.answers[idx]
            if self.answers
            else f"stub answer #{self.calls + 1} [https://example.org/{self.calls}]"
        )
        self.calls += 1
        return _Result(answer=text)


@dataclass
class StubReflectLLM:
    """Scripted reflect verdicts; the i-th call returns verdicts[i]."""

    verdicts: list[ReflectVerdict]
    calls: int = 0

    def invoke(self, messages):  # noqa: ARG002
        idx = min(self.calls, len(self.verdicts) - 1)
        v = self.verdicts[idx]
        self.calls += 1
        return v


class StubArtifactLLM:
    """Deterministic markdown so we can assert the final draft was used."""

    def invoke(self, messages):  # noqa: ARG002
        class R:
            content = "# stub report\n\nbody [https://example.org/cited]"
        return R()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passing_verdict() -> ReflectVerdict:
    return ReflectVerdict(done=True, reasoning="enough evidence", missing_questions=[])


def _retry_verdict(*missing: str) -> ReflectVerdict:
    return ReflectVerdict(
        done=False, reasoning="more to learn", missing_questions=list(missing),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_workflow_finalizes_when_reflect_says_done(store):
    graph = build_workflow(
        plan_llm=StubPlanLLM(plans=[_plan("q1", "q2")]),
        research_agent=StubResearchAgent(
            answers=["a1 [https://x/1]", "a2 [https://x/2]"]
        ),
        reflect_llm=StubReflectLLM(verdicts=[_passing_verdict()]),
        artifact_llm=StubArtifactLLM(),
        store=store,
    )
    final = graph.invoke(
        {"topic": "How does CRAG work?", "max_iterations": 3},
        config={"recursion_limit": 30},
    )
    assert final["outcome"] == "complete"
    assert final["round"] == 1
    artifact: Artifact | None = store.get(final["artifact_id"])
    assert artifact is not None
    assert artifact.outcome == "complete"
    assert artifact.rounds == 1
    assert artifact.findings_count == 2
    assert len(artifact.provenance) == 1
    assert artifact.provenance[0].verdict.done is True


def test_workflow_caps_when_reflect_never_done(store):
    reflect = StubReflectLLM(
        verdicts=[_retry_verdict("missing X"), _retry_verdict("missing Y"), _retry_verdict("missing Z")]
    )
    graph = build_workflow(
        plan_llm=StubPlanLLM(
            plans=[_plan("q1"), _plan("q2"), _plan("q3")]
        ),
        research_agent=StubResearchAgent(
            answers=["a1 [https://x/1]", "a2 [https://x/2]", "a3 [https://x/3]"]
        ),
        reflect_llm=reflect,
        artifact_llm=StubArtifactLLM(),
        store=store,
    )
    final = graph.invoke(
        {"topic": "stubborn topic", "max_iterations": 3},
        config={"recursion_limit": 40},
    )
    assert final["outcome"] == "cap"
    assert final["round"] == 3
    assert reflect.calls == 3
    artifact = store.get(final["artifact_id"])
    assert artifact is not None
    assert artifact.outcome == "cap"
    assert [r.verdict.done for r in artifact.provenance] == [False, False, False]


def test_reflect_missing_questions_flow_into_next_plan(store):
    plan_stub = StubPlanLLM(plans=[_plan("q1"), _plan("q2")])
    graph = build_workflow(
        plan_llm=plan_stub,
        research_agent=StubResearchAgent(answers=["a1 [https://x/1]", "a2 [https://x/2]"]),
        reflect_llm=StubReflectLLM(
            verdicts=[_retry_verdict("what about edge case Z?"), _passing_verdict()]
        ),
        artifact_llm=StubArtifactLLM(),
        store=store,
    )
    final = graph.invoke(
        {"topic": "topic", "max_iterations": 3},
        config={"recursion_limit": 40},
    )
    assert final["outcome"] == "complete"
    assert plan_stub.calls == 2
    # First plan invocation: no follow-ups yet. Second: should receive the
    # missing_questions from reflect verdict #1.
    assert plan_stub.seen_focus[0] == []
    assert plan_stub.seen_focus[1] == ["what about edge case Z?"]


def test_agent_node_runs_once_per_plan_step(store):
    plan = _plan("q1", "q2", "q3")
    agent = StubResearchAgent(
        answers=["a1 [https://x/1]", "a2 [https://x/2]", "a3 [https://x/3]"]
    )
    graph = build_workflow(
        plan_llm=StubPlanLLM(plans=[plan]),
        research_agent=agent,
        reflect_llm=StubReflectLLM(verdicts=[_passing_verdict()]),
        artifact_llm=StubArtifactLLM(),
        store=store,
    )
    final = graph.invoke(
        {"topic": "topic", "max_iterations": 2},
        config={"recursion_limit": 30},
    )
    assert agent.calls == 3
    assert final["findings"] and len(final["findings"]) == 3


def test_sources_are_extracted_from_agent_answers(store):
    agent = StubResearchAgent(
        answers=[
            "First fact [https://a.example/1] second fact [https://b.example/2].",
            "Only one cite here [https://c.example/3].",
        ],
    )
    graph = build_workflow(
        plan_llm=StubPlanLLM(plans=[_plan("q1", "q2")]),
        research_agent=agent,
        reflect_llm=StubReflectLLM(verdicts=[_passing_verdict()]),
        artifact_llm=StubArtifactLLM(),
        store=store,
    )
    final = graph.invoke(
        {"topic": "topic", "max_iterations": 2},
        config={"recursion_limit": 30},
    )
    artifact = store.get(final["artifact_id"])
    findings = artifact.provenance[0].findings
    assert [s.url for s in findings[0].sources] == [
        "https://a.example/1",
        "https://b.example/2",
    ]
    assert [s.url for s in findings[1].sources] == ["https://c.example/3"]


def test_artifact_node_writes_final_draft(store):
    graph = build_workflow(
        plan_llm=StubPlanLLM(plans=[_plan("q1")]),
        research_agent=StubResearchAgent(answers=["a [https://x/1]"]),
        reflect_llm=StubReflectLLM(verdicts=[_passing_verdict()]),
        artifact_llm=StubArtifactLLM(),
        store=store,
    )
    final = graph.invoke(
        {"topic": "topic", "max_iterations": 2},
        config={"recursion_limit": 30},
    )
    assert final["final_draft"].startswith("# stub report")
    artifact = store.get(final["artifact_id"])
    assert artifact.final_draft == final["final_draft"]
