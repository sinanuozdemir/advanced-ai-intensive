"""The four workflow stages and the loop gate.

Graph::

    START -> plan -> agent -> reflect -> [continue -> plan | finalize -> artifact -> END]

Each node is a plain ``Callable[[WorkflowState], WorkflowState]`` so it's
easy to unit-test in isolation. The LangGraph wiring in ``workflow.py`` is
the only place that knows about ``StateGraph``.

The nodes are factories — they take the LLM/agent dependencies as
arguments and return the actual node functions. Tests can swap in stubs
without monkeypatching imports.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from .logging_setup import get_logger
from .metrics import (
    observe_node,
    reflect_rounds_total,
    workflow_runs_total,
)
from .schemas import (
    Artifact,
    Finding,
    PlanStep,
    ReflectVerdict,
    ResearchPlan,
    RoundRecord,
    Source,
)
from .store import ArtifactStore


log = get_logger("agent_api.workflow")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class WorkflowState(TypedDict, total=False):
    """LangGraph state, threaded through every node.

    Pydantic models are stored as dumped dicts so LangGraph's checkpointer
    can JSON-serialize them; helpers below convert back at the boundaries.
    """

    topic: str
    request_id: str
    max_iterations: int

    round: int
    plans: list[dict]                  # list[ResearchPlan.model_dump()]
    findings: list[dict]               # accumulated list[Finding.model_dump()]
    round_records: list[dict]          # list[RoundRecord.model_dump()]
    last_verdict: dict | None          # ReflectVerdict.model_dump()
    next_focus: list[str]              # follow-up questions for the next plan round

    final_draft: str
    outcome: str                       # "complete" | "cap" | "error"
    artifact_id: str | None
    error: str | None


# ---------------------------------------------------------------------------
# Plan node — structured LLM emits a list of focused sub-queries
# ---------------------------------------------------------------------------


_PLAN_SYSTEM = (
    "You are a research planner. Given a topic and (optionally) a list of "
    "follow-up questions left over from a previous round, decompose the "
    "investigation into 2-5 focused, *searchable* sub-queries.\n\n"
    "Good sub-queries are self-contained search strings (the kind you'd "
    "paste into Google), not abstract themes. Each step's intent should "
    "be one short sentence stating what we expect to learn.\n\n"
    "If follow-up questions are provided, prioritise them — they're the "
    "specific gaps a previous round flagged."
)


def make_plan_node(plan_llm: Any) -> Callable[[WorkflowState], WorkflowState]:
    """``plan_llm`` is a ``get_structured_llm(ResearchPlan)`` wrapper."""

    def plan_node(state: WorkflowState) -> WorkflowState:
        with observe_node("plan"):
            round_no = int(state.get("round", 0)) + 1
            focus = state.get("next_focus") or []
            log.info(
                "node.start",
                extra={"node": "plan", "round": round_no, "n_focus": len(focus)},
            )
            human_parts = [f"TOPIC: {state['topic']}"]
            if focus:
                human_parts.append("\nFOLLOW-UP QUESTIONS from the previous round:")
                for q in focus:
                    human_parts.append(f"  - {q}")
            plan: ResearchPlan = plan_llm.invoke([
                SystemMessage(content=_PLAN_SYSTEM),
                HumanMessage(content="\n".join(human_parts)),
            ])
            plans = list(state.get("plans", [])) + [plan.model_dump()]
            log.info(
                "node.end",
                extra={
                    "node": "plan",
                    "round": round_no,
                    "n_steps": len(plan.steps),
                },
            )
            return {"plans": plans, "round": round_no}

    return plan_node


# ---------------------------------------------------------------------------
# Agent node — executes each plan step with real web tools
# ---------------------------------------------------------------------------


_AGENT_TASK = (
    "Answer the following research question using the tools you have. "
    "Use `serpapi_search` to find candidate sources, then `firecrawl_scrape` "
    "to read the most promising ones BEFORE quoting them. Never cite a URL "
    "you didn't successfully fetch.\n\n"
    "Return 2-4 sentences answering the question, with inline citations in "
    "the form [url] after every non-trivial claim. Do not include any "
    "preamble or section headers — just the cited answer.\n\n"
    "QUESTION: {query}\n"
    "WHAT WE WANT TO LEARN: {intent}\n\n"
    "If your searches return nothing useful, say so plainly in one short "
    "sentence — do not fabricate citations."
)


_URL_RE = re.compile(r"\[(https?://[^\]\s]+)\]")


def _extract_sources(answer: str) -> list[Source]:
    seen: set[str] = set()
    out: list[Source] = []
    for url in _URL_RE.findall(answer):
        url = url.rstrip(".,;)")
        if url in seen:
            continue
        seen.add(url)
        out.append(Source(url=url))
    return out


def make_agent_node(research_agent: Any) -> Callable[[WorkflowState], WorkflowState]:
    """``research_agent`` has ``.invoke({'task': str}) -> obj.answer: str``.

    The agent is expected to be a tool-using solo agent with the
    ``serpapi_search`` + ``firecrawl_scrape`` tools loaded.
    """

    def agent_node(state: WorkflowState) -> WorkflowState:
        with observe_node("agent"):
            plans = state.get("plans") or []
            plan = ResearchPlan.model_validate(plans[-1])
            round_no = int(state.get("round", 1))
            log.info(
                "node.start",
                extra={"node": "agent", "round": round_no, "n_steps": len(plan.steps)},
            )
            new_findings: list[Finding] = []
            for step in plan.steps:
                task = _AGENT_TASK.format(query=step.query, intent=step.intent)
                result = research_agent.invoke({"task": task})
                answer = (getattr(result, "answer", None) or str(result)).strip()
                sources = _extract_sources(answer)
                new_findings.append(
                    Finding(step=step, summary=answer, sources=sources)
                )
                log.info(
                    "agent.step",
                    extra={
                        "round": round_no,
                        "query_chars": len(step.query),
                        "answer_chars": len(answer),
                        "n_sources": len(sources),
                    },
                )
            findings = list(state.get("findings", [])) + [
                f.model_dump() for f in new_findings
            ]
            log.info(
                "node.end",
                extra={
                    "node": "agent",
                    "round": round_no,
                    "new_findings": len(new_findings),
                    "total_findings": len(findings),
                },
            )
            return {"findings": findings}

    return agent_node


# ---------------------------------------------------------------------------
# Reflect node — decide whether we've answered the topic
# ---------------------------------------------------------------------------


_REFLECT_SYSTEM = (
    "You are a research auditor. Given a topic and the findings accumulated "
    "across one or more research rounds, decide whether the topic has been "
    "adequately covered.\n\n"
    "RULES\n"
    "1. Be willing to say ``done=true`` if the findings genuinely cover the "
    "   topic. Don't pad with extra rounds for the sake of it — every "
    "   round costs the user latency and money.\n"
    "2. Set ``done=false`` only when there's a *specific* gap. Spell out "
    "   the missing questions in ``missing_questions`` so the next plan "
    "   round has a concrete target.\n"
    "3. If most queries returned ``no sources``, more searching is unlikely "
    "   to help — say ``done=true`` and let the artifact writer note the "
    "   evidence shortage honestly rather than loop forever.\n"
)


def make_reflect_node(reflect_llm: Any) -> Callable[[WorkflowState], WorkflowState]:
    """``reflect_llm`` is a ``get_structured_llm(ReflectVerdict)`` wrapper."""

    def reflect_node(state: WorkflowState) -> WorkflowState:
        with observe_node("reflect"):
            round_no = int(state.get("round", 1))
            findings = state.get("findings") or []
            plans = state.get("plans") or []
            log.info(
                "node.start",
                extra={
                    "node": "reflect",
                    "round": round_no,
                    "total_findings": len(findings),
                },
            )
            findings_text = "\n\n".join(
                f"Q: {f['step']['query']}\nIntent: {f['step']['intent']}\n"
                f"Answer: {f['summary']}\n"
                f"Sources: {[s['url'] for s in f.get('sources', [])]}"
                for f in findings
            ) or "(no findings yet)"
            human = (
                f"TOPIC: {state['topic']}\n\n"
                f"ROUNDS COMPLETED: {round_no}\n\n"
                f"FINDINGS SO FAR:\n{findings_text}"
            )
            verdict: ReflectVerdict = reflect_llm.invoke([
                SystemMessage(content=_REFLECT_SYSTEM),
                HumanMessage(content=human),
            ])
            reflect_rounds_total.inc()

            current_plan = ResearchPlan.model_validate(plans[-1])
            new_findings_count = len(current_plan.steps)
            round_record = RoundRecord(
                round=round_no,
                plan=current_plan,
                findings=[
                    Finding.model_validate(f)
                    for f in findings[-new_findings_count:]
                ],
                verdict=verdict,
            )
            round_records = list(state.get("round_records", [])) + [
                round_record.model_dump()
            ]
            log.info(
                "node.end",
                extra={
                    "node": "reflect",
                    "round": round_no,
                    "done": verdict.done,
                    "n_missing": len(verdict.missing_questions),
                },
            )
            return {
                "last_verdict": verdict.model_dump(),
                "round_records": round_records,
                "next_focus": list(verdict.missing_questions),
            }

    return reflect_node


# ---------------------------------------------------------------------------
# Loop gate — conditional edge after reflect
# ---------------------------------------------------------------------------


def loop_gate(state: WorkflowState) -> str:
    """Return either ``"continue"`` or ``"finalize"``.

    Finalize when the reflector says done OR when we've hit the
    ``max_iterations`` ceiling. Continue otherwise.
    """
    verdict = state.get("last_verdict") or {}
    done = bool(verdict.get("done", False))
    round_no = int(state.get("round", 0))
    max_iters = int(state.get("max_iterations", 3))
    log.info(
        "loop_gate",
        extra={"done": done, "round": round_no, "max_iterations": max_iters},
    )
    if done or round_no >= max_iters:
        return "finalize"
    return "continue"


# ---------------------------------------------------------------------------
# Artifact node — write the final cited report and persist
# ---------------------------------------------------------------------------


_ARTIFACT_SYSTEM = (
    "You are a research writer. You will be given a topic and a list of "
    "findings (each is a sub-question with a short cited answer). Write "
    "the final research report.\n\n"
    "FORMAT\n"
    "- Start with an H1 title that reflects the topic.\n"
    "- Use H2 section headers that group related findings naturally.\n"
    "- Every non-trivial factual claim MUST end with an inline citation in "
    "  the form [url], using only URLs that appear in the supplied findings.\n"
    "- Do NOT invent citations. Do NOT cite URLs the findings don't list.\n"
    "- If the findings are thin or contradictory, say so honestly in a "
    "  short closing 'Caveats' section rather than padding with filler.\n"
    "- Aim for tight, scannable prose. No more than ~500 words total.\n"
)


def make_artifact_node(
    artifact_llm: Any, store: ArtifactStore,
) -> Callable[[WorkflowState], WorkflowState]:
    """``artifact_llm`` is a plain ``get_llm(...)``-style chat model (not
    structured) — it returns a markdown string in ``.content``."""

    def artifact_node(state: WorkflowState) -> WorkflowState:
        with observe_node("artifact"):
            findings = state.get("findings") or []
            round_no = int(state.get("round", 0))
            max_iters = int(state.get("max_iterations", 3))
            verdict = state.get("last_verdict") or {}
            done = bool(verdict.get("done", False))

            if done:
                outcome = "complete"
            elif round_no >= max_iters:
                outcome = "cap"
            else:
                outcome = "error"

            findings_block = "\n\n".join(
                f"### Q{i + 1}: {f['step']['query']}\n"
                f"Intent: {f['step']['intent']}\n"
                f"Answer: {f['summary']}\n"
                f"Sources: {[s['url'] for s in f.get('sources', [])]}"
                for i, f in enumerate(findings)
            ) or "(no findings collected — note this honestly in the report)"

            log.info(
                "node.start",
                extra={
                    "node": "artifact",
                    "outcome": outcome,
                    "n_findings": len(findings),
                },
            )
            messages = [
                SystemMessage(content=_ARTIFACT_SYSTEM),
                HumanMessage(content=(
                    f"TOPIC: {state['topic']}\n\n"
                    f"FINDINGS:\n{findings_block}"
                )),
            ]
            result = artifact_llm.invoke(messages)
            final_draft = (
                getattr(result, "content", None)
                or getattr(result, "answer", None)
                or str(result)
            ).strip()

            round_records_raw = state.get("round_records") or []
            round_records = [
                RoundRecord.model_validate(r) for r in round_records_raw
            ]
            artifact = Artifact(
                artifact_id=str(uuid.uuid4()),
                topic=state["topic"],
                final_draft=final_draft,
                rounds=round_no,
                findings_count=len(findings),
                outcome=outcome,
                provenance=round_records,
                created_at=datetime.now(timezone.utc),
            )
            store.put(artifact)
            workflow_runs_total.labels(outcome=outcome).inc()
            log.info(
                "node.end",
                extra={
                    "node": "artifact",
                    "artifact_id": artifact.artifact_id,
                    "outcome": outcome,
                    "rounds": round_no,
                    "findings_count": len(findings),
                    "draft_chars": len(final_draft),
                },
            )
            return {
                "outcome": outcome,
                "artifact_id": artifact.artifact_id,
                "final_draft": final_draft,
            }

    return artifact_node
