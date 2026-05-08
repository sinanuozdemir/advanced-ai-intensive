"""Three multi-agent topologies + a flat baseline.

Every factory returns a `Topology` — a thin wrapper around a callable so
notebooks can do:

    topo = build_supervisor(workers, model_slug="anthropic/claude-opus-4.7")
    result = topo.invoke({"task": "..."})
    # result == {"answer": "...", "trajectory": [...], "tokens": {...}}

The three topologies share the same return shape so the eval harness can
compare them apples-to-apples.

| Topology      | Coordination          | LLM calls per task                   |
|---------------|-----------------------|--------------------------------------|
| solo          | none                  | 1 plan + N tool loops in one model   |
| supervisor    | central LLM router    | 1 router + 1 worker per delegation   |
| hierarchical  | 2-level routers       | 1 top + 1 mid + 1 worker per delegation |
| peer          | round-robin + vote    | N parallel workers + 1 aggregator    |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from shared import estimate_cost, get_llm, get_structured_llm
from .workers import WorkerSpec


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@dataclass
class TopologyResult:
    answer: str
    trajectory: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    n_worker_calls: int = 0
    n_tool_calls: int = 0


@dataclass
class Topology:
    name: Literal["solo", "supervisor", "hierarchical", "peer"]
    invoke: Callable[[dict], TopologyResult]


# ---------------------------------------------------------------------------
# Helpers shared across topologies
# ---------------------------------------------------------------------------


def _agent_from_spec(spec: WorkerSpec, *, model_slug: str | None = None):
    """Materialise a WorkerSpec into a LangChain react agent."""
    return create_agent(
        model=get_llm(model_slug or spec.model_slug),
        tools=spec.tools,
        system_prompt=spec.system_prompt,
    )


def _count_usage(messages, *, model_slug: str) -> tuple[int, int, float, int]:
    """Sum (input, output, cost, n_tool_calls) across an agent transcript."""
    in_t = out_t = n_tool = 0
    for m in messages:
        um = getattr(m, "usage_metadata", None) or {}
        in_t += int(um.get("input_tokens", 0) or 0)
        out_t += int(um.get("output_tokens", 0) or 0)
        if getattr(m, "tool_calls", None):
            n_tool += len(m.tool_calls)
    cost = estimate_cost(model_slug, in_t, out_t)
    return in_t, out_t, cost, n_tool


# ---------------------------------------------------------------------------
# 1. Flat / solo baseline
# ---------------------------------------------------------------------------


def build_solo(workers: list[WorkerSpec], *, model_slug: str = "openai/gpt-5.4-nano") -> Topology:
    """One agent, all the workers' tools merged. The control baseline."""

    all_tools = [t for w in workers for t in w.tools]
    sys_prompt = (
        "You are a single, capable assistant with multiple tools. "
        "Pick the right tool for each step of the task and answer the user's "
        "question. Cite sources where relevant."
    )
    agent = create_agent(
        model=get_llm(model_slug),
        tools=all_tools,
        system_prompt=sys_prompt,
    )

    def invoke(state: dict) -> TopologyResult:
        task = state["task"]
        out = agent.invoke({"messages": [HumanMessage(content=task)]})
        msgs = out["messages"]
        final = msgs[-1].content if msgs else ""
        in_t, out_t, cost, n_tool = _count_usage(msgs, model_slug=model_slug)
        # Capture the assistant text in the trajectory under "result" so the
        # downstream rubric can grade faithfulness against real evidence.
        # (Without this, solo's trajectory was just a message count and
        # faithfulness scored 0 across the board.)
        return TopologyResult(
            answer=final,
            trajectory=[{"role": "solo", "messages": len(msgs), "result": final}],
            input_tokens=in_t,
            output_tokens=out_t,
            cost_usd=cost,
            n_worker_calls=0,
            n_tool_calls=n_tool,
        )

    return Topology(name="solo", invoke=invoke)


# ---------------------------------------------------------------------------
# 2. Supervisor: LLM router delegates to specialists
# ---------------------------------------------------------------------------


class _RouteDecision(BaseModel):
    worker: str = Field(description="Which worker should handle this sub-task next? "
                                    "Use 'DONE' when no more work is needed.")
    sub_task: str = Field(description="The exact instruction to give that worker.")
    reasoning: str = Field(description="Why this worker, in one sentence.")


def build_supervisor(
    workers: list[WorkerSpec],
    *,
    supervisor_model: str = "openai/gpt-5.4-nano",
    max_steps: int = 4,
) -> Topology:
    """Central LLM picks one worker per step; loops until DONE or `max_steps`."""

    worker_agents = {w.name: _agent_from_spec(w) for w in workers}
    worker_menu = "\n".join(f"- {w.name}: {w.description}" for w in workers)

    router_llm = get_structured_llm(supervisor_model, _RouteDecision)
    router_sys = (
        "You are a supervisor coordinating a small team of specialist workers.\n\n"
        f"Available workers:\n{worker_menu}\n\n"
        "Given the user's task and the work already done, decide which worker "
        "should act next or return 'DONE'. Keep sub-tasks small and self-contained."
    )

    def invoke(state: dict) -> TopologyResult:
        task = state["task"]
        history: list[dict] = []
        in_t = out_t = 0
        cost = 0.0
        n_tool = 0
        n_worker = 0

        for step in range(max_steps):
            # 1. Router decides next move
            history_str = "\n".join(
                f"[{h['worker']}] {h['result'][:300]}" for h in history
            ) or "(none yet)"
            route_prompt = (
                f"User task: {task}\n\nWork so far:\n{history_str}\n\n"
                "Pick the next worker (or DONE) and the exact sub-task."
            )
            decision = router_llm.invoke([
                SystemMessage(content=router_sys),
                HumanMessage(content=route_prompt),
            ])
            # Crude but works: estimate router cost from prompt+answer length
            est_in = (len(router_sys) + len(route_prompt)) // 4
            est_out = len(decision.model_dump_json()) // 4
            in_t += est_in
            out_t += est_out
            cost += estimate_cost(supervisor_model, est_in, est_out)

            if decision.worker.upper() == "DONE":
                break
            agent = worker_agents.get(decision.worker)
            if agent is None:
                history.append({"worker": "supervisor",
                                "result": f"Unknown worker '{decision.worker}'; ending."})
                break

            # 2. Worker executes
            n_worker += 1
            spec = next(w for w in workers if w.name == decision.worker)
            out = agent.invoke({"messages": [HumanMessage(content=decision.sub_task)]})
            msgs = out["messages"]
            answer = msgs[-1].content if msgs else ""
            wi_t, wo_t, wcost, wn_tool = _count_usage(msgs, model_slug=spec.model_slug)
            in_t += wi_t
            out_t += wo_t
            cost += wcost
            n_tool += wn_tool
            history.append({
                "worker": decision.worker,
                "sub_task": decision.sub_task,
                "result": answer,
            })

        # 3. Final synthesis from supervisor (re-uses router LLM, free-form output)
        final_llm = get_llm(supervisor_model)
        full = "\n\n".join(f"[{h['worker']}] {h['result']}" for h in history)
        synth = final_llm.invoke([
            SystemMessage(content="You are a supervisor. Combine your team's "
                                  "findings into a final answer for the user."),
            HumanMessage(content=f"User task: {task}\n\nTeam findings:\n{full}"),
        ])
        if hasattr(synth, "usage_metadata") and synth.usage_metadata:
            in_t += synth.usage_metadata.get("input_tokens", 0)
            out_t += synth.usage_metadata.get("output_tokens", 0)
            cost += estimate_cost(
                supervisor_model,
                synth.usage_metadata.get("input_tokens", 0),
                synth.usage_metadata.get("output_tokens", 0),
            )

        return TopologyResult(
            answer=synth.content,
            trajectory=history,
            input_tokens=in_t,
            output_tokens=out_t,
            cost_usd=cost,
            n_worker_calls=n_worker,
            n_tool_calls=n_tool,
        )

    return Topology(name="supervisor", invoke=invoke)


# ---------------------------------------------------------------------------
# 3. Hierarchical: supervisor of supervisors
# ---------------------------------------------------------------------------


def build_hierarchical(
    teams: dict[str, list[WorkerSpec]],
    *,
    top_model: str = "openai/gpt-5.4-nano",
    sub_model: str = "openai/gpt-5.4-nano",
    max_steps: int = 3,
) -> Topology:
    """Two-layer routing: top supervisor picks a TEAM, team supervisor picks a WORKER.

    Useful when you have many specialists organised by domain (e.g. retrieval team
    vs analysis team vs writing team). For Segment 1 we ship a 2-team / 3-worker
    example so the cost and latency of the extra hop is visible in the data.
    """

    sub_topologies = {
        name: build_supervisor(workers, supervisor_model=sub_model, max_steps=max_steps)
        for name, workers in teams.items()
    }

    class _TopRoute(BaseModel):
        team: str = Field(description="Which team handles next, or 'DONE'.")
        sub_task: str = Field(description="What that team should do.")

    top_router = get_structured_llm(top_model, _TopRoute)
    team_menu = "\n".join(
        f"- {name}: {[w.name for w in ws]}" for name, ws in teams.items()
    )
    top_sys = (
        "You are a director coordinating multiple specialist TEAMS. Each team "
        "has its own supervisor. Pick which team should act next or return 'DONE'.\n\n"
        f"Teams:\n{team_menu}"
    )

    def invoke(state: dict) -> TopologyResult:
        task = state["task"]
        history: list[dict] = []
        in_t = out_t = 0
        cost = 0.0
        n_tool = 0
        n_worker = 0

        for step in range(max_steps):
            history_str = "\n".join(
                f"[{h['team']}] {h['result'][:300]}" for h in history
            ) or "(none yet)"
            decision = top_router.invoke([
                SystemMessage(content=top_sys),
                HumanMessage(content=f"User task: {task}\n\nWork so far:\n{history_str}\n\n"
                                     "Pick a team or DONE."),
            ])
            est_in = (len(top_sys) + len(history_str) + len(task)) // 4
            est_out = len(decision.model_dump_json()) // 4
            in_t += est_in
            out_t += est_out
            cost += estimate_cost(top_model, est_in, est_out)

            if decision.team.upper() == "DONE":
                break
            sub = sub_topologies.get(decision.team)
            if sub is None:
                break
            sub_result = sub.invoke({"task": decision.sub_task})
            in_t += sub_result.input_tokens
            out_t += sub_result.output_tokens
            cost += sub_result.cost_usd
            n_tool += sub_result.n_tool_calls
            n_worker += sub_result.n_worker_calls + 1  # +1 for the team supervisor itself
            history.append({"team": decision.team, "sub_task": decision.sub_task,
                            "result": sub_result.answer})

        final_llm = get_llm(top_model)
        full = "\n\n".join(f"[{h['team']}] {h['result']}" for h in history)
        synth = final_llm.invoke([
            SystemMessage(content="You are a director. Combine each team's report "
                                  "into a single answer."),
            HumanMessage(content=f"User task: {task}\n\nTeam reports:\n{full}"),
        ])
        return TopologyResult(
            answer=synth.content,
            trajectory=history,
            input_tokens=in_t,
            output_tokens=out_t,
            cost_usd=cost,
            n_worker_calls=n_worker,
            n_tool_calls=n_tool,
        )

    return Topology(name="hierarchical", invoke=invoke)


# ---------------------------------------------------------------------------
# 4. Peer: every worker sees the task, an aggregator picks/synthesises
# ---------------------------------------------------------------------------


def build_peer(
    workers: list[WorkerSpec],
    *,
    aggregator_model: str = "openai/gpt-5.4-nano",
) -> Topology:
    """All workers run in parallel on the same task; aggregator combines.

    Useful when you don't know which specialist is right. Cost = sum of all
    workers + aggregator. Latency = max(worker latency) + aggregator latency.
    """

    worker_agents = [(w, _agent_from_spec(w)) for w in workers]

    def invoke(state: dict) -> TopologyResult:
        task = state["task"]
        in_t = out_t = 0
        cost = 0.0
        n_tool = 0
        per_worker = []

        for spec, agent in worker_agents:
            try:
                out = agent.invoke({"messages": [HumanMessage(content=task)]})
                msgs = out["messages"]
                ans = msgs[-1].content if msgs else ""
                wi_t, wo_t, wcost, wn_tool = _count_usage(msgs, model_slug=spec.model_slug)
                in_t += wi_t
                out_t += wo_t
                cost += wcost
                n_tool += wn_tool
                per_worker.append({"worker": spec.name, "answer": ans})
            except Exception as exc:  # noqa: BLE001
                per_worker.append({"worker": spec.name, "answer": f"ERROR: {exc}"})

        # Aggregator
        agg_llm = get_llm(aggregator_model)
        peer_dump = "\n\n".join(f"[{p['worker']}] {p['answer']}" for p in per_worker)
        synth = agg_llm.invoke([
            SystemMessage(content="You are an aggregator. Each peer agent below "
                                  "answered the same task independently. Pick "
                                  "the best answer or merge them, citing which peers agreed."),
            HumanMessage(content=f"User task: {task}\n\nPeer answers:\n{peer_dump}"),
        ])
        if hasattr(synth, "usage_metadata") and synth.usage_metadata:
            in_t += synth.usage_metadata.get("input_tokens", 0)
            out_t += synth.usage_metadata.get("output_tokens", 0)
            cost += estimate_cost(
                aggregator_model,
                synth.usage_metadata.get("input_tokens", 0),
                synth.usage_metadata.get("output_tokens", 0),
            )

        return TopologyResult(
            answer=synth.content,
            trajectory=per_worker,
            input_tokens=in_t,
            output_tokens=out_t,
            cost_usd=cost,
            n_worker_calls=len(per_worker),
            n_tool_calls=n_tool,
        )

    return Topology(name="peer", invoke=invoke)
