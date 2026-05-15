"""LangGraph wiring for the deep-research workflow.

Graph shape::

    START -> plan -> agent -> reflect -> [continue -> plan | finalize -> artifact -> END]

The recursion limit is scaled with ``max_iterations`` because each round
fires four node transitions (plan, agent, reflect, gate).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, START, END

from .nodes import (
    WorkflowState,
    loop_gate,
    make_agent_node,
    make_artifact_node,
    make_plan_node,
    make_reflect_node,
)
from .store import ArtifactStore


def build_workflow(
    *,
    plan_llm: Any,
    research_agent: Any,
    reflect_llm: Any,
    artifact_llm: Any,
    store: ArtifactStore,
):
    """Compile and return the LangGraph.

    Dependencies are passed in explicitly so tests can wire stubs without
    touching the network or the LLM SDK.
    """

    graph = StateGraph(WorkflowState)
    graph.add_node("plan", make_plan_node(plan_llm))
    graph.add_node("agent", make_agent_node(research_agent))
    graph.add_node("reflect", make_reflect_node(reflect_llm))
    graph.add_node("artifact", make_artifact_node(artifact_llm, store))

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "agent")
    graph.add_edge("agent", "reflect")
    graph.add_conditional_edges(
        "reflect",
        loop_gate,
        {"continue": "plan", "finalize": "artifact"},
    )
    graph.add_edge("artifact", END)
    return graph.compile()
