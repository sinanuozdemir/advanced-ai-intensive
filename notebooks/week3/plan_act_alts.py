"""The five plan/act policies under test in the W3 bake-off.

All five are self-contained — none of them import from ``forge.*``. The
bake-off is an upstream experiment whose results will inform which policy
Forge ships; Forge is downstream.

- ``AlwaysActSolo``          — zero-cost lower bound on safety.
- ``AlwaysPlanSuper``        — safety-heavy upper bound on overhead.
- ``ToolRiskHeuristic``      — non-LLM keyword router.
- ``TrajectoryProbe``        — one cheap structured LLM call decides both axes.
- ``PlanThenSelfCritique``   — deliberate twice (draft + critic).

Each satisfies the ``bakeoff_lib.PlanActPolicy`` protocol so it can be passed
directly to ``run_task_on_fixture(..., policy=...)``.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from shared import get_llm, get_structured_llm

from bakeoff_lib import BakeoffContext, Decision


# ---------------------------------------------------------------------------
# always_act_solo
# ---------------------------------------------------------------------------


class AlwaysActSolo:
    """Returns ``(act, solo)`` for every task. Zero-cost baseline."""

    name = "always_act_solo"

    async def decide(
        self, task: str, history: list[Any], ctx: BakeoffContext,
    ) -> Decision:
        return Decision(
            mode="act", topology="solo",
            reason="zero-cost baseline; always dispatch to solo",
        )


# ---------------------------------------------------------------------------
# always_plan_super
# ---------------------------------------------------------------------------


_PLANNER_SYSTEM = (
    "You are the planner. Produce a numbered, concrete plan that a coder "
    "agent can execute. The final step must describe how success is verified."
)


class AlwaysPlanSuper:
    """Always returns ``(plan, supervisor)``. Drafts a plan with the planner LLM."""

    name = "always_plan_super"

    def __init__(self, *, plan_model: str | None = None) -> None:
        self._plan_model = plan_model

    async def decide(
        self, task: str, history: list[Any], ctx: BakeoffContext,
    ) -> Decision:
        model_slug = self._plan_model or ctx.models.planner
        llm = get_llm(model_slug)
        prompt = (
            f"Draft a numbered plan for this task. 3-7 steps. The final step "
            f"must describe how success is verified.\n\nTask:\n{task}"
        )
        msg = await asyncio.to_thread(
            llm.invoke,
            [SystemMessage(content=_PLANNER_SYSTEM),
             HumanMessage(content=prompt)],
        )
        plan_md = msg.content if hasattr(msg, "content") else str(msg)
        return Decision(
            mode="plan", topology="supervisor",
            plan_md=str(plan_md),
            reason="always_plan_super: every task gets a drafted plan + supervisor",
        )


# ---------------------------------------------------------------------------
# tool_risk_heuristic
# ---------------------------------------------------------------------------


_WRITE_VERBS = (
    "write ", "create ", "edit ", "modify ", "delete ", "remove ", "rename ",
    "move ", "refactor ", "commit ", "push ", "reset ", "drop ", "install ",
    "upgrade ", "downgrade ",
)
_MULTIFILE_PATTERNS = (
    re.compile(r"\bacross\s+(?:all|every|the)\b", re.I),
    re.compile(r"\b(?:every|all)\b.*\bfile", re.I),
    re.compile(r"\bmulti(?:-|\s)?file\b", re.I),
)
_WRITE_TOOLS = (
    "fs_write", "fs_edit", "fs_delete",
)


class ToolRiskHeuristic:
    """Non-LLM router. ``(plan, supervisor)`` if the task mentions a write-class
    verb / pattern / tool name; else ``(act, solo)``."""

    name = "tool_risk_heuristic"

    def __init__(
        self,
        *,
        write_verbs: list[str] | None = None,
        write_tools: list[str] | None = None,
    ) -> None:
        self.write_verbs = list(write_verbs or _WRITE_VERBS)
        self.write_tools = list(write_tools or _WRITE_TOOLS)

    async def decide(
        self, task: str, history: list[Any], ctx: BakeoffContext,
    ) -> Decision:
        lower = task.lower()
        write = any(v in lower for v in self.write_verbs)
        multi = any(p.search(lower) for p in _MULTIFILE_PATTERNS)
        explicit = any(t.split("_", 1)[-1] in lower for t in self.write_tools)
        risky = write or multi or explicit
        if risky:
            return Decision(
                mode="plan", topology="supervisor",
                reason=(
                    "tool_risk_heuristic: task looks write-class "
                    f"(write={write}, multifile={multi}, explicit={explicit})"
                ),
            )
        return Decision(
            mode="act", topology="solo",
            reason="tool_risk_heuristic: read-only / single-file task",
        )


# ---------------------------------------------------------------------------
# trajectory_probe
# ---------------------------------------------------------------------------


class _Probe(BaseModel):
    mode: str = Field(description="'plan' or 'act'.")
    topology: str = Field(description="'solo' or 'supervisor'.")
    reason: str = Field(description="One sentence. Why this combo for this task?")


_PROBE_SYS = """\
You are a routing classifier. Given a user message or task, decide:

- mode = 'plan' if the task is multi-step, irreversible, or touches >1 file;
  'act' if it's conversational, a single-file read, or a trivial 1-line edit.
- topology = 'supervisor' if the task is complex and benefits from planner /
  coder / critic separation; 'solo' for simple direct execution.

CRITICAL — second-order effects bias toward plan+supervisor.
Tasks that LOOK simple but routinely have hidden dependencies, even when the
user only mentions one file:

- "rename X to Y" — callers, tests, and docstrings reference X
- "delete <file>" / "remove <function>" — other files may import it
- "move X to Y/" — every import statement for X breaks
- "change the signature of X" — callers pass the old args
- "swap library A for library B" — every import needs updating

When the task involves rename / delete / remove / move / replace / migrate
across ANY code, treat it as multi-step + irreversible by default:
mode='plan', topology='supervisor'. The cost of an unnecessary plan is one
LLM call; the cost of acting on a hidden-dependency task is broken code.

Be terse. Output structured JSON.
"""


class TrajectoryProbe:
    """One cheap structured LLM call returns both ``mode`` and ``topology``.

    Inputs: task + last 3 turns + top-3 episodic recall headlines.
    """

    name = "trajectory_probe"

    def __init__(self, *, model: str | None = None) -> None:
        self._model = model

    async def decide(
        self, task: str, history: list[Any], ctx: BakeoffContext,
    ) -> Decision:
        slug = self._model or ctx.models.trajectory_probe
        llm = get_structured_llm(slug, _Probe)
        last_three = _last_three(history)
        recall = "\n".join(f"- {h}" for h in ctx.episodic_headlines[:3]) or "(none)"
        user = (
            f"Task:\n{task}\n\n"
            f"Last 3 turns of history (may be empty):\n{last_three}\n\n"
            f"Top episodic recall headlines:\n{recall}"
        )
        out = await asyncio.to_thread(
            llm.invoke,
            [SystemMessage(content=_PROBE_SYS), HumanMessage(content=user)],
        )
        mode = "plan" if str(out.mode).lower() == "plan" else "act"
        topology = "supervisor" if str(out.topology).lower() == "supervisor" else "solo"
        return Decision(
            mode=mode, topology=topology,
            reason=f"trajectory_probe: {out.reason}",
        )


def _last_three(history: list[Any]) -> str:
    if not history:
        return "(none)"
    turns: list[str] = []
    for m in history[-3:]:
        role = (
            getattr(m, "type", None)
            or m.__class__.__name__.replace("Message", "").lower()
        )
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        turns.append(f"{role.upper()}: {content[:200]}")
    return "\n".join(turns)


# ---------------------------------------------------------------------------
# plan_then_self_critique
# ---------------------------------------------------------------------------


_CRITIC_SYSTEM = (
    "You are reviewing a PLAN, not finished work. Score the plan's quality "
    "0-5 based on: (a) covers all the user's requirements, (b) steps are "
    "concrete and not too coarse, (c) the final step describes verification."
)


class _PlanCritique(BaseModel):
    score: int = Field(description="Plan quality 0-5. 0=useless, 5=excellent.")
    fixes: list[str] = Field(
        default_factory=list,
        description="Specific concrete fixes if score < 5.",
    )


class PlanThenSelfCritique:
    """Drafts a plan -> critic scores it 0-5 -> if score < threshold, revise once.
    Always dispatches to supervisor."""

    name = "plan_then_self_critique"

    def __init__(
        self,
        *,
        plan_model: str | None = None,
        critic_model: str | None = None,
        threshold: float = 3.5,
    ) -> None:
        self._plan_model = plan_model
        self._critic_model = critic_model
        self._threshold = threshold

    async def decide(
        self, task: str, history: list[Any], ctx: BakeoffContext,
    ) -> Decision:
        plan_slug = self._plan_model or ctx.models.planner
        critic_slug = self._critic_model or ctx.models.critic
        threshold = self._threshold
        plan = await self._draft_plan(plan_slug, task)
        critique = await self._critique_plan(critic_slug, task, plan)
        revised = False
        if critique.score < threshold:
            plan = await self._revise_plan(plan_slug, task, plan, critique.fixes)
            revised = True
        return Decision(
            mode="plan", topology="supervisor",
            plan_md=plan,
            reason=(
                f"plan_then_self_critique: score={critique.score}/5 "
                f"{'-> revised' if revised else 'accepted'}"
            ),
            meta={"score": critique.score, "revised": revised},
        )

    async def _draft_plan(self, slug: str, task: str) -> str:
        llm = get_llm(slug)
        out = await asyncio.to_thread(
            llm.invoke,
            [SystemMessage(content=_PLANNER_SYSTEM),
             HumanMessage(content=f"Task:\n{task}")],
        )
        return out.content if hasattr(out, "content") else str(out)

    async def _critique_plan(
        self, slug: str, task: str, plan_md: str,
    ) -> _PlanCritique:
        llm = get_structured_llm(slug, _PlanCritique)
        return await asyncio.to_thread(
            llm.invoke,
            [SystemMessage(content=_CRITIC_SYSTEM),
             HumanMessage(content=f"Task:\n{task}\n\nPlan:\n{plan_md}")],
        )

    async def _revise_plan(
        self, slug: str, task: str, plan_md: str, fixes: list[str],
    ) -> str:
        llm = get_llm(slug)
        fixes_str = "\n".join(f"- {f}" for f in fixes) or "(none specific)"
        out = await asyncio.to_thread(
            llm.invoke,
            [SystemMessage(content=_PLANNER_SYSTEM),
             HumanMessage(content=(
                 f"Revise the plan below based on the critic feedback.\n\n"
                 f"Task:\n{task}\n\nCurrent plan:\n{plan_md}\n\n"
                 f"Critic fixes:\n{fixes_str}"
             ))],
        )
        return out.content if hasattr(out, "content") else str(out)


__all__ = [
    "AlwaysActSolo",
    "AlwaysPlanSuper",
    "ToolRiskHeuristic",
    "TrajectoryProbe",
    "PlanThenSelfCritique",
]
