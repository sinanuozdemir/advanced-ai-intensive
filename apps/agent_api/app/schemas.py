"""Pydantic models shared between the HTTP layer, the workflow, and the store.

Graph: ``plan -> agent -> reflect -> [continue | finalize] -> artifact``.

* ``ResearchPlan`` is the structured output of the plan node: 2-5 focused
  sub-queries to investigate.
* ``Finding`` is what the agent node produces per plan step: a synthesized
  answer with the list of source URLs it cited.
* ``ReflectVerdict`` is the structured output of the reflect node: a
  done/not-done decision plus follow-up questions if we need another round.
* ``Artifact`` is what we persist and return to the caller.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request / response
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    """Body of ``POST /research``."""

    topic: str = Field(min_length=4, description="The research question or topic to brief.")
    max_iterations: int | None = Field(
        default=None, ge=1, le=10,
        description="Cap on plan -> agent -> reflect rounds. Defaults to settings.MAX_ITERATIONS.",
    )


class ResearchResponse(BaseModel):
    """Body of ``POST /research`` on success."""

    artifact_id: str
    request_id: str
    outcome: Literal["complete", "cap", "error"]
    rounds: int
    findings_count: int
    artifact: "Artifact"


# ---------------------------------------------------------------------------
# Workflow stages
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """One focused sub-query for the agent to chase."""

    query: str = Field(
        min_length=4,
        description="Self-contained search query — what you'd paste into Google.",
    )
    intent: str = Field(
        description="One sentence describing what we hope to learn from this step.",
    )


class ResearchPlan(BaseModel):
    """Structured output of the plan node.

    The agent fans this out into individual search + scrape passes, one
    per ``step``. Keep ``steps`` small — every step costs at least one
    LLM call and several HTTP fetches.
    """

    steps: list[PlanStep] = Field(
        min_length=1, max_length=5,
        description="2-5 sub-queries, ordered roughly by how foundational they are.",
    )
    rationale: str = Field(
        description="One paragraph explaining why this plan covers the topic.",
    )


class Source(BaseModel):
    """A URL the agent cited while writing a finding."""

    url: str = Field(description="The URL of the source.")
    title: str = Field(default="", description="Source page title, if known.")


class Finding(BaseModel):
    """One executed plan step: the agent's synthesized answer + its citations."""

    step: PlanStep
    summary: str = Field(
        description=(
            "2-4 sentences answering the step's question, with inline "
            "citations in [url] form."
        ),
    )
    sources: list[Source] = Field(default_factory=list)


class ReflectVerdict(BaseModel):
    """Structured output of the reflect node.

    Drives the conditional edge: if ``done`` we head to the artifact
    writer; otherwise the workflow loops back to ``plan`` with
    ``missing_questions`` appended as the next round's seed.
    """

    done: bool = Field(
        description=(
            "True when the accumulated findings answer the topic well "
            "enough that another search round wouldn't materially help."
        ),
    )
    reasoning: str = Field(
        description="One paragraph justifying the decision.",
    )
    missing_questions: list[str] = Field(
        default_factory=list,
        description=(
            "If not done, the specific follow-up questions the next "
            "plan round should pursue. Empty when done."
        ),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class RoundRecord(BaseModel):
    """One full plan -> agent -> reflect round, kept for provenance."""

    round: int = Field(ge=1)
    plan: ResearchPlan
    findings: list[Finding] = Field(default_factory=list)
    verdict: ReflectVerdict


class Artifact(BaseModel):
    """A persisted research artifact returned by GET /artifacts/{id}."""

    artifact_id: str
    topic: str
    final_draft: str
    rounds: int
    findings_count: int
    outcome: Literal["complete", "cap", "error"]
    provenance: list[RoundRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ArtifactSummary(BaseModel):
    """Lighter shape returned by GET /artifacts (no draft body)."""

    artifact_id: str
    topic: str
    rounds: int
    findings_count: int
    outcome: Literal["complete", "cap", "error"]
    created_at: datetime


class ArtifactListResponse(BaseModel):
    items: list[ArtifactSummary]
    total: int
    limit: int
    offset: int


ResearchResponse.model_rebuild()
