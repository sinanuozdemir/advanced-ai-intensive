"""Build the runtime LLM + agent dependencies the workflow needs.

Kept in its own module so the FastAPI lifespan can construct everything
once at startup, and tests can construct stubs without importing the real
LLM SDK at all. (Tests pass their own ``plan_llm`` / ``research_agent`` /
``reflect_llm`` / ``artifact_llm`` to ``build_workflow``.)
"""

from __future__ import annotations

import os
from typing import Any

from shared import get_llm, get_structured_llm
from multi_agent.topologies import build_solo
from multi_agent.workers import WorkerSpec

from .schemas import ReflectVerdict, ResearchPlan
from .settings import Settings
from .tools import RESEARCH_TOOLS


def build_plan_llm(settings: Settings):
    return get_structured_llm(settings.plan_model, ResearchPlan)


def build_reflect_llm(settings: Settings):
    return get_structured_llm(settings.reflect_model, ReflectVerdict)


def build_artifact_llm(settings: Settings):
    """Plain (non-structured) chat model. Returns markdown in ``.content``."""
    return get_llm(settings.artifact_model)


def build_research_agent(settings: Settings) -> Any:
    """Tool-using solo agent with the real-internet research tools.

    Returns the ``Topology`` whose ``.invoke({"task": ...})`` returns a
    ``TopologyResult`` with ``.answer``. The agent must successfully
    search + scrape before citing a URL — its system prompt enforces
    that, and the tools themselves return ``"ERROR: ..."`` strings on
    failure that the LLM can recover from.
    """

    spec = WorkerSpec(
        name="researcher",
        description=(
            "Real-internet research specialist with `serpapi_search` and "
            "`firecrawl_scrape` tools. Answers one sub-question per call "
            "with inline [url] citations."
        ),
        system_prompt=(
            "You are a research specialist with two tools: `serpapi_search` "
            "to find candidate sources, and `firecrawl_scrape` to read the "
            "body of a specific URL.\n\n"
            "Workflow for every question:\n"
            "1. Call `serpapi_search` with a focused query.\n"
            "2. Pick the 1-3 most promising results and call "
            "   `firecrawl_scrape` on each to read the body.\n"
            "3. Answer the user's question in 2-4 sentences with inline "
            "   citations in the form [url]. Cite only URLs you actually "
            "   scraped.\n\n"
            "If the tools return errors or no useful content, say so "
            "plainly in one short sentence. Do NOT fabricate URLs or "
            "claims under any circumstances."
        ),
        tools=RESEARCH_TOOLS,
        model_slug=settings.research_model,
    )
    return build_solo([spec], model_slug=settings.research_model)


# ---------------------------------------------------------------------------
# Readiness checks
# ---------------------------------------------------------------------------


def llm_smoke_check(settings: Settings) -> tuple[bool, str]:
    """Cheap readiness probe — does the configured key resolve a model?

    We don't make a real chat call (that costs money on every /readyz hit);
    we just construct the client and check that the OpenRouter key is
    present, mirroring what ``get_llm`` would refuse on first invocation.
    """
    needs_openrouter = any(
        not (slug or "").startswith("ollama/")
        for slug in (
            settings.plan_model,
            settings.research_model,
            settings.reflect_model,
            settings.artifact_model,
        )
    )
    if needs_openrouter and not settings.openrouter_api_key:
        return False, "OPENROUTER_API_KEY is not set"
    try:
        get_llm(settings.research_model)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"llm init failed: {type(exc).__name__}: {exc}"


def search_tools_check(settings: Settings) -> tuple[bool, str]:
    """Readiness probe for the web-research tool credentials.

    SerpAPI is required — without it the agent can't actually search the
    web. Firecrawl is optional; we warn when it's missing but don't fail.
    """
    serp = settings.serpapi_api_key or os.environ.get("SERPAPI_API_KEY", "")
    if not serp:
        return False, "SERPAPI_API_KEY is not set (real web search disabled)"
    fc = settings.firecrawl_api_key or os.environ.get("FIRECRAWL_API_KEY", "")
    if not fc:
        return True, "ok (FIRECRAWL_API_KEY missing — falling back to naive HTML fetch)"
    return True, "ok"
