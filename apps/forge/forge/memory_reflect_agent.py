"""Reflection as an agent loop, not a one-shot structured-output call.

End-of-thread reflection used to be a single LLM call that produced a
summary + a list of new procedural skills (see ``src/memory/reflect.py``).
That call could not see existing skills, which is why the procedural store
drifts into near-duplicates over time (``search_api_auth_failure_handling``
vs ``research_tool_auth_failure_handling`` etc.).

This module replaces the Forge-side reflection with a ``create_agent``
loop. The agent gets four tools and a system prompt that forces it to
search before writing:

  * ``list_existing_skills`` — orientation: every skill's name + cue.
  * ``search_procedural``    — cosine search over ``when_to_use`` cues.
  * ``save_episode``         — write the thread summary (exactly once).
  * ``save_skill``           — write a NEW procedural skill (skip-only,
                               per the merge policy decision: never
                               touches existing rows).

Tools emit the same ``tool_call`` / ``tool_result`` / ``memory_write``
trace events that ``build_memory_tools`` uses, so reflection activity is
indistinguishable from a normal agent turn in the live trace stream and
in the Memory panel.

The standalone ``src/memory/reflect.py`` is intentionally not touched —
it is still imported by the week-2 notebook and the SDR demo app.
"""
from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from memory.episodic import EpisodicEntry
from memory.procedural import ProceduralSkill
from shared import get_llm

from .memory import MemoryStores, _messages_to_dicts
from .trace import Tracer


_AGENT_NAME = "reflector"

_REFLECT_SYSTEM = """\
You are reviewing an ongoing conversation thread to maintain a single rolling
summary plus, occasionally, reusable procedural skills.

Reflection runs after every chat turn. The episode tool is an UPSERT keyed
on this thread — there will be at most one episode row per thread, and your
write replaces whatever is there. If a PRIOR EPISODE block is provided in
the user message, your new summary should be a refinement of it that
incorporates the latest turn, not a start-from-scratch rewrite that loses
earlier context.

You MUST:
  1. Call save_episode exactly once with a 2-4 sentence cumulative summary
     covering the whole thread so far: what the user wanted, what was done,
     and whether it succeeded. Refine the prior episode if one is shown.
  2. Decide whether the latest turn surfaced any reusable procedural skill
     — a non-obvious technique, recovery strategy, or domain rule worth
     applying in future threads. Most turns surface none.

Before writing a procedural skill:
  - Call list_existing_skills first to see what's already captured.
  - Call search_procedural with a short query describing your candidate
    skill's trigger so you can spot near-duplicates the name list might
    miss.
  - If any existing skill already covers that trigger or technique, DO NOT
    write a duplicate. Returning zero new skills is the right answer.
  - Only call save_skill when you've confirmed the situation is genuinely
    not captured by any existing entry.

Bias toward zero new skills. One is acceptable when truly novel. Two or
more should be rare. When you are done, reply with a one-sentence note
about what (if anything) you saved — that final message is for the trace,
nothing else reads it.
"""


def _format_thread(messages: list[dict]) -> str:
    """Render a [{role, content}] transcript as a flat string for the user msg."""
    out = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        out.append(f"{role}:\n{content}")
    return "\n\n".join(out)


def _build_reflection_tools(
    *,
    stores: MemoryStores,
    tracer: Tracer | None,
    thread_id: str,
    rubric_score: float,
    episode_state: dict,
    skills_written: list[str],
) -> list[Any]:
    """Construct the four reflection tools, all closing over the same
    shared state so we can enforce "save_episode only once" and collect
    saved skill names without an extra return-channel."""

    def _emit(event_type: str, **fields: Any) -> None:
        if tracer is not None:
            tracer.emit(event_type, **fields)

    @tool
    def list_existing_skills() -> str:
        """List every existing procedural skill (name + when_to_use cue).
        Call this FIRST to orient yourself before deciding to save a new
        skill — most thread learnings are already captured."""
        _emit(
            "tool_call", agent_name=_AGENT_NAME,
            tool="list_existing_skills", args={},
        )
        try:
            rows = stores.procedural.top(n=50)
        except Exception as exc:  # noqa: BLE001
            _emit(
                "tool_result", agent_name=_AGENT_NAME,
                tool="list_existing_skills", ok=False,
                preview=f"error: {exc!r}"[:240],
            )
            raise
        if not rows:
            result = "(no skills yet)"
        else:
            lines = [
                f"- {r.name}: {r.when_to_use or '(no cue)'}"
                for r in rows
            ]
            result = "\n".join(lines)
        _emit(
            "tool_result", agent_name=_AGENT_NAME,
            tool="list_existing_skills", ok=True,
            preview=result[:240],
        )
        return result

    @tool
    def search_procedural(query: str) -> str:
        """Semantic search over existing skill `when_to_use` cues. Use
        this to confirm a candidate skill isn't a rephrase of one that
        already exists.

        Args:
            query: A short trigger description ("authentication errors
                during web search", "user wants a deep dive on a person").
        """
        _emit(
            "tool_call", agent_name=_AGENT_NAME,
            tool="search_procedural", args={"query": query},
        )
        try:
            hits = stores.procedural.search_when(
                query, k=5, min_score=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            _emit(
                "tool_result", agent_name=_AGENT_NAME,
                tool="search_procedural", ok=False,
                preview=f"error: {exc!r}"[:240],
            )
            raise
        if not hits:
            result = "(no matching skills)"
        else:
            lines = []
            for skill, sim in hits:
                frag = (skill.fragment or "").strip().replace("\n", " ")
                if len(frag) > 120:
                    frag = frag[:117] + "..."
                lines.append(
                    f"- {skill.name} (sim={sim:.2f}) "
                    f"when: {skill.when_to_use or '(no cue)'} :: {frag}",
                )
            result = "\n".join(lines)
        _emit(
            "tool_result", agent_name=_AGENT_NAME,
            tool="search_procedural", ok=True,
            preview=result[:240],
        )
        return result

    @tool
    def save_episode(summary: str) -> str:
        """Persist the rolling thread summary. Upserts on ``thread_id``
        — there is at most one episode row per thread, so this call
        REPLACES any prior summary for the current thread. Call this
        EXACTLY ONCE per reflection; a second call within the same
        reflection returns an error.

        Args:
            summary: 2-4 sentence cumulative summary covering the whole
                thread so far — what the user wanted, what the agent
                did, and whether it succeeded. Refine the prior episode
                if one was shown in the user message.
        """
        _emit(
            "tool_call", agent_name=_AGENT_NAME,
            tool="save_episode", args={"summary": summary},
        )
        if episode_state.get("written"):
            msg = "error: save_episode already called for this reflection"
            _emit(
                "tool_result", agent_name=_AGENT_NAME,
                tool="save_episode", ok=False, preview=msg,
            )
            return msg
        try:
            stores.episodic.upsert_by_thread(EpisodicEntry(
                summary=summary,
                thread_id=thread_id,
                score=float(rubric_score),
            ))
        except Exception as exc:  # noqa: BLE001
            _emit(
                "tool_result", agent_name=_AGENT_NAME,
                tool="save_episode", ok=False,
                preview=f"error: {exc!r}"[:240],
            )
            raise
        episode_state["written"] = True
        episode_state["summary"] = summary
        action = "updated" if episode_state.get("had_prior") else "created"
        _emit(
            "memory_write", store="episodic",
            thread_id=thread_id, summary=summary[:200], action=action,
        )
        _emit(
            "tool_result", agent_name=_AGENT_NAME,
            tool="save_episode", ok=True, preview=f"{action}",
        )
        return action

    @tool
    def save_skill(name: str, fragment: str, when_to_use: str) -> str:
        """Persist a NEW procedural skill. Only call this AFTER searching
        and confirming no existing skill covers the same trigger.

        Args:
            name: short snake_case identifier.
            fragment: 1-3 sentences of guidance, written as an
                instruction to a future agent ("If X happens, do Y.").
            when_to_use: one-line trigger cue ("When user asks about
                cold outreach to a fintech CTO.").
        """
        _emit(
            "tool_call", agent_name=_AGENT_NAME,
            tool="save_skill",
            args={"name": name, "when_to_use": when_to_use},
        )
        try:
            stores.procedural.save(ProceduralSkill(
                name=name,
                fragment=fragment,
                when_to_use=when_to_use,
                score=float(rubric_score),
            ))
        except Exception as exc:  # noqa: BLE001
            _emit(
                "tool_result", agent_name=_AGENT_NAME,
                tool="save_skill", ok=False,
                preview=f"error: {exc!r}"[:240],
            )
            raise
        skills_written.append(name)
        _emit(
            "memory_write", store="procedural",
            thread_id=thread_id, skill=name,
        )
        _emit(
            "tool_result", agent_name=_AGENT_NAME,
            tool="save_skill", ok=True, preview=f"stored {name}",
        )
        return f"stored {name}"

    return [list_existing_skills, search_procedural, save_episode, save_skill]


def reflect_with_agent(
    *,
    stores: MemoryStores,
    tracer: Tracer | None,
    thread_id: str,
    messages: list[Any],
    rubric_score: float = 0.0,
    model_slug: str = "openai/gpt-5.4-nano",
) -> dict:
    """Run reflection as an agent. Returns ``{"summary", "skills"}``,
    matching the contract of the legacy ``reflect_main_thread``.

    ``messages`` is the LangChain message list from the main agent's
    transcript — same shape the engine passes today. Failures in the
    agent loop are swallowed (with the error noted in the result) so a
    bad reflection never sinks an otherwise-successful thread.
    """
    if tracer is not None:
        tracer.emit(
            "agent_spawn", agent_name=_AGENT_NAME,
            kind="reflector", parent="main", thread_id=thread_id,
        )

    # Look up any prior episode for this thread so the agent knows it's
    # refining a rolling summary, not starting from zero. We surface the
    # prior summary inline in the user message rather than spending a tool
    # call on it — there's at most one and the agent always needs it.
    prior = stores.episodic.get_by_thread(thread_id)
    episode_state: dict[str, Any] = {
        "written": False,
        "summary": "",
        "had_prior": prior is not None,
    }
    skills_written: list[str] = []

    tools = _build_reflection_tools(
        stores=stores,
        tracer=tracer,
        thread_id=thread_id,
        rubric_score=rubric_score,
        episode_state=episode_state,
        skills_written=skills_written,
    )

    transcript = _format_thread(_messages_to_dicts(messages))
    prior_block = (
        f"PRIOR EPISODE for this thread (refine this — your save_episode "
        f"call will REPLACE it):\n{prior.summary}\n\n"
        if prior is not None and prior.summary
        else ""
    )
    user_msg = f"{prior_block}Thread {thread_id}:\n\n{transcript}"

    try:
        agent = create_agent(
            model=get_llm(model_slug),
            tools=tools,
            system_prompt=_REFLECT_SYSTEM,
        )
        agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    except Exception as exc:  # noqa: BLE001
        if tracer is not None:
            tracer.emit(
                "agent_done", agent_name=_AGENT_NAME,
                result=f"error: {exc!r}"[:240],
            )
        return {
            "summary": episode_state.get("summary", ""),
            "skills": list(skills_written),
            "error": repr(exc),
        }

    if tracer is not None:
        tracer.emit(
            "agent_done", agent_name=_AGENT_NAME,
            result=(
                f"summary={'yes' if episode_state.get('written') else 'no'} "
                f"skills={len(skills_written)}"
            ),
        )

    return {
        "summary": episode_state.get("summary", ""),
        "skills": list(skills_written),
    }


__all__ = ["reflect_with_agent"]
