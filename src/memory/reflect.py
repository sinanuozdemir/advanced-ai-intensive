"""End-of-thread reflection.

When a thread concludes, we run one LLM call that:
  1. Summarizes the thread (-> EpisodicMemory).
  2. Proposes 0-3 new procedural skills if the agent struggled or did
     something noteworthy (-> ProceduralMemory).

Semantic memory is NOT touched here — it's already been written turn by turn.

This split (real-time semantic / batched episodic+procedural) is the central
design choice of the segment: the hot path stays cheap; the expensive
reasoning happens once per thread.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from shared import get_structured_llm

if TYPE_CHECKING:
    from .episodic import EpisodicMemory
    from .procedural import ProceduralMemory


_REFLECT_PROMPT = """\
You are an agent reviewing a completed conversation thread to extract reusable
learnings.

Produce two things:

  1. A 2-4 sentence summary of what the user wanted, what the agent did, and
     whether it succeeded. This goes into episodic memory and will be retrieved
     when a similar request arrives later.

  2. Zero to three NEW procedural skills, ONLY if the agent struggled or
     discovered a non-obvious technique worth reusing. A skill is a short
     prompt fragment to inject in future threads. Each skill needs:
     - name: short, snake_case
     - fragment: 1-3 sentences of guidance, written as instruction to the agent
     - when_to_use: one line cue describing the trigger

Be conservative on skills. Returning [] for skills is the right answer
when nothing noteworthy happened.
"""


class _Skill(BaseModel):
    name: str
    fragment: str
    when_to_use: str = ""


class _Reflection(BaseModel):
    summary: str
    skills: list[_Skill] = Field(default_factory=list)


def _format_thread(messages: list[dict]) -> str:
    """`messages` is [{role, content}, ...]."""
    out = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        out.append(f"{role}:\n{content}")
    return "\n\n".join(out)


def reflect_on_thread(
    *,
    thread_id: str,
    messages: list[dict],
    episodic: "EpisodicMemory",
    procedural: "ProceduralMemory",
    rubric_score: float = 0.0,
    model_slug: str = "openai/gpt-5.4-nano",
) -> dict:
    """Reflect on a completed thread, persist to episodic + procedural memory.

    Returns a dict with the summary and any skills written.
    """
    from .episodic import EpisodicEntry
    from .procedural import ProceduralSkill

    llm = get_structured_llm(model_slug, _Reflection)
    transcript = _format_thread(messages)
    try:
        result = llm.invoke([
            {"role": "system", "content": _REFLECT_PROMPT},
            {"role": "user", "content": f"Thread {thread_id}:\n\n{transcript}"},
        ])
    except Exception as exc:  # noqa: BLE001
        return {"summary": "", "skills": [], "error": repr(exc)}

    episodic.write(EpisodicEntry(
        summary=result.summary,
        thread_id=thread_id,
        score=float(rubric_score),
    ))
    skills_written = []
    for sk in result.skills:
        procedural.save(ProceduralSkill(
            name=sk.name, fragment=sk.fragment, when_to_use=sk.when_to_use,
            score=float(rubric_score),
        ))
        skills_written.append(sk.name)

    return {
        "summary": result.summary,
        "skills": skills_written,
    }
