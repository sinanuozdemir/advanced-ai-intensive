"""LLM-as-judge primitives.

Three Pydantic schemas:
- `RetrievalGrade`     — used by the CRAG-style retrieval grader (nb 3, nb 5).
- `FaithfulnessScore`  — used by the self-correcting RAG inner loop (nb 5).
- `RubricResult`       — used by the eval harness (nb 5) for end-to-end answer grading.

Plus a `judge_with_rubric` convenience that calls a swappable judge LLM with
structured output. Default judge is ``get_llm("anthropic/claude-opus-4.7")`` so
notebooks can show how judge-model choice affects scores.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# CRAG-style retrieval grading
# ---------------------------------------------------------------------------

class RetrievalGrade(BaseModel):
    """Per-chunk relevance grade. Drop chunks where ``relevant=False``."""

    relevant: bool = Field(description="True if the chunk meaningfully answers the question.")
    reason: str = Field(description="One short sentence explaining the verdict.")


_GRADER_SYSTEM = (
    "You are a strict relevance grader for a retrieval-augmented generation system. "
    "Given a user question and a candidate chunk, decide whether the chunk is "
    "directly useful for answering the question. A chunk is RELEVANT only if a "
    "thoughtful answer to the question would meaningfully draw on its content. "
    "Tangentially related chunks are NOT relevant."
)


def grade_chunk(question: str, chunk_text: str, *, judge_llm: Any | None = None) -> RetrievalGrade:
    """Grade a single chunk for relevance to a question."""
    bound = _bind_judge(judge_llm, RetrievalGrade)
    return bound.invoke([
        SystemMessage(content=_GRADER_SYSTEM),
        HumanMessage(content=f"QUESTION:\n{question}\n\nCHUNK:\n{chunk_text}"),
    ])


# ---------------------------------------------------------------------------
# Faithfulness (used inside the adaptive workflow's gap analyzer)
# ---------------------------------------------------------------------------

class FaithfulnessScore(BaseModel):
    """How well a draft answer is supported by the retrieved evidence."""

    faithfulness: int = Field(description="0=hallucinated, 5=fully supported by evidence.")
    missing: list[str] = Field(
        default_factory=list,
        description="Specific facts the draft asserts that are NOT supported.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Sub-questions whose answer would strengthen the draft.",
    )


_FAITHFULNESS_SYSTEM = (
    "You are an evidence auditor. Given a question, a draft answer, and the "
    "evidence chunks the answer was based on, score how faithful the draft is "
    "to the evidence (0-5). Identify any unsupported claims and any open "
    "sub-questions whose evidence would strengthen the draft."
)


def score_faithfulness(
    question: str,
    draft_answer: str,
    evidence: list[str],
    *,
    judge_llm: Any | None = None,
) -> FaithfulnessScore:
    bound = _bind_judge(judge_llm, FaithfulnessScore)
    joined = "\n\n---\n\n".join(evidence) if evidence else "(no evidence retrieved)"
    return bound.invoke([
        SystemMessage(content=_FAITHFULNESS_SYSTEM),
        HumanMessage(
            content=f"QUESTION:\n{question}\n\nDRAFT ANSWER:\n{draft_answer}\n\nEVIDENCE:\n{joined}"
        ),
    ])


# ---------------------------------------------------------------------------
# End-to-end answer rubric (used by the eval harness)
# ---------------------------------------------------------------------------

class RubricResult(BaseModel):
    """Holistic grade of an end-to-end RAG answer against a reference."""

    faithfulness: int = Field(description="0=hallucinated, 5=fully supported.")
    completeness: int = Field(description="0=misses everything, 5=covers all key aspects.")
    correctness: int = Field(description="0=wrong, 5=matches reference answer.")
    conciseness: int = Field(description="0=padded/wandering, 5=tight and on-topic.")
    overall: float = Field(description="Weighted overall (0-5).")
    notes: str = Field(description="One short paragraph justifying the scores.")


_RUBRIC_SYSTEM_BASE = (
    "You are an expert RAG evaluator. Score the candidate ANSWER for the given "
    "QUESTION on four dimensions, each 0-5: "
    "completeness (covers all key facts in the REFERENCE), "
    "correctness (factually agrees with the REFERENCE), "
    "conciseness (no padding). "
    "Then compute `overall` as a weighted average where correctness counts "
    "twice as much as the others. Write a short `notes` field."
)

_RUBRIC_SYSTEM_NO_EVIDENCE = _RUBRIC_SYSTEM_BASE + (
    " Score `faithfulness` as how well the candidate avoids contradicting "
    "the REFERENCE (no contradictions = 5)."
)

_RUBRIC_SYSTEM_WITH_EVIDENCE = _RUBRIC_SYSTEM_BASE + (
    " Score `faithfulness` strictly: every factual claim in the candidate "
    "answer must be directly supported by the EVIDENCE chunks. Claims that "
    "are correct (per the REFERENCE) but NOT in the EVIDENCE are unfaithful "
    "— the model is using pretraining knowledge instead of retrieval, which "
    "is the failure mode RAG is supposed to prevent. "
    "5 = every claim grounded in evidence; "
    "3 = mix of grounded and ungrounded; "
    "1 = mostly fabricated even if coincidentally correct; "
    "0 = directly contradicted by evidence. "
    "An answer that honestly admits 'I don't have evidence for X' is MORE "
    "faithful than one that fabricates X from pretraining, even if X is true."
)


def judge_with_rubric(
    question: str,
    answer: str,
    reference: str,
    *,
    evidence: list[str] | None = None,
    judge_llm: Any | None = None,
) -> RubricResult:
    """Score an answer against a reference using a 4-dimension rubric.

    When ``evidence`` is supplied, ``faithfulness`` is scored strictly
    against the retrieved chunks (the correct behavior for RAG eval —
    a coincidentally-correct hallucination should score LOW). When
    ``evidence`` is None, ``faithfulness`` falls back to a weaker
    "no contradictions of REFERENCE" check.
    """
    bound = _bind_judge(judge_llm, RubricResult)
    if evidence is None:
        sys_prompt = _RUBRIC_SYSTEM_NO_EVIDENCE
        user_msg = (
            f"QUESTION:\n{question}\n\n"
            f"REFERENCE ANSWER:\n{reference}\n\n"
            f"CANDIDATE ANSWER:\n{answer}"
        )
    else:
        sys_prompt = _RUBRIC_SYSTEM_WITH_EVIDENCE
        joined = "\n\n---\n\n".join(evidence) if evidence else "(no evidence retrieved)"
        user_msg = (
            f"QUESTION:\n{question}\n\n"
            f"REFERENCE ANSWER:\n{reference}\n\n"
            f"EVIDENCE ({len(evidence)} chunks):\n{joined}\n\n"
            f"CANDIDATE ANSWER:\n{answer}"
        )
    return bound.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_msg),
    ])


# ---------------------------------------------------------------------------
# Trajectory scoring (for agents in nb 5)
# ---------------------------------------------------------------------------

class TrajectoryScore(BaseModel):
    """Subjective grade of an agent's tool-call trajectory."""

    efficiency: int = Field(description="0=lots of wasted calls, 5=minimal trajectory.")
    coherence: int = Field(description="Did each tool call build on the last?")
    notes: str = Field(description="What went well or wrong.")


_TRAJECTORY_SYSTEM = (
    "You are an agent-trajectory auditor. Given a question and a sequence of "
    "(tool_name, tool_input, tool_output) tuples that an agent invoked, score "
    "the trajectory's efficiency and coherence (0-5 each)."
)


def score_trajectory(
    question: str,
    trajectory: list[dict],
    *,
    judge_llm: Any | None = None,
) -> TrajectoryScore:
    bound = _bind_judge(judge_llm, TrajectoryScore)
    pretty = "\n".join(
        f"- {t.get('tool_name','?')}({_short(t.get('tool_input',''))}) -> {_short(t.get('tool_output',''))}"
        for t in trajectory
    ) or "(no tool calls)"
    return bound.invoke([
        SystemMessage(content=_TRAJECTORY_SYSTEM),
        HumanMessage(content=f"QUESTION:\n{question}\n\nTRAJECTORY:\n{pretty}"),
    ])


def _short(x: Any, n: int = 200) -> str:
    s = str(x)
    return s if len(s) <= n else s[: n - 3] + "..."


def _bind_judge(judge_llm: Any | None, schema: type) -> Any:
    """Bind a Pydantic schema to a judge LLM using function-calling +
    one-shot validation retry. If `judge_llm` is None, default to
    Claude Opus 4.7 via `get_structured_llm` (function-calling, generous
    max_tokens, retry on ValidationError). If a custom LLM is provided,
    we still wrap it in function-calling mode for robustness.
    """
    from llm import get_structured_llm, _StructuredRetryWrapper

    if judge_llm is None:
        return get_structured_llm("anthropic/claude-opus-4.7", schema)
    bound = judge_llm.with_structured_output(schema, method="function_calling")
    return _StructuredRetryWrapper(bound=bound, schema=schema, max_retries=1)


__all__ = [
    "RetrievalGrade",
    "FaithfulnessScore",
    "RubricResult",
    "TrajectoryScore",
    "grade_chunk",
    "score_faithfulness",
    "judge_with_rubric",
    "score_trajectory",
]
