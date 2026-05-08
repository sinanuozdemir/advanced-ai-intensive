"""Conversation-compression middlewares for ``create_agent``.

Strategies (Week 2 Segment 4 + Week 1 Notebook 4):

- :class:`RecursiveSummaryMiddleware` — maintain a running summary; refresh
  every ``update_every`` turns; never dropped.
- :class:`HierarchicalCompressionMiddleware` — last ``keep_last`` verbatim,
  next ``mid_window`` as paragraph summaries, older messages dropped (in a
  real system they'd go to episodic memory; we leave a marker line).
- :class:`MapReduceSummarizationMiddleware` — chunk old messages into ``k``
  groups, summarize each, then summarize the summaries.
- :class:`RefineSummarizationMiddleware` — start with the first chunk's
  summary, refine it iteratively across the rest.
- :class:`RulesFirstSummaryMiddleware` — extract rules + facts as structured
  fields first, then write the prose summary; rules sit at the top of the
  injected message.

The plain "sliding window with summary tail" pattern is just LangChain's
built-in ``SummarizationMiddleware``; no custom version lives here.

All accept ``trigger`` in the same shape as ``SummarizationMiddleware``
(``("messages", n)`` or ``("tokens", n)``) and tag the produced summary with
``additional_kwargs={"lc_source": "summarization", "strategy": <name>}`` so a
notebook can pluck it out.
"""
from __future__ import annotations

from typing import Any, Iterable, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from typing_extensions import override

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT
from langchain.chat_models import init_chat_model

ContextSize = tuple[Literal["messages", "tokens"], int]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_model(model: str | BaseChatModel) -> BaseChatModel:
    if isinstance(model, str):
        return init_chat_model(model)
    return model


def _format_turn(m: AnyMessage) -> str:
    role = m.__class__.__name__.replace("Message", "").upper()
    content = m.content if isinstance(m.content, str) else str(m.content)
    return f"{role}: {content.strip()}"


def _format_messages(messages: Iterable[AnyMessage]) -> str:
    return "\n\n".join(_format_turn(m) for m in messages)


def _summarize(model: BaseChatModel, prompt: str) -> str:
    """One-shot LLM call that returns a string summary."""
    out = model.invoke([HumanMessage(content=prompt)])
    return (out.content if isinstance(out.content, str) else str(out.content)).strip()


def _build_summary_message(text: str, *, strategy: str, header: str) -> HumanMessage:
    return HumanMessage(
        content=f"{header}\n\n{text}",
        additional_kwargs={"lc_source": "summarization", "strategy": strategy},
    )


def _is_summary_message(m: AnyMessage) -> bool:
    return (
        isinstance(m, HumanMessage)
        and (m.additional_kwargs or {}).get("lc_source") == "summarization"
    )


def _trigger_fires(
    messages: list[AnyMessage], trigger: ContextSize | None
) -> bool:
    if trigger is None:
        return False
    kind, value = trigger
    if kind == "messages":
        return len(messages) >= value
    if kind == "tokens":
        return count_tokens_approximately(messages) >= value
    return False


def _split_keep_tail(
    messages: list[AnyMessage], keep_last: int
) -> tuple[list[AnyMessage], list[AnyMessage]]:
    """Split off the last ``keep_last`` messages, never breaking an
    AIMessage(tool_calls) -> ToolMessage pair."""
    if keep_last <= 0 or keep_last >= len(messages):
        return messages, []
    cutoff = len(messages) - keep_last
    while cutoff < len(messages) and isinstance(messages[cutoff], ToolMessage):
        cutoff += 1
    return messages[:cutoff], messages[cutoff:]


# ---------------------------------------------------------------------------
# 1. Recursive running summary
# ---------------------------------------------------------------------------


_RECURSIVE_INIT_PROMPT = """\
Write a single concise running summary (4-7 sentences) of the conversation
below. Capture: user goals, any rules or constraints they stated, decisions
made, open threads. Be faithful — no facts that aren't in the transcript.

CONVERSATION:
{transcript}
"""

_RECURSIVE_UPDATE_PROMPT = """\
You are maintaining a running summary of an ongoing conversation. Update the
existing summary with the new turns below. Keep it concise (4-7 sentences).
Preserve any rules, constraints, or commitments. Drop stale details that are
no longer relevant. Do not invent facts.

EXISTING SUMMARY:
{summary}

NEW TURNS:
{transcript}

UPDATED SUMMARY:
"""


class RecursiveSummaryMiddleware(AgentMiddleware):
    """Maintain a running summary; refresh every ``update_every`` new turns."""

    strategy_name = "recursive_summary"

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        update_every: int = 5,
        keep_last: int = 4,
        trigger: ContextSize | None = ("messages", 10),
    ) -> None:
        super().__init__()
        self.model = _resolve_model(model)
        self.update_every = update_every
        self.keep_last = keep_last
        self.trigger = trigger

    @override
    def before_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        messages = list(state["messages"])
        if not _trigger_fires(messages, self.trigger):
            return None

        existing_summary = next(
            (m.content for m in messages if _is_summary_message(m)), None
        )
        old, kept = _split_keep_tail(messages, self.keep_last)
        new_old = [m for m in old if not _is_summary_message(m)]
        if not new_old:
            return None

        if existing_summary:
            prompt = _RECURSIVE_UPDATE_PROMPT.format(
                summary=existing_summary,
                transcript=_format_messages(new_old),
            )
        else:
            prompt = _RECURSIVE_INIT_PROMPT.format(
                transcript=_format_messages(new_old)
            )
        summary_text = _summarize(self.model, prompt)
        summary_msg = _build_summary_message(
            summary_text,
            strategy=self.strategy_name,
            header="Running summary of the conversation so far:",
        )
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                summary_msg,
                *kept,
            ]
        }


# ---------------------------------------------------------------------------
# 2. Hierarchical compression
# ---------------------------------------------------------------------------


_HIERARCHY_PROMPT = """\
Summarize the conversation snippet below into 1-2 short paragraphs. Preserve
any rules, constraints, or commitments. Skip greetings and small talk.

CONVERSATION:
{transcript}
"""


class HierarchicalCompressionMiddleware(AgentMiddleware):
    """Last ``keep_last`` verbatim; next ``mid_window`` as a paragraph; older dropped.

    The dropped chunk is replaced with a single marker line so the agent knows
    older history existed (in production this is where you'd write to
    episodic memory — see ``notebooks/week2/2_memory_systems.ipynb``).
    """

    strategy_name = "hierarchical"

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        keep_last: int = 5,
        mid_window: int = 10,
        trigger: ContextSize | None = ("messages", 16),
    ) -> None:
        super().__init__()
        self.model = _resolve_model(model)
        self.keep_last = keep_last
        self.mid_window = mid_window
        self.trigger = trigger

    @override
    def before_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        messages = list(state["messages"])
        if not _trigger_fires(messages, self.trigger):
            return None

        non_summary = [m for m in messages if not _is_summary_message(m)]
        if len(non_summary) <= self.keep_last:
            return None

        # head: oldest messages (dropped to a marker)
        # mid: paragraph-summarized chunk
        # tail: kept verbatim
        cutoff_tail = len(non_summary) - self.keep_last
        cutoff_mid = max(0, cutoff_tail - self.mid_window)
        head = non_summary[:cutoff_mid]
        mid = non_summary[cutoff_mid:cutoff_tail]
        tail = non_summary[cutoff_tail:]
        # Walk forward if tail starts with an orphan ToolMessage.
        while tail and isinstance(tail[0], ToolMessage):
            mid.append(tail.pop(0))

        new_head = []
        if head:
            new_head.append(
                _build_summary_message(
                    f"[{len(head)} earlier messages dropped — in production these "
                    "would have moved to episodic long-term memory.]",
                    strategy=self.strategy_name,
                    header="Older history (compressed away):",
                )
            )
        if mid:
            mid_text = _summarize(
                self.model,
                _HIERARCHY_PROMPT.format(transcript=_format_messages(mid)),
            )
            new_head.append(
                _build_summary_message(
                    mid_text,
                    strategy=self.strategy_name,
                    header="Mid-window summary (older but still relevant):",
                )
            )

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_head,
                *tail,
            ]
        }


# ---------------------------------------------------------------------------
# 3. Map-reduce summarization (mirrors week 1)
# ---------------------------------------------------------------------------


_MAP_PROMPT = """\
Summarize this conversation chunk in 2-4 sentences. Preserve any rules,
constraints, or commitments stated by the user. Skip pleasantries.

CHUNK:
{transcript}
"""

_REDUCE_PROMPT = """\
You are combining several partial summaries of one continuous conversation
into a single faithful overall summary (5-8 sentences). Preserve every rule
or constraint mentioned in any partial summary.

PARTIAL SUMMARIES:
{partials}

OVERALL SUMMARY:
"""


class MapReduceSummarizationMiddleware(AgentMiddleware):
    """Chunk older messages into ``k`` groups, map-summarize, then reduce."""

    strategy_name = "map_reduce"

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        k: int = 3,
        keep_last: int = 4,
        trigger: ContextSize | None = ("messages", 12),
    ) -> None:
        super().__init__()
        self.model = _resolve_model(model)
        self.k = max(2, k)
        self.keep_last = keep_last
        self.trigger = trigger

    @override
    def before_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        messages = list(state["messages"])
        if not _trigger_fires(messages, self.trigger):
            return None
        old, kept = _split_keep_tail(messages, self.keep_last)
        non_summary_old = [m for m in old if not _is_summary_message(m)]
        if len(non_summary_old) < self.k:
            return None

        chunks = self._chunk(non_summary_old, self.k)
        partials = [
            _summarize(
                self.model,
                _MAP_PROMPT.format(transcript=_format_messages(chunk)),
            )
            for chunk in chunks
            if chunk
        ]
        partials_block = "\n\n".join(
            f"[chunk {i + 1}] {p}" for i, p in enumerate(partials)
        )
        overall = _summarize(
            self.model,
            _REDUCE_PROMPT.format(partials=partials_block),
        )
        summary_msg = _build_summary_message(
            overall,
            strategy=self.strategy_name,
            header="Map-reduce summary of earlier conversation:",
        )
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                summary_msg,
                *kept,
            ]
        }

    @staticmethod
    def _chunk(items: list[AnyMessage], k: int) -> list[list[AnyMessage]]:
        if not items:
            return []
        per = max(1, (len(items) + k - 1) // k)
        return [items[i : i + per] for i in range(0, len(items), per)]


# ---------------------------------------------------------------------------
# 4. Iterative refine summarization (mirrors week 1)
# ---------------------------------------------------------------------------


_REFINE_INIT_PROMPT = """\
Summarize this opening conversation chunk in 3-5 sentences. Preserve any
rules, constraints, or commitments stated by the user.

CHUNK:
{transcript}
"""

_REFINE_STEP_PROMPT = """\
You are maintaining a faithful running summary of one continuous
conversation. Refine the existing draft using the new chunk below. Preserve
every rule or constraint mentioned anywhere. Keep length to 5-8 sentences.
Do not invent facts.

EXISTING DRAFT:
{draft}

NEW CHUNK:
{transcript}

REFINED DRAFT:
"""


class RefineSummarizationMiddleware(AgentMiddleware):
    """Start a draft from chunk 1, refine it through chunks 2..k."""

    strategy_name = "refine"

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        k: int = 3,
        keep_last: int = 4,
        trigger: ContextSize | None = ("messages", 12),
    ) -> None:
        super().__init__()
        self.model = _resolve_model(model)
        self.k = max(2, k)
        self.keep_last = keep_last
        self.trigger = trigger

    @override
    def before_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        messages = list(state["messages"])
        if not _trigger_fires(messages, self.trigger):
            return None
        old, kept = _split_keep_tail(messages, self.keep_last)
        non_summary_old = [m for m in old if not _is_summary_message(m)]
        if len(non_summary_old) < self.k:
            return None

        chunks = MapReduceSummarizationMiddleware._chunk(non_summary_old, self.k)
        if not chunks:
            return None

        draft = _summarize(
            self.model,
            _REFINE_INIT_PROMPT.format(transcript=_format_messages(chunks[0])),
        )
        for chunk in chunks[1:]:
            draft = _summarize(
                self.model,
                _REFINE_STEP_PROMPT.format(
                    draft=draft, transcript=_format_messages(chunk)
                ),
            )

        summary_msg = _build_summary_message(
            draft,
            strategy=self.strategy_name,
            header="Iteratively refined summary of earlier conversation:",
        )
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                summary_msg,
                *kept,
            ]
        }


# ---------------------------------------------------------------------------
# 5. Rules-first summarization
# ---------------------------------------------------------------------------
#
# The other middlewares all ask one open-ended question: "summarize the
# conversation". Whether a rule survives is incidental.
#
# This one asks a *structured* question first: extract rules + important
# facts as separate fields, then write the summary. The produced summary
# message is shaped as Markdown so the rules sit at the top of the message,
# right where the model is most likely to attend to them.
#
# Output shape:
#
#   # Rules
#   - rule 1
#   - rule 2
#
#   # Important facts
#   - fact 1
#
#   # Summary of what has happened so far
#   <prose>


from pydantic import BaseModel, Field  # noqa: E402  (kept local to this section)


class _RulesFirstExtraction(BaseModel):
    """Structured extraction the rules-first middleware asks the LLM for."""

    rules: list[str] = Field(
        default_factory=list,
        description=(
            "Verbatim or near-verbatim user-stated rules, constraints, or "
            "commitments from the conversation that the agent must keep "
            "obeying. Things like 'don't delete X without permission' or "
            "'only send email to specific addresses'. Do NOT infer; only "
            "include rules that were explicitly stated. One bullet per rule."
        ),
    )
    important_facts: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete facts established in the conversation that the agent "
            "would need to do its job (names, ids, deadlines, dollar "
            "amounts, decisions made). Skip pleasantries."
        ),
    )
    summary: str = Field(
        description=(
            "A 4-7 sentence narrative of what has happened in the "
            "conversation, written for an agent that will pick up where this "
            "one left off."
        ),
    )


_RULES_FIRST_PROMPT = """\
You are compressing a long conversation between a user and an email-management
agent. Your job is to produce a structured extraction so the agent can keep
working without losing critical context.

Extract three things, in order:

1. RULES — every user-stated rule, constraint, or commitment that the agent
   must keep obeying. Look for phrases like "don't ___", "always ___",
   "only ___", "never send to ___ unless ___", "ask first before ___".
   These are non-negotiable for the rest of the session.
2. IMPORTANT FACTS — concrete things the agent will need to keep doing its
   job: names, ids, dates, amounts, decisions already made, drafts already
   created.
3. SUMMARY — a short narrative (4-7 sentences) of what happened in the
   conversation.

Be faithful. Do not invent rules or facts. Do not paraphrase rules in a way
that weakens them.

Respond as JSON with EXACTLY this shape:
{{
  "rules": ["<one rule per string>", "..."],
  "important_facts": ["<one fact per string>", "..."],
  "summary": "<4-7 sentence narrative>"
}}

CRITICAL: `rules` and `important_facts` MUST be flat JSON arrays of plain
strings. Do NOT nest objects, dicts, or sub-arrays inside them. If a fact
has structure (e.g. an invoice with id, amount, due date), flatten it into
one descriptive string like "Invoice INV-2026-Q3-0184 for $48,200 from
finance-team@acme.com, due Oct 27".

CONVERSATION:
{transcript}
"""


def _format_rules_first(extraction: _RulesFirstExtraction) -> str:
    rules_block = "\n".join(f"- {r}" for r in extraction.rules) or "- (none stated)"
    facts_block = "\n".join(f"- {f}" for f in extraction.important_facts) or "- (none)"
    return (
        "# Rules\n"
        f"{rules_block}\n\n"
        "# Important facts\n"
        f"{facts_block}\n\n"
        "# Summary of what has happened so far\n"
        f"{extraction.summary.strip()}"
    )


class RulesFirstSummaryMiddleware(AgentMiddleware):
    """Compression that extracts rules + facts as structured fields first.

    Differs from every other middleware in this module in two ways:
    1. The LLM is asked for a Pydantic-schema-shaped extraction, not free
       prose. Rules and facts are first-class fields, not buried in a
       paragraph.
    2. The injected ``HumanMessage`` is Markdown-structured with a ``# Rules``
       section at the top, so a reader (model or human) sees the constraints
       before any narrative.

    Empirically tends to preserve early-stated rules better than the other
    strategies because rule extraction is the *primary* objective of the
    LLM call, not a side effect.
    """

    strategy_name = "rules_first"

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        keep_last: int = 4,
        trigger: ContextSize | None = ("messages", 12),
    ) -> None:
        super().__init__()
        self.model = _resolve_model(model)
        self.keep_last = keep_last
        self.trigger = trigger
        # `method="json_mode"` instead of the default function-calling so this
        # middleware works through OpenRouter's Bedrock-routed Anthropic
        # (which rejects the function-calling schema field).
        self._structured = self.model.with_structured_output(_RulesFirstExtraction, method="json_mode")

    @override
    def before_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        messages = list(state["messages"])
        if not _trigger_fires(messages, self.trigger):
            return None
        old, kept = _split_keep_tail(messages, self.keep_last)
        non_summary_old = [m for m in old if not _is_summary_message(m)]
        if not non_summary_old:
            return None

        extraction: _RulesFirstExtraction = self._structured.invoke(
            _RULES_FIRST_PROMPT.format(transcript=_format_messages(non_summary_old))
        )
        text = _format_rules_first(extraction)
        summary_msg = HumanMessage(
            content=text,
            additional_kwargs={"lc_source": "summarization", "strategy": self.strategy_name},
        )
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                summary_msg,
                *kept,
            ]
        }
