"""OpenRouter-backed LLM factory (canonical copy under ``src/shared``).

Used by course notebooks (via ``notebooks/week1/llm.py`` shim), Week 2
libraries, and apps. One key (`OPENROUTER_API_KEY`), one client pattern
(`ChatOpenAI` at the OpenRouter base URL). Resolve models by registry *role*
(e.g. ``get_llm("cheap_workhorse")``) or raw OpenRouter slug.

If a slug is unavailable on OpenRouter at runtime, edit `MODEL_REGISTRY`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Named roles -> OpenRouter slugs. Verified against the OpenRouter catalog
# (https://openrouter.ai/models). Last refreshed: 2026-04-28.
# Swap any slug here as the catalog evolves — every notebook resolves through
# this registry, so a single edit propagates.
MODEL_REGISTRY: dict[str, str] = {
    # Small / dirt-cheap workhorse — current OpenAI cheap tier
    "cheap_workhorse":     "openai/gpt-5.4-nano",
    # General-purpose "frontier chat" — Anthropic's flagship as of Apr 2026.
    "frontier_chat":       "anthropic/claude-opus-4.7",
    # Vendor-specific frontier slots (handy when notebooks compare across vendors).
    "frontier_openai":     "openai/gpt-5.5",
    "frontier_anthropic":  "anthropic/claude-opus-4.7",
    # Dedicated reasoning models. o4-mini is still OpenAI's most cost-effective
    # reasoner; Kimi K2 Thinking
    "reasoning_openai":    "openai/o4-mini",
    "reasoning_open":      "moonshotai/kimi-k2-thinking",
    # Qwen3.6 35B-A3B is a recent MoE — very cheap on OpenRouter, strong
    # instruction following.
    "open_weight":         "qwen/qwen3.6-35b-a3b",
    # Cheap-and-fast slot for high-throughput streaming / agentic loops.
    "fast_open":           "x-ai/grok-4.1-fast",
}


# Approximate $/1M tokens for the registry. Used by CostTrackingLLM as a
# fallback when OpenRouter doesn't return cost in the response payload.
# Verified against OpenRouter pricing on 2026-04-28.
_PRICE_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    # OpenAI
    "openai/gpt-5.4-nano":                   (0.20, 1.25),
    "openai/gpt-5.4-mini":                   (0.75, 4.50),
    "openai/gpt-5.4":                        (2.50, 10.00),
    "openai/gpt-5.5":                        (5.00, 20.00),
    "openai/gpt-5.5-pro":                    (10.00, 40.00),
    "openai/o4-mini":                        (1.10, 4.40),
    # Legacy/transition slugs — kept so older results files still resolve and
    # so notebook 1's "OpenAI evolution timeline" cell has accurate prices.
    "openai/gpt-3.5-turbo":                  (0.50, 1.50),
    "openai/gpt-4-turbo":                    (10.00, 30.00),
    "openai/gpt-4o":                         (2.50, 10.00),
    "openai/gpt-4o-mini":                    (0.15, 0.60),
    "openai/gpt-4.1-mini":                   (0.40, 1.60),
    "openai/gpt-4.1-nano":                   (0.10, 0.40),
    "openai/gpt-5-mini":                     (0.25, 2.00),
    "openai/gpt-5":                          (1.25, 10.00),
    # Anthropic
    "anthropic/claude-opus-4.7":             (15.00, 75.00),
    "anthropic/claude-opus-4.6":             (15.00, 75.00),
    "anthropic/claude-sonnet-4.6":           (3.00, 15.00),
    "anthropic/claude-sonnet-4.5":           (3.00, 15.00),
    "anthropic/claude-haiku-4.5":            (1.00, 5.00),
    # Google
    "google/gemini-2.5-pro":                 (1.25, 10.00),
    "google/gemini-2.5-flash":               (0.30, 2.50),
    "google/gemini-2.5-flash-lite":          (0.10, 0.40),
    # Open weights / cheap providers
    "moonshotai/kimi-k2-thinking":           (0.60, 2.50),
    "moonshotai/kimi-k2.6":                  (0.74, 4.66),
    "qwen/qwen3.6-35b-a3b":                  (0.16, 0.97),
    "qwen/qwen3.6-flash":                    (0.25, 1.50),
    "qwen/qwen3.6-max-preview":              (1.04, 6.24),
    "deepseek/deepseek-v4-flash":            (0.14, 0.28),
    "deepseek/deepseek-v4-pro":              (0.43, 0.87),
    "meta-llama/llama-4-maverick":           (0.15, 0.60),
    "meta-llama/llama-4-scout":              (0.08, 0.30),
    "x-ai/grok-4.1-fast":                    (0.20, 0.50),
    "x-ai/grok-4-fast":                      (0.20, 0.50),
}


def list_models() -> dict[str, str]:
    """Return a copy of the registry, useful from notebooks for `pd.Series(...)`."""
    return dict(MODEL_REGISTRY)


def resolve_slug(name: str) -> str:
    """Return the OpenRouter slug for a role name, or pass `name` through if it
    looks like a raw slug (contains a '/')."""
    if "/" in name:
        return name
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    raise KeyError(
        f"Unknown model role {name!r}. "
        f"Available roles: {sorted(MODEL_REGISTRY)} or pass a raw slug like 'openai/gpt-4o-mini'."
    )


def get_llm(
    name: str,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    track_cost: bool = False,
    **kwargs: Any,
) -> ChatOpenAI:
    """Return a `ChatOpenAI` configured for OpenRouter.

    Args:
        name: A role (e.g. "cheap_workhorse") or a raw OpenRouter slug
            (e.g. "openai/gpt-4o-mini").
        temperature: Sampling temperature.
        max_tokens: Optional max output tokens.
        track_cost: If True, wraps the LLM in `CostTrackingLLM`. The wrapper
            captures token usage and an estimated $ cost on every call;
            access via `llm.last_usage`.
        **kwargs: Forwarded to `ChatOpenAI`.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    slug = resolve_slug(name)

    # Reasoning models often reject `temperature` — only set it if it makes sense.
    extra: dict[str, Any] = {}
    if not _is_reasoning(slug):
        extra["temperature"] = temperature
    if max_tokens is not None:
        extra["max_tokens"] = max_tokens

    llm = ChatOpenAI(
        model=slug,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/sinanuozdemir/advanced-agentic-ai-in-three-weeks",
            "X-Title": "advanced-agentic-ai-in-three-weeks",
        },
        **extra,
        **kwargs,
    )

    if track_cost:
        return CostTrackingLLM(llm, slug)  # type: ignore[return-value]
    return llm


def get_structured_llm(
    name: str,
    schema: type[BaseModel],
    *,
    method: str = "function_calling",
    max_tokens: int | None = None,
    temperature: float = 0.0,
    track_cost: bool = False,
    max_retries: int = 1,
    **kwargs: Any,
) -> "_StructuredRetryWrapper":
    """Return an LLM bound to a Pydantic schema, configured for OpenRouter.

    Why this exists (lessons learned the hard way):

    * `method="function_calling"` is the default because it is the most
      robust path on OpenRouter. The alternative `method="json_schema"` asks
      the provider to emit raw JSON; if any provider in the OpenRouter chain
      ignores the schema or truncates the output, you get a partial JSON blob
      that pydantic can't parse (`Invalid JSON: EOF while parsing ...`).
      Function calls are emitted in OpenAI's tool-call format, which is
      far less prone to silent truncation.
    * `max_tokens` defaults are deliberately generous: 2048 for chat models,
      4096 for reasoning models (which spend tokens on hidden reasoning
      before producing the structured output). You can override per call.
    * `max_retries=1` adds a single corrective retry on `ValidationError` —
      sometimes a model emits a field with the wrong type and a retry with
      the error message attached fixes it.
    """
    slug = resolve_slug(name)
    if max_tokens is None:
        max_tokens = 4096 if _is_reasoning(slug) else 2048

    llm = get_llm(
        name,
        temperature=temperature,
        max_tokens=max_tokens,
        track_cost=track_cost,
        **kwargs,
    )
    bound = llm.with_structured_output(schema, method=method)
    return _StructuredRetryWrapper(bound=bound, schema=schema, max_retries=max_retries)


def _is_reasoning(slug: str) -> bool:
    s = slug.lower()
    return (
        ":thinking" in s
        or "/o1" in s
        or "/o3" in s
        or "/o4" in s
        or "deepseek-r1" in s
        or "thinking" in s
    )


@dataclass
class Usage:
    """Per-call token + cost record."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    model: str = ""


@dataclass
class CostTrackingLLM:
    """Thin wrapper around `ChatOpenAI` that records token usage and an
    estimated $ cost for the most recent call.

    Behaves like the wrapped LLM for the methods our notebooks use
    (`invoke`, `with_structured_output`, `bind_tools`).
    """

    llm: ChatOpenAI
    slug: str
    usage_history: list[Usage] = field(default_factory=list)
    last_usage: Usage | None = None

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        import time
        t0 = time.time()
        result = self.llm.invoke(*args, **kwargs)
        elapsed = time.time() - t0
        self._record(result, elapsed)
        return result

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        # Default to function-calling: it's far more robust than json_schema
        # mode against truncation and across OpenRouter providers.
        kwargs.setdefault("method", "function_calling")
        bound = self.llm.with_structured_output(schema, **kwargs)
        return _StructuredCostWrapper(bound, self)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self.llm.bind_tools(tools, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.llm, attr)

    def _record(self, result: Any, elapsed: float) -> None:
        usage = _extract_usage(result, self.slug, elapsed)
        self.last_usage = usage
        self.usage_history.append(usage)


@dataclass
class _StructuredCostWrapper:
    bound: Any
    parent: "CostTrackingLLM"

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        import time
        t0 = time.time()
        result = self.bound.invoke(*args, **kwargs)
        elapsed = time.time() - t0
        self.parent._record(result, elapsed)
        return result

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.bound, attr)


@dataclass
class _StructuredRetryWrapper:
    """Adds a single corrective retry on `ValidationError` to a structured LLM.

    On failure, we re-invoke with an extra system note that quotes the
    pydantic error and asks the model to try again. This catches the
    common failure mode where a model emits e.g. a string where a list is
    expected, or omits a required field.
    """

    bound: Any
    schema: type[BaseModel]
    max_retries: int = 1

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        from langchain_core.messages import SystemMessage

        attempt = 0
        while True:
            problem: str | None = None
            try:
                result = self.bound.invoke(messages, *args, **kwargs)
                # `with_structured_output(method="function_calling")` returns
                # None when the model declines to make a tool call (e.g.
                # answers in plain text instead). Treat that as a retryable
                # failure so we can prod the model to actually emit the
                # structured output.
                if result is None:
                    problem = (
                        "Your previous response did NOT include a function "
                        f"call to emit a {self.schema.__name__}. You MUST "
                        "respond by calling the provided function with "
                        "arguments matching the schema. Do not respond with "
                        "plain text."
                    )
                else:
                    return result
            except ValidationError as exc:
                problem = (
                    "Your previous response failed schema validation with "
                    f"this error:\n{exc}\n\nRe-emit a response that conforms "
                    f"to the {self.schema.__name__} schema. Pay attention to "
                    "required fields and types."
                )
            # Non-validation, non-None errors bubble up via the bare except
            # of the next loop iteration -- but actually we don't catch
            # generic Exception, so they already bubble out of `try`.

            if attempt >= self.max_retries:
                raise RuntimeError(
                    f"Structured output failed after {attempt + 1} attempts "
                    f"for schema {self.schema.__name__}: {problem}"
                )

            correction = SystemMessage(content=problem or "Retry.")
            if isinstance(messages, list):
                messages = list(messages) + [correction]
            else:
                messages = [SystemMessage(content=str(messages)), correction]
            attempt += 1

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.bound, attr)


def _extract_usage(result: Any, slug: str, elapsed_s: float) -> Usage:
    """Pull token + cost data out of a LangChain AIMessage or structured-output result."""
    msg = result if hasattr(result, "usage_metadata") else getattr(result, "_message", None)
    in_t = out_t = total_t = 0
    cost = 0.0
    if msg is not None and getattr(msg, "usage_metadata", None):
        um = msg.usage_metadata
        in_t = int(um.get("input_tokens", 0))
        out_t = int(um.get("output_tokens", 0))
        total_t = int(um.get("total_tokens", in_t + out_t))
    if msg is not None:
        rmeta = getattr(msg, "response_metadata", {}) or {}
        # OpenRouter sometimes returns cost under `usage.cost` or
        # `usage.total_cost` — be defensive.
        usage_block = rmeta.get("usage") or {}
        for key in ("cost", "total_cost"):
            if key in usage_block:
                try:
                    cost = float(usage_block[key])
                    break
                except (TypeError, ValueError):
                    pass
    if cost == 0.0:
        cost = _estimate_cost(slug, in_t, out_t)
    return Usage(
        input_tokens=in_t,
        output_tokens=out_t,
        total_tokens=total_t or (in_t + out_t),
        cost_usd=cost,
        latency_s=elapsed_s,
        model=slug,
    )


def estimate_cost(slug: str, in_tokens: int, out_tokens: int) -> float:
    """Estimate per-call USD cost from token counts and the model's
    `_PRICE_PER_M_TOKENS` entry. Returns 0.0 for unknown slugs."""
    in_price, out_price = _PRICE_PER_M_TOKENS.get(slug, (0.0, 0.0))
    return (in_tokens / 1_000_000) * in_price + (out_tokens / 1_000_000) * out_price


# Backwards-compat alias for any internal callers.
_estimate_cost = estimate_cost


__all__ = [
    "MODEL_REGISTRY",
    "OPENROUTER_BASE_URL",
    "Usage",
    "CostTrackingLLM",
    "get_llm",
    "get_structured_llm",
    "estimate_cost",
    "list_models",
    "resolve_slug",
]
