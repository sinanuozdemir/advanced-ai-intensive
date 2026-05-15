"""Ollama-backed LLM factory.

Tiny sibling of ``shared.openrouter_llm``. Activated when a slug starts with
``ollama/`` (e.g. ``ollama/llama3.2``, ``ollama/qwen2.5-coder``); the suffix
after the prefix is passed straight to ``ChatOllama`` as ``model``.

Why a separate factory:

* OpenRouter and Ollama have different SDK shapes (``ChatOpenAI`` vs
  ``ChatOllama``). Trying to coerce one client to talk to the other inevitably
  breaks tool calls or streaming.
* Cost is always **$0** for Ollama — it's local inference — so the
  ``CostTrackingLLM`` wrapper still works but always records ``cost_usd=0.0``.
* The endpoint defaults to ``OLLAMA_HOST`` (``http://localhost:11434``) so
  users running Ollama natively don't need any extra config.

Structured output: we use ``ChatOllama.with_structured_output(method="function_calling")``
on recent versions of Ollama that support tool calls. Smaller models can flake
on validation; the existing ``_StructuredRetryWrapper`` in ``openrouter_llm``
catches that and retries with the error attached.
"""

from __future__ import annotations

import os
from typing import Any

OLLAMA_PREFIX = "ollama/"


def is_ollama_slug(name: str) -> bool:
    """True if ``name`` is an ollama/* slug (case-sensitive prefix)."""
    return name.startswith(OLLAMA_PREFIX)


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _strip_prefix(slug: str) -> str:
    return slug[len(OLLAMA_PREFIX):] if slug.startswith(OLLAMA_PREFIX) else slug


def get_ollama_llm(
    name: str,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> Any:
    """Return a ``ChatOllama`` for ``ollama/<model>`` slugs.

    Args:
        name: The full slug (``ollama/llama3.2``) or just the model
            (``llama3.2``). Both forms are accepted.
        temperature: Sampling temperature.
        max_tokens: Forwarded as ``num_predict``.
        **kwargs: Forwarded to ``ChatOllama``.
    """
    try:
        from langchain_ollama import ChatOllama  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Ollama support requires the 'langchain-ollama' package. "
            "Install with: pip install langchain-ollama"
        ) from exc

    model = _strip_prefix(name)
    extra: dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        extra["num_predict"] = max_tokens
    return ChatOllama(model=model, base_url=_ollama_host(), **extra, **kwargs)


__all__ = ["OLLAMA_PREFIX", "get_ollama_llm", "is_ollama_slug"]
