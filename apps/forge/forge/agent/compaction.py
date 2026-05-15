"""Build the configured compaction middleware.

Default Forge config is ``strategy="refine"`` with
``summarizer="anthropic/claude-opus-4.7"`` — operationalizes the W2 N4 finding
("summarizer model dominates rule-preservation"). The strategy and summarizer
are both swappable in ``.forge/config.toml``.
"""
from __future__ import annotations

from typing import Any

from middleware.compression import (
    HierarchicalCompressionMiddleware,
    MapReduceSummarizationMiddleware,
    RecursiveSummaryMiddleware,
    RefineSummarizationMiddleware,
    RulesFirstSummaryMiddleware,
)

from ..config import CompactionConfig


_STRATEGY_CLASSES = {
    "refine": RefineSummarizationMiddleware,
    "rules_first": RulesFirstSummaryMiddleware,
    "map_reduce": MapReduceSummarizationMiddleware,
    "recursive": RecursiveSummaryMiddleware,
    "hierarchical": HierarchicalCompressionMiddleware,
}


def build_compaction_with_summarizer(cfg: CompactionConfig, summarizer_slug: str) -> list[Any]:
    """Build the compaction middleware bound to ``summarizer_slug``.

    Returns an empty list when ``strategy == "none"`` so eval / notebooks
    can disable compaction cleanly. The summarizer LLM is resolved through
    ``shared.get_llm`` (OpenRouter-aware) so non-LangChain-native slugs like
    ``anthropic/claude-opus-4.7`` work."""
    if cfg.strategy == "none":
        return []
    from shared import get_llm
    summarizer_llm = get_llm(summarizer_slug)
    trigger = (cfg.trigger_kind, cfg.trigger_threshold)
    cls = _STRATEGY_CLASSES.get(cfg.strategy)
    if cls is None:
        # lc_sliding_window — fall through to the built-in if available.
        try:
            from langchain.agents.middleware import SummarizationMiddleware  # type: ignore
            return [SummarizationMiddleware()]
        except Exception:  # noqa: BLE001
            return []
    return [cls(model=summarizer_llm, keep_last=cfg.keep_last, trigger=trigger)]


__all__ = ["build_compaction_with_summarizer"]
