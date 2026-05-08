"""Drop-in conversation-compression middlewares for ``langchain.agents.create_agent``.

Five strategies that mirror the patterns taught in Week 2 Segment 4 and Week 1
Notebook 4. All subclass :class:`langchain.agents.middleware.AgentMiddleware`,
implement ``before_model``, and emit the same ``RemoveMessage(REMOVE_ALL_MESSAGES)
+ HumanMessage(summary) + preserved_tail`` shape as the built-in
``SummarizationMiddleware`` so they're swappable.
"""
from __future__ import annotations

from .compression import (
    HierarchicalCompressionMiddleware,
    MapReduceSummarizationMiddleware,
    RecursiveSummaryMiddleware,
    RefineSummarizationMiddleware,
    RulesFirstSummaryMiddleware,
)

__all__ = [
    "HierarchicalCompressionMiddleware",
    "MapReduceSummarizationMiddleware",
    "RecursiveSummaryMiddleware",
    "RefineSummarizationMiddleware",
    "RulesFirstSummaryMiddleware",
]
