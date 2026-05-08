"""Stable imports for Week 2 code and notebooks.

- **OpenRouter LLM helpers** live in `shared.openrouter_llm` (this package) so
  apps and libraries never depend on `notebooks/week1/` at import time.
- **Eval, judging, retrieval** helpers still live under `notebooks/week1/` for
  course notebooks; they are loaded lazily on first use so optional deps
  (e.g. pandas) are not required to import `shared.make_checkpointer` or
  `shared.get_llm`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

from .checkpointer import make_checkpointer
from .openrouter_llm import (
    MODEL_REGISTRY,
    CostTrackingLLM,
    estimate_cost,
    get_llm,
    get_structured_llm,
)

# Names delegated to `notebooks/week1/` (judges, eval_harness, corpus, retrievers).
_WEEK1_LAZY: Final[frozenset[str]] = frozenset(
    {
        "judge_with_rubric",
        "grade_chunk",
        "RunResult",
        "Variant",
        "run_suite",
        "plot_overall_bar",
        "plot_quality_vs_cost",
        "plot_pareto_frontier",
        "plot_by_difficulty",
        "variant_summary",
        "load_chroma",
        "load_bm25",
        "load_gold_set",
        "HybridRetriever",
        "CrossEncoderReranker",
    }
)

_week1_exports: dict[str, Any] | None = None


def _ensure_week1_on_path() -> None:
    """Prepend ``…/notebooks/week1`` to ``sys.path`` if that tree exists."""
    if getattr(_ensure_week1_on_path, "_done", False):  # type: ignore[misc]
        return
    here = Path(__file__).resolve().parent
    for base in here.parents:
        cand = base / "notebooks" / "week1"
        if (cand / "judges.py").is_file():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            setattr(_ensure_week1_on_path, "_done", True)
            return
    setattr(_ensure_week1_on_path, "_done", True)


def _load_week1_exports() -> dict[str, Any]:
    global _week1_exports
    if _week1_exports is not None:
        return _week1_exports
    _ensure_week1_on_path()
    from judges import judge_with_rubric, grade_chunk  # noqa: PLC0415
    from corpus import load_bm25, load_chroma, load_gold_set  # noqa: PLC0415
    from eval_harness import (  # noqa: PLC0415
        RunResult,
        Variant,
        plot_by_difficulty,
        plot_overall_bar,
        plot_pareto_frontier,
        plot_quality_vs_cost,
        run_suite,
        variant_summary,
    )
    from retrievers import CrossEncoderReranker, HybridRetriever  # noqa: PLC0415

    _week1_exports = {
        "judge_with_rubric": judge_with_rubric,
        "grade_chunk": grade_chunk,
        "RunResult": RunResult,
        "Variant": Variant,
        "run_suite": run_suite,
        "plot_overall_bar": plot_overall_bar,
        "plot_quality_vs_cost": plot_quality_vs_cost,
        "plot_pareto_frontier": plot_pareto_frontier,
        "plot_by_difficulty": plot_by_difficulty,
        "variant_summary": variant_summary,
        "load_chroma": load_chroma,
        "load_bm25": load_bm25,
        "load_gold_set": load_gold_set,
        "HybridRetriever": HybridRetriever,
        "CrossEncoderReranker": CrossEncoderReranker,
    }
    return _week1_exports


def __getattr__(name: str) -> Any:
    if name in _WEEK1_LAZY:
        try:
            return _load_week1_exports()[name]
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"shared.{name} requires the course tree at notebooks/week1 "
                f"(and its dependencies, e.g. pandas). Original error: {exc}"
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MODEL_REGISTRY",
    "CostTrackingLLM",
    "estimate_cost",
    "get_llm",
    "get_structured_llm",
    "judge_with_rubric",
    "grade_chunk",
    "RunResult",
    "Variant",
    "run_suite",
    "plot_overall_bar",
    "plot_quality_vs_cost",
    "plot_pareto_frontier",
    "plot_by_difficulty",
    "variant_summary",
    "load_chroma",
    "load_bm25",
    "load_gold_set",
    "HybridRetriever",
    "CrossEncoderReranker",
    "make_checkpointer",
]
