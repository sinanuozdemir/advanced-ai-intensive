"""Stable imports for course code and notebooks.

- **OpenRouter LLM helpers** live in `shared.openrouter_llm` (this package) so
  apps and libraries never depend on `notebooks/` at import time.
- **Eval, judging, retrieval** helpers still live under `notebooks/` for
  course notebooks; they are loaded lazily on first use so optional deps
  (e.g. pandas) are not required to import `shared.make_checkpointer` or
  `shared.get_llm`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

from .checkpointer import make_checkpointer
from .ollama_llm import is_ollama_slug
from .openrouter_llm import (
    MODEL_REGISTRY,
    CostTrackingLLM,
    estimate_cost,
    get_llm,
    get_structured_llm,
)

# Names delegated to `notebooks/` (judges, eval_harness, corpus, retrievers).
_NOTEBOOK_LAZY: Final[frozenset[str]] = frozenset(
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

_notebook_exports: dict[str, Any] | None = None


def _ensure_notebooks_on_path() -> None:
    """Prepend ``…/notebooks`` to ``sys.path`` if that tree exists."""
    if getattr(_ensure_notebooks_on_path, "_done", False):  # type: ignore[misc]
        return
    here = Path(__file__).resolve().parent
    for base in here.parents:
        cand = base / "notebooks"
        if (cand / "judges.py").is_file():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            setattr(_ensure_notebooks_on_path, "_done", True)
            return
    setattr(_ensure_notebooks_on_path, "_done", True)


def _load_notebook_exports() -> dict[str, Any]:
    global _notebook_exports
    if _notebook_exports is not None:
        return _notebook_exports
    _ensure_notebooks_on_path()
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

    _notebook_exports = {
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
    return _notebook_exports


def __getattr__(name: str) -> Any:
    if name in _NOTEBOOK_LAZY:
        try:
            return _load_notebook_exports()[name]
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"shared.{name} requires the course tree at notebooks/ "
                f"(and its dependencies, e.g. pandas). Original error: {exc}"
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MODEL_REGISTRY",
    "CostTrackingLLM",
    "estimate_cost",
    "get_llm",
    "get_structured_llm",
    "is_ollama_slug",
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
