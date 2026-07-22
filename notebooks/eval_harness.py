"""Multi-variant x multi-model evaluation runner used by notebook 5.

A `Variant` is any callable ``(question: str, model_slug: str) -> RunResult``
that runs an end-to-end answer given a model. The harness:
  - takes a list of variants and a list of model slugs,
  - runs the cross-product against the gold set ``n_runs`` times,
  - judges each answer with the rubric in ``judges.py``,
  - persists results to a CSV that can be re-loaded without re-running,
  - produces standard plots (bar, scatter, Pareto frontier, by difficulty).

`DEMO_MODE` (controlled by `run_suite(..., demo_mode=True)`) caps the work to
~5 questions x 1 model x 1 run so the bake-off finishes in ~2 minutes during
live class.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from corpus import GoldQuestion
from judges import RubricResult, judge_with_rubric


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """One end-to-end run: variant x model x query x run_idx."""

    variant: str
    model: str
    query_id: str
    run_idx: int
    question: str
    answer: str
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    n_retrievals: int = 0
    n_tool_calls: int = 0
    trajectory: list[dict] = field(default_factory=list)
    # Raw text of the chunks shown to the synthesis LLM. Used to score
    # faithfulness *against the retrieval* rather than against the reference
    # answer alone — without this, models that fabricate from pretraining
    # trivially get faithfulness=5. Empty list means "no evidence captured;
    # fall back to reference-only faithfulness."
    evidence_texts: list[str] = field(default_factory=list)
    error: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "model": self.model,
            "query_id": self.query_id,
            "run_idx": self.run_idx,
            "question": self.question,
            "answer": self.answer,
            "latency_s": self.latency_s,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "n_retrievals": self.n_retrievals,
            "n_tool_calls": self.n_tool_calls,
            "error": self.error or "",
        }


Variant = Callable[[str, str], RunResult]


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_suite(
    variants: dict[str, Variant],
    models: list[str],
    gold_set: list[GoldQuestion],
    *,
    n_runs: int = 1,
    demo_mode: bool = False,
    judge_llm: Any | None = None,
    cache_path: Path | str | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Run every (variant, model) pair against ``gold_set`` ``n_runs`` times,
    judge with the rubric, and return a tidy DataFrame.

    Args:
        variants: ``{variant_name: callable}``. Each callable takes
            ``(question, model_slug)`` and returns a `RunResult`.
        models: List of OpenRouter slugs (or role names) to sweep across.
        gold_set: Loaded via ``corpus.load_gold_set()``.
        n_runs: How many times to repeat each (variant, model, question).
        demo_mode: If True, caps to 5 questions x 1 model x 1 run for live
            class. Models[0] is used.
        judge_llm: Optional pre-built judge LLM. Defaults to
            ``anthropic/claude-opus-4.7``.
        cache_path: If given, write rows to CSV after each run for resumability.

    Returns:
        Tidy DataFrame with one row per (variant, model, query, run_idx) plus
        rubric scores columns.
    """
    if demo_mode:
        gold_set = gold_set[:5]
        models = models[:1]
        n_runs = 1

    rows: list[dict[str, Any]] = []
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(variants) * len(models) * len(gold_set) * n_runs
    done = 0
    started = time.time()

    for variant_name, variant_fn in variants.items():
        for model in models:
            for q in gold_set:
                for run_idx in range(n_runs):
                    done += 1
                    if progress:
                        elapsed = time.time() - started
                        rate = elapsed / max(done, 1)
                        eta = rate * (total - done)
                        print(
                            f"[{done:>3}/{total}] {variant_name:>14} | "
                            f"{model:<45} | {q.id:<10} | run {run_idx+1}/{n_runs} | "
                            f"eta {eta:5.0f}s",
                            flush=True,
                        )
                    rr = _safe_run(variant_fn, q.question, model, variant_name, q.id, run_idx)
                    rubric = _safe_judge(q, rr, judge_llm)
                    row = rr.to_row()
                    row.update(_rubric_to_row(rubric))
                    row["reference_answer"] = q.reference_answer
                    row["difficulty"] = q.difficulty
                    row["hop_count"] = q.hop_count
                    rows.append(row)
                    if cache_path is not None:
                        pd.DataFrame(rows).to_csv(cache_path, index=False)

    df = pd.DataFrame(rows)
    if cache_path is not None and not df.empty:
        df.to_csv(cache_path, index=False)
    return df


def _safe_run(
    fn: Variant, question: str, model: str, variant: str, qid: str, run_idx: int,
) -> RunResult:
    t0 = time.time()
    try:
        rr = fn(question, model)
        rr.query_id = qid
        rr.run_idx = run_idx
        return rr
    except Exception as exc:  # noqa: BLE001
        return RunResult(
            variant=variant,
            model=model,
            query_id=qid,
            run_idx=run_idx,
            question=question,
            answer="",
            latency_s=time.time() - t0,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}",
        )


def _safe_judge(q: GoldQuestion, rr: RunResult, judge_llm: Any | None) -> RubricResult | None:
    if rr.error or not rr.answer.strip():
        return None
    try:
        return judge_with_rubric(
            q.question,
            rr.answer,
            q.reference_answer,
            evidence=rr.evidence_texts or None,
            judge_llm=judge_llm,
        )
    except Exception as exc:  # noqa: BLE001
        rr.error = (rr.error or "") + f"\nJUDGE_ERR: {exc}"
        return None


def _rubric_to_row(r: RubricResult | None) -> dict[str, Any]:
    if r is None:
        return {
            "faithfulness": None,
            "completeness": None,
            "correctness": None,
            "conciseness": None,
            "overall": None,
            "judge_notes": "",
        }
    return {
        "faithfulness": r.faithfulness,
        "completeness": r.completeness,
        "correctness": r.correctness,
        "conciseness": r.conciseness,
        "overall": r.overall,
        "judge_notes": r.notes,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_overall_bar(df: pd.DataFrame, ax=None):
    """Mean rubric `overall` per variant, error bars across models + runs."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    grouped = df.dropna(subset=["overall"]).groupby("variant")["overall"]
    means = grouped.mean()
    stds = grouped.std().fillna(0.0)
    ax.bar(means.index, means.values, yerr=stds.values, capsize=4)
    ax.set_ylabel("Rubric overall (0-5)")
    ax.set_title("Mean answer quality by variant (error bars = std across models x runs)")
    ax.set_ylim(0, 5)
    return ax


def plot_quality_vs_cost(df: pd.DataFrame, ax=None):
    """Scatter: x=cost_usd, y=overall, colored by model, marker by variant."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.5))
    sub = df.dropna(subset=["overall", "cost_usd"]).copy()
    if sub.empty:
        ax.text(0.5, 0.5, "no scored rows", ha="center", va="center", transform=ax.transAxes)
        return ax

    agg = (
        sub.groupby(["variant", "model"])
        .agg(overall=("overall", "mean"), cost_usd=("cost_usd", "mean"))
        .reset_index()
    )
    markers = {"workflow": "o", "agent_vanilla": "s", "agent_reasoning": "^", "agent_cot": "D"}
    cmap = plt.get_cmap("tab10")
    models = sorted(agg["model"].unique())
    color_map = {m: cmap(i % 10) for i, m in enumerate(models)}
    for _, row in agg.iterrows():
        ax.scatter(
            row["cost_usd"],
            row["overall"],
            marker=markers.get(row["variant"], "o"),
            color=color_map[row["model"]],
            s=120,
            edgecolors="black",
            linewidths=0.6,
            label=f"{row['variant']} | {row['model']}",
        )
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel("Mean cost per query (USD, log)")
    ax.set_ylabel("Mean rubric overall (0-5)")
    ax.set_title("Quality vs cost — colored by model, shape by variant")
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return ax


def plot_pareto_frontier(df: pd.DataFrame, ax=None):
    """For each variant, show only the Pareto-best (cost, quality) (variant,model) points."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.5))
    sub = df.dropna(subset=["overall", "cost_usd"]).copy()
    if sub.empty:
        ax.text(0.5, 0.5, "no scored rows", ha="center", va="center", transform=ax.transAxes)
        return ax
    agg = (
        sub.groupby(["variant", "model"])
        .agg(overall=("overall", "mean"), cost_usd=("cost_usd", "mean"))
        .reset_index()
    )
    cmap = plt.get_cmap("tab10")
    variants = sorted(agg["variant"].unique())
    for i, v in enumerate(variants):
        pts = agg[agg["variant"] == v].sort_values("cost_usd")
        # Pareto: as cost rises, only keep points whose quality strictly improves
        best = -1.0
        keep = []
        for _, r in pts.iterrows():
            if r["overall"] > best:
                keep.append(r)
                best = r["overall"]
        keep_df = pd.DataFrame(keep)
        ax.plot(
            keep_df["cost_usd"],
            keep_df["overall"],
            marker="o",
            color=cmap(i % 10),
            label=v,
            linewidth=2,
        )
        for _, r in keep_df.iterrows():
            ax.annotate(
                r["model"].split("/")[-1],
                (r["cost_usd"], r["overall"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel("Mean cost per query (USD, log)")
    ax.set_ylabel("Mean rubric overall (0-5)")
    ax.set_title("Pareto frontier per variant")
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_by_difficulty(df: pd.DataFrame, ax=None):
    """Mean overall per variant, faceted by `difficulty`."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    sub = df.dropna(subset=["overall"]).copy()
    if sub.empty:
        ax.text(0.5, 0.5, "no scored rows", ha="center", va="center", transform=ax.transAxes)
        return ax
    pivot = sub.pivot_table(
        index="variant", columns="difficulty", values="overall", aggfunc="mean"
    )
    diffs = ["easy", "medium", "hard"]
    cols = [c for c in diffs if c in pivot.columns]
    pivot = pivot[cols]
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(cols), 1)
    for i, c in enumerate(cols):
        ax.bar(x + i * width, pivot[c].values, width=width, label=c)
    ax.set_xticks(x + width * (len(cols) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=20)
    ax.set_ylabel("Mean rubric overall (0-5)")
    ax.set_title("Quality by query difficulty")
    ax.set_ylim(0, 5)
    ax.legend(title="difficulty")
    return ax


def variant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compact per-variant summary table."""
    sub = df.dropna(subset=["overall"]).copy()
    return (
        sub.groupby("variant")
        .agg(
            mean_overall=("overall", "mean"),
            mean_correctness=("correctness", "mean"),
            mean_faithfulness=("faithfulness", "mean"),
            mean_cost_usd=("cost_usd", "mean"),
            mean_latency_s=("latency_s", "mean"),
            mean_n_retrievals=("n_retrievals", "mean"),
            n_runs=("overall", "size"),
            n_errors=("error", lambda s: (s.fillna("").astype(str).str.strip() != "").sum()),
        )
        .round(3)
        .sort_values("mean_overall", ascending=False)
    )


__all__ = [
    "RunResult",
    "Variant",
    "run_suite",
    "plot_overall_bar",
    "plot_quality_vs_cost",
    "plot_pareto_frontier",
    "plot_by_difficulty",
    "variant_summary",
]
