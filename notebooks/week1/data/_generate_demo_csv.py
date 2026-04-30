"""Generate plausible (synthetic) `results_full.csv` and `results_demo.csv` so
the analysis cells in notebook 5 render before students run the live sweep.

Run from `notebooks/week1/`:

    python data/_generate_demo_csv.py

This is a teaching artifact — the numbers are made-up but the SHAPE of the
data matches what the live sweep produces:

- `results_full.csv`  ~ what `DEMO_MODE = False` writes (4 variants x up to 3
  models x 30 questions x 3 runs). Uses raw OpenRouter slugs in the `model`
  column because that's how the homework sweep is configured.
- `results_demo.csv`  ~ what `DEMO_MODE = True`  writes (4 variants x 1 model
  x 5 questions x 1 run = 20 rows). Uses `openai/gpt-5.4-nano` in
  the `model` column to match what `run_suite` actually writes in demo mode.

Re-running notebook 5 will overwrite whichever cache matches the current mode
with real numbers.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
GOLD_PATH = HERE / "gold_set.jsonl"
FULL_OUT_PATH = HERE / "results_full.csv"
DEMO_OUT_PATH = HERE / "results_demo.csv"


VARIANT_BASELINES = {
    # (mean_overall_easy, mean_overall_hard, mean_cost, mean_latency, mean_n_retrievals)
    "workflow":        {"easy": 4.0, "medium": 4.2, "hard": 4.3, "cost_mult": 4.0, "latency_mult": 4.0, "retr": 4},
    "agent_vanilla":   {"easy": 4.1, "medium": 3.4, "hard": 2.6, "cost_mult": 1.0, "latency_mult": 1.0, "retr": 1},
    "agent_reasoning": {"easy": 4.2, "medium": 3.9, "hard": 3.7, "cost_mult": 6.0, "latency_mult": 3.0, "retr": 2},
    "agent_cot":       {"easy": 4.0, "medium": 3.7, "hard": 3.4, "cost_mult": 1.5, "latency_mult": 1.6, "retr": 3},
}

# Approx $/call for a single end-to-end RAG response. Picked to match the
# rough shape of the OpenRouter pricing differences in April 2026.
MODEL_BASE_COST = {
    "openai/gpt-5.4-nano":               0.0004,
    "openai/gpt-5.5":                    0.012,
    "anthropic/claude-opus-4.7":         0.025,
    "openai/o4-mini":                    0.003,
    "moonshotai/kimi-k2-thinking":       0.0015,
    "qwen/qwen3.6-35b-a3b":              0.0003,
    "x-ai/grok-4.1-fast":                0.0009,
}


FALLBACK_QUESTIONS = [
    {"id": f"cross_{i:02d}", "question": f"cross-source #{i}", "reference_answer": "ref",
     "difficulty": "hard", "hop_count": 2}
    for i in range(1, 11)
] + [
    {"id": f"single_{i:02d}", "question": f"single-source #{i}", "reference_answer": "ref",
     "difficulty": "easy" if i % 2 else "medium", "hop_count": 1}
    for i in range(1, 11)
] + [
    {"id": f"hotpot_{i:02d}", "question": f"hotpot #{i}", "reference_answer": "ref",
     "difficulty": "medium" if i < 5 else "hard", "hop_count": 2}
    for i in range(10)
]


def _model_quality_bonus(model: str, diff: str) -> float:
    """Per-model adjustments layered on top of the variant baseline."""
    bonus = 0.0
    if "claude-opus" in model or "gpt-5.5" in model:
        bonus += 0.15
    if "o4-mini" in model and diff == "hard":
        bonus += 0.2
    if "kimi" in model and "thinking" in model and diff == "hard":
        bonus += 0.15
    if "qwen" in model or "nano" in model:
        bonus -= 0.15
    return bonus


def _build_rows(
    questions: list[dict[str, Any]],
    variant_models: dict[str, list[str]],
    n_runs: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for variant, base in VARIANT_BASELINES.items():
        if variant not in variant_models:
            continue
        for model in variant_models[variant]:
            model_factor = 1.0 + (0.0 if "nano" in model or "qwen" in model else 0.2)
            for q in questions:
                diff = q["difficulty"]
                hop = q["hop_count"]
                base_quality = base[diff] + _model_quality_bonus(model, diff)
                # multi-hop bias toward workflow
                if variant == "workflow" and hop >= 2:
                    base_quality += 0.2
                if variant == "agent_vanilla" and hop >= 2:
                    base_quality -= 0.4
                for run_idx in range(n_runs):
                    quality = max(0.0, min(5.0, base_quality + rng.gauss(0, 0.3)))
                    correctness = max(0, min(5, round(quality + rng.gauss(0, 0.3))))
                    faithfulness = max(0, min(5, round(quality + rng.gauss(0, 0.3))))
                    completeness = max(0, min(5, round(quality + rng.gauss(0, 0.4))))
                    conciseness = max(0, min(5, round(4.0 + rng.gauss(0, 0.5))))

                    in_t = int(rng.randint(800, 2400) * base["retr"] * (1.5 if hop >= 2 else 1.0))
                    out_t = int(rng.randint(150, 600))
                    base_cost = MODEL_BASE_COST.get(model, 0.001)
                    cost = base_cost * base["cost_mult"] * model_factor * rng.uniform(0.7, 1.3)
                    cost = round(cost, 6)
                    lat = round(rng.uniform(2.5, 9.0) * base["latency_mult"] / 4, 2)

                    rows.append({
                        "variant": variant,
                        "model": model,
                        "query_id": q["id"],
                        "run_idx": run_idx,
                        "question": q["question"][:80] + "...",
                        "answer": "(synthetic placeholder)",
                        "latency_s": lat,
                        "input_tokens": in_t,
                        "output_tokens": out_t,
                        "cost_usd": cost,
                        "n_retrievals": int(base["retr"] + rng.gauss(0, 1)),
                        "n_tool_calls": int(base["retr"] + rng.gauss(0, 1)),
                        "error": "",
                        "faithfulness": faithfulness,
                        "completeness": completeness,
                        "correctness": correctness,
                        "conciseness": conciseness,
                        "overall": round(quality, 2),
                        "judge_notes": "(synthetic)",
                        "reference_answer": q["reference_answer"][:80] + "...",
                        "difficulty": diff,
                        "hop_count": hop,
                    })
    return rows


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {out_path}")


def _load_questions() -> list[dict[str, Any]]:
    if GOLD_PATH.exists():
        return [json.loads(line) for line in GOLD_PATH.read_text().splitlines() if line.strip()]
    print(f"({GOLD_PATH} not found; using fallback question stubs for shape only)")
    return FALLBACK_QUESTIONS


def _demo_subset(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mirror the deterministic 5-question mix from notebook 5's demo cell:
    2 cross-source, 1 single-source, 2 hotpot."""
    cross   = [q for q in questions if q["id"].startswith("cross_")][:2]
    single  = [q for q in questions if q["id"].startswith("single_")][:1]
    hotpot  = [q for q in questions if q["id"].startswith("hotpot_")][:2]
    return cross + single + hotpot


def main() -> None:
    questions = _load_questions()

    full_variant_models = {
        "workflow":        ["openai/gpt-5.4-nano", "anthropic/claude-opus-4.7", "qwen/qwen3.6-35b-a3b"],
        "agent_vanilla":   ["openai/gpt-5.4-nano", "anthropic/claude-opus-4.7", "qwen/qwen3.6-35b-a3b"],
        "agent_reasoning": ["openai/o4-mini", "moonshotai/kimi-k2-thinking"],
        "agent_cot":       ["openai/gpt-5.4-nano", "anthropic/claude-opus-4.7", "qwen/qwen3.6-35b-a3b"],
    }
    _write_csv(_build_rows(questions, full_variant_models, n_runs=3, seed=0), FULL_OUT_PATH)

    # Demo mode mirrors notebook 5's `if DEMO_MODE:` branch: every variant
    # runs against the same single model (`openai/gpt-5.4-nano`), one run each,
    # on the 5-question mix.
    demo_variant_models = {v: ["openai/gpt-5.4-nano"] for v in VARIANT_BASELINES}
    _write_csv(
        _build_rows(_demo_subset(questions), demo_variant_models, n_runs=1, seed=1),
        DEMO_OUT_PATH,
    )


if __name__ == "__main__":
    main()
