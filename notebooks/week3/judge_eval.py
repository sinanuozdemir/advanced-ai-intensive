"""Meta-evaluation of the rubric judge.

Reads `data/judge_gold_set.jsonl`, scores each item with `judge_with_rubric`,
and reports where the judge agrees / disagrees with our subjective expected
range. The point isn't to "fix" the judge — it's to expose its biases so we
can talk about them in the eval-deep-dive lecture.

Usage:
    cd notebooks/week3
    python judge_eval.py
    # writes data/judge_eval_results.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "notebooks" / "week1"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from judges import judge_with_rubric  # noqa: E402

GOLD = _HERE / "data" / "judge_gold_set.jsonl"
OUT = _HERE / "data" / "judge_eval_results.csv"


def _load() -> list[dict]:
    return [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]


def _verdict(score: float, lo: float, hi: float) -> str:
    if score < lo:
        return "TOO LOW"
    if score > hi:
        return "TOO HIGH"
    return "ok"


def main() -> None:
    items = _load()
    rows = []
    for it in items:
        rb = judge_with_rubric(
            question=it["question"],
            answer=it["answer"],
            reference=it.get("reference") or "",
            evidence=it.get("evidence"),
        )
        verdict = _verdict(rb.overall, it["expected_overall_min"], it["expected_overall_max"])
        rows.append({
            "id": it["id"],
            "probe": it["probe"],
            "expected_lo": it["expected_overall_min"],
            "expected_hi": it["expected_overall_max"],
            "judge_overall": rb.overall,
            "judge_correctness": rb.correctness,
            "judge_faithfulness": rb.faithfulness,
            "verdict": verdict,
            "rationale": it["rationale"],
        })
        flag = "  " if verdict == "ok" else "!!"
        print(f"{flag} [{it['probe']:<28s}] {it['id']:<28s} "
              f"expected {it['expected_overall_min']}-{it['expected_overall_max']}  "
              f"judge={rb.overall:.2f}  -> {verdict}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n=== Summary by probe ===")
    summary = (df.assign(off=(df["verdict"] != "ok").astype(int))
                 .groupby("probe")
                 .agg(n=("id", "count"),
                      n_off=("off", "sum"),
                      mean_overall=("judge_overall", "mean"))
                 .round(2))
    print(summary.to_string())

    n_off = int((df["verdict"] != "ok").sum())
    print(f"\n{n_off}/{len(df)} items outside expected range. "
          f"Wrote {OUT.relative_to(_REPO_ROOT)}.")


if __name__ == "__main__":
    main()
