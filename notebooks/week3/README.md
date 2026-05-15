# Week 3 — Eval deep dive

The bake-off in week 2 surfaced a subtle finding: the rubric judge has its own biases. It punished honest "I couldn't search" answers ~2 points lower than honest "I searched and found nothing" answers, even though both are epistemically equivalent. It also dinged verbose, well-cited answers in favor of terser ones.

This week treats **the judge itself as a system to evaluate**.

## Artifacts

- `data/judge_gold_set.jsonl` — 32 hand-crafted (question, answer) pairs with expected score ranges and `group` ids (so paired items stay together in train/test splits). Probe categories:
  - `ground_truth_sanity` (6) — correct, confidently wrong, partial, empty.
  - `honesty_vs_fabrication` (6) — same question, three answers: searched-and-refused, fabricated-but-plausible, lazy-IDK. Two question groups.
  - `format_vs_verbosity` (3) — same correct content in tight prose, markdown-heavy, padded conversational.
  - `citation_presence` (4) — correct content with vs. without quoted sources, two topics.
  - `refusal_mode_equivalence` (4) — "searched, nothing found" vs. "couldn't search" — both honest, two topics.
  - `tone_vs_correctness` (4) — hedging-but-correct vs. confidently wrong, two topics.
  - `length_bias` (3) — same correct content at ~50, ~110, ~360 words.
  - `specificity` (2) — vague-but-not-wrong vs. precise-with-mechanisms.
- `judge_eval.py` — runs the judge against every gold item and reports which scores fall outside the expected range.
- `1_judge_meta_eval.ipynb` — sweeps several judge LLMs, builds agreement and signed-bias heatmaps.
- `2_dspy_judge_optimization.ipynb` — wraps the judge as a `dspy.Module`, optimizes few-shot demos with `BootstrapFewShotWithRandomSearch` against a probe-weighted distance-to-band metric, compares stock vs. optimized on a held-out test split.

## How to read the output

For each item the runner prints:

```
   [probe                       ] id                            expected 4.3-5.0  judge=4.50  -> ok
!! [refusal_mode_equivalence    ] E2_could_not_search           expected 3.0-4.5  judge=2.20  -> TOO LOW
```

`!!` flags are biases worth discussing in lecture. The `Summary by probe` block at the end shows how often each probe trips the judge.

## What to look for

These are the predictions worth testing in class:

1. **`A2_confidently_wrong` will not score near zero** — well-written nonsense gets partial credit. (Fluency bias.)
2. **`B2_fabricated_plausible` will outscore `B1_honest_no_corpus`** — the judge rewards confident answers over honest refusals when the answer "sounds right." (Confidence bias.)
3. **`C2_verbose_correct` will outscore `C1_tight_correct` by 0.3+** — markdown formatting reads as effort. (Format bias.)
4. **`E2_could_not_search` will score 1.5+ below `E1_searched_found_nothing`** — same epistemic position, very different score. (Search-evidence bias.)
5. **`F2_confident_wrong` will not score floor** — confidence + fluency bypasses correctness checks.

If 3 of those 5 hold, you have a slide-worthy lecture: *"your judge has opinions you didn't ask for."*

## Running

```bash
cd notebooks/week3
python judge_eval.py
```

Writes `data/judge_eval_results.csv`. Costs ~16 LLM calls (~$0.10 with Claude Opus 4.7 as the default judge).

## Extending

- Add probes for **length bias** (same answer at 50 / 200 / 800 words).
- Add probes for **language register** (academic vs. casual phrasing of the same correct content).
- Add probes for **multilingual** if you care.
- Compare two judges side-by-side: bind a smaller model (e.g. `openai/gpt-5-mini`) and run `judge_eval.py` against both. The disagreement matrix is a great teaching artifact.
