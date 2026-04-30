# Advanced Agentic AI in Three Weeks

Pearson live event by **Sinan Ozdemir** — go beyond introductory RAG and agents to build advanced workflows and retrieval systems with multi-hop reasoning, query planning, agentic RAG, MCP, multi-agent orchestration, harnesses, and production-grade evaluation.

This repo contains live-coded notebooks, shared utilities, and a heterogeneous benchmark for the three-week intensive.

## Setup

1. **Python 3.11+** in a fresh virtualenv:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Copy `.env.example` to `.env`** and add at minimum an `OPENROUTER_API_KEY`. One key gives you access to every model used in the course (OpenAI, Anthropic, DeepSeek, Llama, Qwen, Grok, ...).

3. **Build the corpus once** by running [`notebooks/week1/0_build_corpus.ipynb`](notebooks/week1/0_build_corpus.ipynb). It scrapes Beehiiv, pulls a Wikipedia AI-history slice, samples HotpotQA, and writes a Chroma index + BM25 index + gold set into `notebooks/week1/data/`. This is idempotent — re-running is a no-op once cached.

4. Open notebook 1 and go.

## OpenRouter as the multi-model backbone

Every notebook calls models through a single helper:

```python
from llm import get_llm

llm = get_llm("cheap_workhorse")        # by named role
llm = get_llm("openai/gpt-5.5")         # or by raw OpenRouter slug
```

Named roles in `notebooks/week1/llm.py`:

| Role                 | Default slug                                  |
|----------------------|-----------------------------------------------|
| `cheap_workhorse`    | `openai/gpt-5.4-nano`                         |
| `frontier_chat`      | `anthropic/claude-opus-4.7`                   |
| `frontier_openai`    | `openai/gpt-5.5`                              |
| `frontier_anthropic` | `anthropic/claude-opus-4.7`                   |
| `reasoning_openai`   | `openai/o4-mini`                              |
| `reasoning_open`     | `moonshotai/kimi-k2-thinking`                 |
| `open_weight`        | `qwen/qwen3.6-35b-a3b`                        |
| `fast_open`          | `x-ai/grok-4.1-fast`                          |

Slugs verified against OpenRouter on 2026-04-28. They're tweakable in one place — swap them in `notebooks/week1/llm.py` as the catalog evolves.

## Week 1 — Advanced Workflows, RAG, and Context

| #   | Notebook                                                                                              | Course segment |
|-----|-------------------------------------------------------------------------------------------------------|----------------|
| 0   | [`0_build_corpus.ipynb`](notebooks/week1/0_build_corpus.ipynb)                                        | Setup (run once) |
| 1   | [`1_rag_workflows.ipynb`](notebooks/week1/1_rag_workflows.ipynb)        | S1 — Where simple RAG breaks down |
| 2   | [`2_multi_hop_and_query_decomposition.ipynb`](notebooks/week1/2_multi_hop_and_query_decomposition.ipynb) | S2 — Multi-hop retrieval + query planning |
| 3   | [`3_hybrid_search_rerank_grade.ipynb`](notebooks/week1/3_hybrid_search_rerank_grade.ipynb)            | S3 — Hybrid search, reranking, grading |
| 4   | [`4_context_window_optimization.ipynb`](notebooks/week1/4_context_window_optimization.ipynb)          | S4 — Context window optimization |
| 5   | [`5_adaptive_rag_capstone.ipynb`](notebooks/week1/5_adaptive_rag_capstone.ipynb)                      | S5 — Agentic RAG + 4-way bake-off |

The Week 1 spine is an **adaptive RAG loop** (`retrieve -> rerank -> grade -> gap-analyze -> iterate`) compared head-to-head against three tool-calling agent variants across multiple OpenRouter models.

## Week 2 — Multi-Agent Systems, MCP, and Memory

Coming soon. Notebooks live under [`notebooks/week2/`](notebooks/week2/). The `gap_analyzer` node from Week 1's capstone evolves into a supervisor that delegates sub-queries to specialist research agents.

## Week 3 — Evaluation, Observability, and Deployment

Coming soon. Notebooks live under [`notebooks/week3/`](notebooks/week3/). The `eval_harness.py` from Week 1 generalizes into the agent harness for benchmarking SWE-bench / GAIA-style tasks.

## Recommended preparation

- *Building Agentic AI* by Sinan Ozdemir (book)
- *Quick Start Guide to Large Language Models* by Sinan Ozdemir (book + video)
- [`oreilly-langgraph`](https://github.com/sinanuozdemir/oreilly-langgraph) — the introductory course this one builds on
- [`oreilly-ai-agents`](https://github.com/sinanuozdemir/oreilly-ai-agents) — broader survey of agent frameworks

## Repo layout

```
notebooks/
  week1/
    0_build_corpus.ipynb
    1_..ipynb  ...  5_adaptive_rag_capstone.ipynb
    llm.py            # OpenRouter model registry + factory
    corpus.py         # Loaders for the prebuilt index + gold set
    retrievers.py     # Hybrid (BM25 + dense + RRF), cross-encoder rerank
    judges.py         # Pydantic rubrics + LLM-as-judge helpers
    eval_harness.py   # Multi-variant x multi-model evaluation runner
    data/             # gitignored: corpus_cache, chroma_db, gold_set, results
  week2/  (placeholder)
  week3/  (placeholder)
```
