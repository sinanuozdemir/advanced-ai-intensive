![oreilly-logo](images/oreilly.png)

# Advanced Agentic Systems Intensive

This repository contains code for my O'Reilly live course: [Advanced Agentic Systems Intensive](https://learning.oreilly.com/live-events/advanced-agentic-systems-intensive/0642572350505/).

This three-week intensive goes beyond introductory RAG and agents to build advanced workflows and retrieval systems with multi-hop reasoning, query planning, agentic RAG, MCP, multi-agent orchestration, long-term memory, agent harnesses, and production-grade evaluation. Through live coding and case studies from production systems, you will learn the architectural patterns, evaluation frameworks, and deployment strategies that separate demos from reliable, shippable AI systems.

The repo contains live-coded notebooks, shared utilities, and a heterogeneous benchmark used throughout the cohort.

## Setup

### Using a Python 3.11 Virtual Environment

At the time of writing, we need a Python virtual environment with Python 3.11 or later.

#### Step 1: Create and activate the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

#### Step 2: Configure API keys

Copy `.env.example` to `.env` and add at minimum an `OPENROUTER_API_KEY`. One key gives you access to every model used in the course (OpenAI, Anthropic, DeepSeek, Llama, Qwen, Grok, ...).

#### Step 3: Build the corpus once

Run [`notebooks/week1/0_build_corpus.ipynb`](notebooks/week1/0_build_corpus.ipynb). It scrapes Beehiiv, pulls a Wikipedia AI-history slice, samples HotpotQA, and writes a Chroma index + BM25 index + gold set into `notebooks/week1/data/`. This is idempotent — re-running is a no-op once cached.

#### Step 4: Open notebook 1 and go

```bash
python3 -m jupyter notebook
```

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

## Notebooks

### Week 1 — Advanced Workflows, RAG, and Context

| #   | Notebook                                                                                              | Course segment |
|-----|-------------------------------------------------------------------------------------------------------|----------------|
| 0   | [`0_build_corpus.ipynb`](notebooks/week1/0_build_corpus.ipynb)                                        | Setup (run once) |
| 1   | [`1_rag_workflows.ipynb`](notebooks/week1/1_rag_workflows.ipynb)                                      | S1 — Advanced workflows + where simple RAG breaks down |
| 2   | [`2_multi_hop_and_query_decomposition.ipynb`](notebooks/week1/2_multi_hop_and_query_decomposition.ipynb) | S2 — Multi-hop retrieval + query decomposition |
| 3   | [`3_hybrid_search_rerank_grade.ipynb`](notebooks/week1/3_hybrid_search_rerank_grade.ipynb)            | S3 — Hybrid search, re-ranking, filtering |
| 4   | [`4_context_window_optimization.ipynb`](notebooks/week1/4_context_window_optimization.ipynb)          | S4 — Context window optimization |
| 5   | [`5_adaptive_rag_capstone.ipynb`](notebooks/week1/5_adaptive_rag_capstone.ipynb)                      | S5 — Agentic RAG + 4-way bake-off |

The Week 1 spine is an **adaptive RAG loop** (`retrieve -> rerank -> grade -> gap-analyze -> iterate`) compared head-to-head against three tool-calling agent variants across multiple OpenRouter models.

### Week 2 — Multi-Agent Systems, MCP, and Memory

| #   | Notebook                                                                                  | Course segment |
|-----|-------------------------------------------------------------------------------------------|----------------|
| 0   | [`0_supervisor_vs_solo.ipynb`](notebooks/week2/0_supervisor_vs_solo.ipynb)                | S1 — Solo / supervisor / hierarchical / peer topologies, measured |
| 1   | [`1_mcp_orchestration.ipynb`](notebooks/week2/1_mcp_orchestration.ipynb)                  | S2 — MCP + tool orchestration: four wiring patterns, baked off |
| 2   | [`2_memory_systems.ipynb`](notebooks/week2/2_memory_systems.ipynb)                        | S3 — Long-term memory: semantic, episodic, procedural |
| 3   | [`3_context_compression.ipynb`](notebooks/week2/3_context_compression.ipynb)              | S4 — Context compression: 5-strategy bake-off under rule-survival |
| 4   | [`4_checkpointing_resumable.ipynb`](notebooks/week2/4_checkpointing_resumable.ipynb)      | S5 — Checkpointing: crash recovery, history, time-travel, HITL |
| —   | [`apps/sdr_multi_agent/`](apps/sdr_multi_agent/)                                          | S6 — Capstone: supervisor + memory + checkpointing in one Flask/Celery app |

Production code lives in [`src/`](src/) (`multi_agent/`, `mcp_demo/`, `memory/`, `middleware/`, `shared/`); the notebooks are thin walkthroughs that call into it. The capstone wires the same primitives into a real SDR app behind `USE_SUPERVISOR=1` and `USE_MEMORY_AGENT=1`. See [`notebooks/week2/README.md`](notebooks/week2/README.md) for run order, costs, and architecture.

### Week 3 — Evaluation, Observability, and Deployment

Coming soon. Notebooks live under [`notebooks/week3/`](notebooks/week3/). The `eval_harness.py` from Week 1 generalizes into the agent harness for benchmarking SWE-bench / GAIA-style tasks, paired with LangSmith tracing, human-in-the-loop guardrails, and deployment patterns (containers, cost/latency, monitoring).

## Prerequisites

- Intermediate-to-advanced Python (async, classes, multi-file projects).
- Working knowledge of LLM APIs (OpenAI / Anthropic / OpenRouter).
- Prior experience with RAG (embeddings, vector DBs, basic pipelines).
- Familiarity with at least one agent framework (LangChain, LangGraph, CrewAI, etc.). This course does **not** cover agent fundamentals.

## Recommended preparation

- Read: *Building Agentic AI* by Sinan Ozdemir — [O'Reilly](https://learning.oreilly.com/library/view/building-agentic-ai/9780135489710/) · [Amazon](https://a.co/d/eaTeURV)
- Read: *Quick Start Guide to Large Language Models* (2nd ed.) by Sinan Ozdemir — [O'Reilly](https://learning.oreilly.com/library/view/quick-start-guide/9780135346570/) · [Amazon](https://www.amazon.com/Quick-Start-Guide-Language-Models-dp-0135346568/dp/0135346568)
- Watch: [Quick Start Guide to Large Language Models: ChatGPT, Llama, Embeddings, Fine-Tuning and Multimodal AI](https://learning.oreilly.com/videos/-/9780135384800/) by Sinan Ozdemir
- Explore: [AI Unveiled Expert Playlist](https://learning.oreilly.com/playlists/0c7b9a4a-de71-4235-864e-c23c64473276/) by Sinan Ozdemir
- [`oreilly-langgraph`](https://github.com/sinanuozdemir/oreilly-langgraph) — the introductory course this one builds on
- [`oreilly-ai-agents`](https://github.com/sinanuozdemir/oreilly-ai-agents) — broader survey of agent frameworks

## Recommended follow-up

- Watch: [Designing and Optimizing LLM Pipelines](https://learning.oreilly.com/live-events/designing-and-deploying-llm-pipelines/0642572014796/) by Sinan Ozdemir
- Watch: [Modern AI Agents](https://learning.oreilly.com/course/modern-ai-agents/9780135882634/) by Sinan Ozdemir

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
  week2/
    0_supervisor_vs_solo.ipynb  ...  4_checkpointing_resumable.ipynb
    _path_setup.py    # adds repo root to sys.path so notebooks import src/
    data/             # gitignored: memory stores, checkpoint sqlite, bake-off CSVs
  week3/  (placeholder)
src/                  # production code shared across week 2 notebooks + apps
  multi_agent/        # solo / supervisor / hierarchical / peer topologies
  mcp_demo/           # teaching MCP server + 4 client wiring patterns
  memory/             # semantic / episodic / procedural + reflection
  middleware/         # conversation-compression AgentMiddleware classes
  shared/             # checkpointer factory + OpenRouter LLM helpers
apps/
  sdr_multi_agent/    # Week 2 capstone: Flask + Celery + MCP + supervisor
```

## Instructor

**Sinan Ozdemir** is the founder of Crucible, an AI factory platform that helps teams convert existing workflows into custom models. He is a Y Combinator alum, AI & LLM Advisor at Tola Capital, and the author of multiple books on data science and machine learning, including *Building Agentic AI*, *Quick Start Guide to LLMs*, and *Principles of Data Science*. Sinan is a former lecturer of Data Science at Johns Hopkins University and the founder of Kylie.ai, an enterprise-grade conversational AI platform (acquired 2014). He holds a master's degree in Pure Mathematics from Johns Hopkins University and is based in San Francisco, California.
