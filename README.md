![oreilly-logo](images/oreilly.png)

# Advanced Agentic Systems Intensive & AI Engineering Intensive

This repository contains code for the O'Reilly live courses: [Advanced Agentic Systems Intensive](https://learning.oreilly.com/live-events/advanced-agentic-systems-intensive/0642572350505/) & [AI Engineering Intensive](https://www.oreilly.com/live-events/ai-engineering-intensive/0642572375317/0642572375300/).

These three-week intensives go beyond introductory RAG and agents to build advanced AI workflows, models, and retrieval systems with multi-hop reasoning, query planning. We cover agentic RAG, MCP, multi-agent orchestration, long-term memory, agent harnesses, post-training, and production-grade evaluation. Through live coding and case studies from production systems, you will learn the architectural patterns, evaluation frameworks, and deployment strategies that separate demos from reliable, shippable AI systems.

All notebooks live in a **single flat directory** ([`notebooks/`](notebooks/)) with unnumbered filenames. **Week order is defined only by the schedules below** (and in [`notebooks/README.md`](notebooks/README.md)) — the two courses share files but diverge in sequence starting Week 2.

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

For the CORD-V2 vision SFT notebook (`cord_v2_receipt_sft.ipynb`), also set `FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT_ID`.

#### Step 3: Build the corpus once

Run [`notebooks/build_corpus.ipynb`](notebooks/build_corpus.ipynb). It scrapes Beehiiv, pulls a Wikipedia AI-history slice, samples HotpotQA, and writes a Chroma index + BM25 index + gold set into `notebooks/data/`. This is idempotent — re-running is a no-op once cached.

#### Step 4: Open notebooks and go

```bash
python3 -m jupyter notebook
# open files under notebooks/ in the schedule order for your course
```

## Schedules

### Shared Week 1 — Advanced Workflows, RAG, and Context

| Notebook | Segment |
|---|---|
| [`build_corpus.ipynb`](notebooks/build_corpus.ipynb) | Setup (run once) |
| [`rag_workflows.ipynb`](notebooks/rag_workflows.ipynb) | Advanced workflows + where simple RAG breaks down |
| [`multi_hop_and_query_decomposition.ipynb`](notebooks/multi_hop_and_query_decomposition.ipynb) | Multi-hop retrieval + query decomposition |
| [`hybrid_search_rerank_grade.ipynb`](notebooks/hybrid_search_rerank_grade.ipynb) | Hybrid search, re-ranking, filtering |
| [`context_window_optimization.ipynb`](notebooks/context_window_optimization.ipynb) | Context window optimization |
| [`adaptive_rag_capstone.ipynb`](notebooks/adaptive_rag_capstone.ipynb) | **Advanced Agentic only (Week 1 close):** Agentic RAG + 4-way bake-off — AI Engineering runs this as Week 2 opener instead |

### AI Engineering Intensive — Week 2

| Order | Artifact | Segment |
|---|---|---|
| 1 | [`adaptive_rag_capstone.ipynb`](notebooks/adaptive_rag_capstone.ipynb) | Agentic RAG + 4-way bake-off |
| 2 | [`mcp_orchestration.ipynb`](notebooks/mcp_orchestration.ipynb) | MCP + tool orchestration: four wiring patterns |
| 3 | [`supervisor_vs_solo.ipynb`](notebooks/supervisor_vs_solo.ipynb) | Solo / supervisor / hierarchical / peer topologies |
| 4 | [`memory_systems.ipynb`](notebooks/memory_systems.ipynb) | Long-term memory: semantic, episodic, procedural |
| 5 | [`cord_v2_receipt_sft.ipynb`](notebooks/cord_v2_receipt_sft.ipynb) | Post-training: CORD-V2 vision SFT via Fireworks SDK |
| 6 | [`apps/sdr_multi_agent/`](apps/sdr_multi_agent/) | Capstone: supervisor + memory + checkpointing (Flask/Celery) |

### Advanced Agentic Systems Intensive — Week 2

| Order | Artifact | Segment |
|---|---|---|
| 1 | [`supervisor_vs_solo.ipynb`](notebooks/supervisor_vs_solo.ipynb) | Solo / supervisor / hierarchical / peer topologies |
| 2 | [`mcp_orchestration.ipynb`](notebooks/mcp_orchestration.ipynb) | MCP + tool orchestration: four wiring patterns |
| 3 | [`memory_systems.ipynb`](notebooks/memory_systems.ipynb) | Long-term memory: semantic, episodic, procedural |
| 4 | [`context_compression.ipynb`](notebooks/context_compression.ipynb) | Context compression: 5-strategy bake-off |
| 5 | [`checkpointing_resumable.ipynb`](notebooks/checkpointing_resumable.ipynb) | Checkpointing: crash recovery, history, time-travel, HITL |
| 6 | [`apps/sdr_multi_agent/`](apps/sdr_multi_agent/) | Capstone: supervisor + memory + checkpointing |

Production code lives in [`src/`](src/) (`multi_agent/`, `mcp_demo/`, `memory/`, `middleware/`, `shared/`); the notebooks are thin walkthroughs that call into it. The SDR capstone wires the same primitives behind `USE_SUPERVISOR=1` and `USE_MEMORY_AGENT=1`. See [`notebooks/README.md`](notebooks/README.md) for costs and architecture notes.

### Week 3 — Evaluation, Memory at Scale, and a Capstone Agent

(Primarily Advanced Agentic; usable by both cohorts.)

| Notebook / app | Segment |
|---|---|
| [`judge_meta_eval.ipynb`](notebooks/judge_meta_eval.ipynb) | LLM-as-judge meta-eval: agreement and bias probes |
| [`dspy_judge_optimization.ipynb`](notebooks/dspy_judge_optimization.ipynb) | Optimizing the judge with DSPy |
| [`memory_benchmarks.ipynb`](notebooks/memory_benchmarks.ipynb) | Benchmarking semantic / episodic / procedural memory |
| [`plan_act_bakeoff.ipynb`](notebooks/plan_act_bakeoff.ipynb) | Plan-then-act vs act-only vs trajectory-probe routing |
| [`agent_workflow_api.ipynb`](notebooks/agent_workflow_api.ipynb) | Ship an agentic workflow as a service |
| [`apps/agent_api/`](apps/agent_api/) | FastAPI plan/research/reflect/artifact + FastMCP wrapper |
| [`apps/forge/`](apps/forge/) | Capstone: Forge coding agent (Electron + FastAPI) |

The Week 3 throughline is that **the eval is the product**: every claim about a judge, a memory tier, or a planning policy needs paired comparisons and a probe-weighted metric, not vibes. See [`notebooks/README.md`](notebooks/README.md) and [`apps/forge/README.md`](apps/forge/README.md).

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
notebooks/                 # flat; week order is README-only
  README.md                # dual-course schedules + costs
  _path_setup.py           # adds src/ + notebooks/ to sys.path
  build_corpus.ipynb
  rag_workflows.ipynb
  multi_hop_and_query_decomposition.ipynb
  hybrid_search_rerank_grade.ipynb
  context_window_optimization.ipynb
  adaptive_rag_capstone.ipynb
  supervisor_vs_solo.ipynb
  mcp_orchestration.ipynb
  memory_systems.ipynb
  context_compression.ipynb
  checkpointing_resumable.ipynb
  cord_v2_receipt_sft.ipynb
  judge_meta_eval.ipynb
  dspy_judge_optimization.ipynb
  memory_benchmarks.ipynb
  plan_act_bakeoff.ipynb
  agent_workflow_api.ipynb
  corpus.py llm.py retrievers.py judges.py eval_harness.py
  ep_eval.py bakeoff_lib.py judge_eval.py plan_act_alts.py
  data/                    # corpus, bake-off CSVs, gold sets, memory stores
src/                       # production code shared by notebooks + apps
  multi_agent/  mcp_demo/  memory/  middleware/  shared/
apps/
  sdr_multi_agent/         # Week 2 capstone: Flask + Celery + MCP + supervisor
  agent_api/               # Week 3: FastAPI research workflow + FastMCP wrapper
  forge/                   # Week 3 capstone: Electron + FastAPI coding agent
```

## Instructor

**Sinan Ozdemir** is the founder of Crucible, an AI factory platform that helps teams convert existing workflows into custom models. He is a Y Combinator alum, AI & LLM Advisor at Tola Capital, and the author of multiple books on data science and machine learning, including *Building Agentic AI*, *Quick Start Guide to LLMs*, and *Principles of Data Science*. Sinan is a former lecturer of Data Science at Johns Hopkins University and the founder of Kylie.ai, an enterprise-grade conversational AI platform (acquired 2014). He holds a master's degree in Pure Mathematics from Johns Hopkins University and is based in San Francisco, California.
