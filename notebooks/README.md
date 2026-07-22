# Notebooks

Flat directory of course notebooks (no `week1/` / `week2/` / `week3/` folders, no `0_` / `1_` prefixes). **Run order is defined by the schedules below** — the same files serve both O'Reilly intensives.

Helpers (`corpus.py`, `judges.py`, `ep_eval.py`, …) and `data/` live beside the notebooks. Import `import _path_setup` at the top of agent / MCP / memory notebooks so `src/` resolves.

## AI Engineering Intensive — Week 2 (live order)

1. [`adaptive_rag_capstone.ipynb`](adaptive_rag_capstone.ipynb) — agentic RAG vs tool-calling agents bake-off  
2. [`mcp_orchestration.ipynb`](mcp_orchestration.ipynb) — four tool-wiring patterns  
3. [`supervisor_vs_solo.ipynb`](supervisor_vs_solo.ipynb) — multi-agent topologies, measured  
4. [`memory_systems.ipynb`](memory_systems.ipynb) — semantic / episodic / procedural memory  
5. [`cord_v2_receipt_sft.ipynb`](cord_v2_receipt_sft.ipynb) — CORD-V2 vision SFT via Fireworks (`FIREWORKS_API_KEY`, `FIREWORKS_ACCOUNT_ID`)  
6. [`../apps/sdr_multi_agent/`](../apps/sdr_multi_agent/) — SDR capstone (`docker compose up`)

## Advanced Agentic Systems Intensive — Week 2

1. [`supervisor_vs_solo.ipynb`](supervisor_vs_solo.ipynb)  
2. [`mcp_orchestration.ipynb`](mcp_orchestration.ipynb)  
3. [`memory_systems.ipynb`](memory_systems.ipynb)  
4. [`context_compression.ipynb`](context_compression.ipynb)  
5. [`checkpointing_resumable.ipynb`](checkpointing_resumable.ipynb)  
6. [`../apps/sdr_multi_agent/`](../apps/sdr_multi_agent/)

## Shared Week 1 (both courses)

1. [`build_corpus.ipynb`](build_corpus.ipynb) — run once  
2. [`rag_workflows.ipynb`](rag_workflows.ipynb)  
3. [`multi_hop_and_query_decomposition.ipynb`](multi_hop_and_query_decomposition.ipynb)  
4. [`hybrid_search_rerank_grade.ipynb`](hybrid_search_rerank_grade.ipynb)  
5. [`context_window_optimization.ipynb`](context_window_optimization.ipynb)  
6. [`adaptive_rag_capstone.ipynb`](adaptive_rag_capstone.ipynb) — **Advanced Agentic Week 1 close**; AI Engineering runs this as Week 2 #1 instead

## Week 3 (eval + shipping)

1. [`judge_meta_eval.ipynb`](judge_meta_eval.ipynb)  
2. [`dspy_judge_optimization.ipynb`](dspy_judge_optimization.ipynb)  
3. [`memory_benchmarks.ipynb`](memory_benchmarks.ipynb)  
4. [`plan_act_bakeoff.ipynb`](plan_act_bakeoff.ipynb)  
5. [`agent_workflow_api.ipynb`](agent_workflow_api.ipynb) + [`../apps/agent_api/`](../apps/agent_api/)  
6. Capstone: [`../apps/forge/`](../apps/forge/)

## Setup notes

```bash
# from repo root
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # OPENROUTER_API_KEY required

# CORD notebook extras
# FIREWORKS_API_KEY=...
# FIREWORKS_ACCOUNT_ID=...
pip install --pre fireworks-ai eval-protocol
```

Optional CLIs for the coding-agent MCP path:

```bash
brew install claude
brew install opencode-ai/tap/opencode
```

## Costs (rough)

- RAG / topology / memory notebooks (default cheap models): ~$0.05–$0.10 each  
- `mcp_orchestration.ipynb` full bake-off: ~$0.30–$1.50  
- `cord_v2_receipt_sft.ipynb`: real GPU fine-tune + deploy on Fireworks — budget separately  
- Week 3 judge / plan-act sweeps: a few dollars depending on model choices  

## `src/` used by these notebooks

```
src/
├── shared/             # OpenRouter + checkpointer; lazy re-exports of notebook helpers
├── multi_agent/        # solo / supervisor / hierarchical / peer
├── mcp_demo/           # teaching MCP server + 4 clients
├── memory/             # semantic / episodic / procedural + reflection
└── middleware/         # conversation-compression AgentMiddleware
```
