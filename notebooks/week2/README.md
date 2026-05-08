# Week 2 — Advanced Agentic Systems Intensive

Multi-agent topologies, MCP, long-term memory, context compression, checkpointing, and one deployable capstone.

## What's different from Week 1

- Production code lives in `src/`; notebooks are thin walkthroughs that call into it.
- Two real apps ship at the end: **Forge** (local Electron) and **SDR** (Fly.io).
- Notebooks run against real LLM calls. No faked data.

## Segment → notebook → src

| # | Segment | Notebook | Source |
|---|---|---|---|
| 1 | Supervisor vs solo: when does multi-agent earn its keep? | `0_supervisor_vs_solo.ipynb` | `src/multi_agent/` |
| 2 | MCP and tool orchestration (four wiring patterns) | `1_mcp_orchestration.ipynb` | `src/mcp_demo/` |
| 3 | Long-term memory: semantic, episodic, procedural | `2_memory_systems.ipynb` | `src/memory/` |
| 4 | Context compression: which strategies survive contact with reality? | `3_context_compression.ipynb` | `src/middleware/` |
| 5 | Context management at scale: checkpointing & resume | `4_checkpointing_resumable.ipynb` | `src/shared/checkpointer.py` |
| 6 | Capstone: deploying the SDR app | _no notebook_ — read & run `apps/sdr_multi_agent/` | `apps/sdr_multi_agent/` + `apps/sdr_multi_agent/deploy/README.md` |

## Run order

Independent enough to read in any order. To feel a build:

1. `0_supervisor_vs_solo` — proves multi-agent isn't free.
2. `1_mcp_orchestration` — same RAG primitives via four clients (`direct`, `via_mcp`, `via_programmatic`, `via_coding_agent`); baked off for cost, latency, rubric.
3. `2_memory_systems` — `semantic_write` / `semantic_search` tools; episodic + procedural reflection.
4. `3_context_compression` — bake-off of summarization strategies under a rule-survival torture test.
5. `4_checkpointing_resumable` — crash recovery, history, time-travel, HITL.
6. **Capstone** — `cd apps/sdr_multi_agent && docker compose up`, then ship to Fly via `apps/sdr_multi_agent/deploy/README.md`.

> **Deferred to Week 3.** The DSPy MIPROv2 notebook from a prior version optimized a triple-based fact extractor. Semantic memory is now natural-language and tool-driven; triple-F1 is no longer the right objective. DSPy returns in Week 3 against an NL memory + retrieval metric.

Only `1_mcp_orchestration.ipynb` writes a resume-safe results CSV (`data/mcp_orchestration.csv`, keyed on `(client, query_id, run_idx)`). The other notebooks read/write data files but are not gated on a CSV check.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "OPENROUTER_API_KEY=sk-or-..." >> .env
```

Optional:

```bash
brew install claude            # for the via_coding_agent CLI handoff path
brew install opencode-ai/tap/opencode
cd apps/forge && npm install   # for the Forge Electron app
```

## Two apps you can ship

### Forge — local hybrid CLI + MCP coding agent

```bash
cd apps/forge && npm install && npm run dev
```

See `apps/forge/README.md` for architecture and `apps/forge/sample_repo/` for the demo project.

### SDR — cloud-deployed multi-agent SDR

```bash
fly apps create forge-sdr-app
fly postgres create --name forge-sdr-pg
fly postgres attach --app forge-sdr-app forge-sdr-pg
fly secrets set --app forge-sdr-app \
    OPENROUTER_API_KEY=... HUBSPOT_API_KEY=... \
    USE_MEMORY_AGENT=1 USE_SUPERVISOR=1
fly deploy --config apps/sdr_multi_agent/deploy/fly.toml \
           --dockerfile apps/sdr_multi_agent/deploy/Dockerfile
```

Walkthrough: `apps/sdr_multi_agent/deploy/README.md`. Architecture: `apps/sdr_multi_agent/README.md`.

## Recap

1. **Topology** — solo / supervisor / hierarchical / peer. Measured, not assumed.
2. **MCP + CLI agents** — four wiring patterns, one trade-off matrix.
3. **Memory** — semantic (tool-driven), episodic (reflection), procedural (reflection).
4. **Context compression** — five strategies bake-off; rule survival > token shrink.
5. **Checkpointing** — one primitive that buys crash recovery, history, time-travel, HITL.

Capstone: the SDR app turns these on behind `USE_MEMORY_AGENT=1`, `USE_SUPERVISOR=1`, and `DATABASE_URL`.

Week 3: agents that modify their own procedural memory safely, judge meta-evaluation, and DSPy prompt optimization.

## src/ layout

```
src/
├── shared/             # Re-exports of week-1 modules + checkpointer factory.
├── multi_agent/        # solo / supervisor / hierarchical / peer topologies.
├── mcp_demo/           # Teaching MCP server + 4 clients
│                       # (direct, via_mcp, via_programmatic, via_coding_agent).
│                       # Named mcp_demo because the official SDK owns `mcp`.
├── memory/             # semantic.py + episodic.py + procedural.py + reflect.py
└── middleware/         # Drop-in conversation-compression AgentMiddleware classes
                        # (5 strategies in the bake-off; recursive + hierarchical
                        # are defined but not swept).
```

## Costs (rough)

End-to-end across all 5 notebooks lands in the **$0.50–$2.00** range. The biggest line item is `1_mcp_orchestration.ipynb`: 240 trajectories on `anthropic/claude-sonnet-4.5` plus rubric judging on `claude-opus-4.7` typically costs **$0.30–$1.50**. The other four notebooks default to `openai/gpt-5.4-nano` and land at **$0.05–$0.10** each. Cached OpenRouter responses or a swap to a cheaper bake-off model both pull the total down.
