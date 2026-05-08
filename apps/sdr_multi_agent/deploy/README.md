# Deploying the Week 2 SDR app to Fly.io

End-to-end recipe to ship the SDR app + Postgres + Redis on Fly.io with the
Week 2 extensions (memory + checkpointing) turned on.

> **Broker note.** The local Compose stack runs **RabbitMQ**; the Fly recipe
> below uses **Upstash Redis** (managed, free tier). Either works — Celery
> only cares about `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND`.

## Prereqs

- A Fly.io account and `fly` CLI installed (`brew install flyctl`).
- An OpenRouter API key (`OPENROUTER_API_KEY`).
- A HubSpot private-app access token (`HUBSPOT_API_KEY`) for the HubSpot MCP server.
- (Optional) `claude` CLI binary if you want the CLI handoff path enabled.

## 1. Create the apps

We need three things:

```bash
# from the repo root
cd /path/to/advanced-agentic-ai-in-three-weeks

# (a) The SDR app itself
fly apps create forge-sdr-app --machines

# (b) Managed Postgres for LangGraph checkpoints + episodic vectors metadata
fly postgres create --name forge-sdr-pg --region iad --vm-size shared-cpu-1x --volume-size 10
fly postgres attach --app forge-sdr-app forge-sdr-pg
# -> sets DATABASE_URL secret on forge-sdr-app

# (c) Managed Redis for Celery
fly redis create --name forge-sdr-redis --region iad
# -> follow prompts; copy the connection string into the next step
```

## 2. Set secrets

```bash
REDIS_URL="redis://default:<token>@<host>:6379"

fly secrets set --app forge-sdr-app \
    OPENROUTER_API_KEY=sk-or-v1-... \
    HUBSPOT_API_KEY=pat-na1-... \
    SECRET_KEY="$(openssl rand -hex 32)" \
    CELERY_BROKER_URL="$REDIS_URL" \
    CELERY_RESULT_BACKEND="$REDIS_URL" \
    USE_MEMORY_AGENT=1 \
    USE_SUPERVISOR=1 \
    SUPERVISOR_MODEL=anthropic/claude-opus-4.6
```

## 3. Deploy

```bash
# from the repo root (build context must include src/)
fly deploy \
    --app forge-sdr-app \
    --config apps/sdr_multi_agent/deploy/fly.toml \
    --dockerfile apps/sdr_multi_agent/deploy/Dockerfile
```

Watch the logs:

```bash
fly logs --app forge-sdr-app
# Look for: "SDR running with MemoryAgent"
```

Visit `https://forge-sdr-app.fly.dev/`.

## How the upgrades turn on

Three env vars flip the entire surface:

- **`DATABASE_URL`** (set automatically by `fly postgres attach`) — `MemoryAgent` calls `make_async_postgres_checkpointer` instead of `make_async_sqlite_checkpointer`. LangGraph state survives Celery worker crashes and process restarts.
- **`USE_MEMORY_AGENT=1`** — `flask_app/app.py` calls `make_agent()` which returns `MemoryAgent` (in `flask_app/memory_agent.py`). That subclass swaps in semantic + episodic + procedural memory and the async checkpointer. The base `GenericAgent` is otherwise unmodified.
- **`USE_SUPERVISOR=1`** — the chat endpoint binds to the supervisor agent (`flask_app/supervisor.py`) instead of a per-config sub-agent. Requires a live Celery worker + a broker (RabbitMQ locally, Redis on Fly), since the supervisor's `delegate_*` tools enqueue jobs the worker picks up. Set `SUPERVISOR_MODEL` to control the routing model.

Read `flask_app/memory_agent.py` and `flask_app/supervisor.py` to see the diffs in one file each.

## 4. Verify week-2 features end-to-end

```bash
# Trigger a chat
curl -X POST https://forge-sdr-app.fly.dev/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hi, I am Anna at Acme Corp, 250-person Berlin fintech.","conversation_id":"smoke-1"}'

# Inspect persisted state. AGENT_DATA_ROOT=/data, and stores are partitioned
# per agent scope (supervisor / lead_gen / qualifier / email).
fly ssh console --app forge-sdr-app -C "ls -la /data/supervisor/semantic_chroma"
```

You should see:
1. The chat response.
2. The agent's tool calls in the trace include `semantic_write(...)` whenever
   it decides a fact is worth keeping.
3. Chroma collections at `/data/<scope>/semantic_chroma` persist across
   worker restarts (the volume is mounted at `/data`).
4. After running `end_thread(...)`, an entry in `episodic_chroma/` and (sometimes) skills in `procedural.sqlite`.

> DSPy-driven memory-extraction optimization is deferred to **Week 3**.
> Semantic memory is now natural-language and tool-driven, so there is no
> `extractor_prompt.txt` to ship.

## Costs

Rough monthly estimate at light/demo traffic:

- Fly app (shared-cpu-1x, 2 processes): ~$5
- Postgres (shared-cpu-1x, 10GB): ~$10
- Redis (free Upstash tier): $0
- OpenRouter usage: depends on volume; expect $0.10-$2 per 1k SDR turns on `gpt-5.4-nano`
