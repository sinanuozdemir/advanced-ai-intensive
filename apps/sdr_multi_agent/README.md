# sdr_multi_agent — Week 2 capstone

This app is the deployable artifact for **Week 2 Segment 6**.

## Origin

The base app (`flask_app/`, `mcp_servers/`, `docker-compose.yml`) is vendored
from chapter "SDR Multi-Agent" of *Building Agentic AI in Production* by
Sinan Ozdemir:

  [https://github.com/sinanuozdemir/building-agentic-ai/tree/main/sdr_multi_agent](https://github.com/sinanuozdemir/building-agentic-ai/tree/main/sdr_multi_agent)

## What Week 2 added

The base app was already a real multi-agent SDR with a HubSpot MCP server, a
research MCP server, and a Celery-backed Flask UI. Week 2 wires in three
ideas from the segment without rewriting the agent loop:


| Concept (segment)         | What we added                                                       | Where                           |
| ------------------------- | ------------------------------------------------------------------- | ------------------------------- |
| Multi-agent topology (S0) | Supervisor agent that delegates to the three sub-agents via Celery  | `flask_app/supervisor.py`       |
| Long-term memory (S3)     | Embedding-based semantic recall + end-of-thread reflection          | `flask_app/memory_agent.py`     |
| Checkpointing (S5)        | Async SQLite/Postgres checkpointer instead of `MemorySaver`         | `flask_app/memory_agent.py`     |

> Cloud deploy is deferred to **Week 3** alongside the broader deployment + observability segment.


The `MemoryAgent` class in `flask_app/memory_agent.py` is a subclass of
the base `GenericAgent` — when you point `app.py` at it, you get the
upgrades with no other code changes.

> **DSPy-optimized memory extraction** has moved to Week 3. Semantic memory
> is now natural-language strings written via an explicit `semantic_write`
> tool, so the old triple-F1 objective no longer applies.

## Run locally

```bash
cd apps/sdr_multi_agent
docker compose up        # spins flask app, postgres, rabbitmq, celery worker, MCP servers
# UI at http://localhost:8080
```

To opt in to the MemoryAgent upgrades, set `USE_MEMORY_AGENT=1` in your
environment before launching.

## How MemoryAgent plugs in

The base `GenericAgent` is unmodified. `flask_app/memory_agent.py`
defines `MemoryAgent`, which subclasses it and wires in:

- `make_async_sqlite_checkpointer` / `make_async_postgres_checkpointer` from
  `src/shared/checkpointer.py` (SQLite locally; switches to Postgres
  automatically when `DATABASE_URL` is set).
- `SemanticMemory`, `EpisodicMemory`, `ProceduralMemory` from `src/memory/`,
  partitioned per agent under `AGENT_DATA_ROOT/<memory_scope>/`.
- `reflect_on_thread()` at end-of-thread for episodic + procedural writes.

`flask_app/app.py` calls `make_agent()`, which returns `MemoryAgent`
when `USE_MEMORY_AGENT=1`. Otherwise the base agent runs unchanged.

Read `flask_app/memory_agent.py` end-to-end to see the full diff. The
notebooks 0–4 already demonstrate each primitive in isolation; this app
is what they look like wired together.

## The supervisor

The chat you see in the UI **is** the supervisor. The dropdown in the
sidebar still routes to per-config sub-agents (lead-gen / qualifier /
email) so you can talk to one directly for debugging, but the default
path is supervisor → sub-agent.

```
+-----------+        +--------------+       +---------------+
|   user    | <----> |  supervisor  |  -->  | Celery worker |
+-----------+        +------+-------+       +-------+-------+
                            |                       |
                            |  delegate_to_*        |  process_chat_task
                            |  (returns task_id)    |    config = lead_gen /
                            |                       |    qualifier / email
                            v                       v
                    +---------------+        +---------------+
                    | subtasks DB   | <----- | per-config    |
                    | (sqlite)      |        | GenericAgent  |
                    +---------------+        +-------+-------+
                            ^                        |
                            |                        v
         GET /api/subtasks/<conversation_id>  +---------------+
         (UI panel: initial fetch + manual    |  MCP servers  |
          refresh button)                     | (HubSpot etc) |
                                              +---------------+
```

The supervisor (`flask_app/supervisor.py`) is built with:

- **HubSpot MCP tools** (loaded from the `hubspot-mcp-server` container via
  `docker exec` stdio).
- **`semantic_write` / `semantic_search`** memory tools.
- **Three `delegate_to_*` tools** (`lead_gen`, `qualifier`, `email`) that
  enqueue Celery jobs against the matching agent config.
- **`read_subtask_result`** for chaining ("qualify the leads you just found"
  → look up the lead-gen task's output, then dispatch the qualifier).
- **`search_subtasks`** for cross-conversation lookup of past dispatches.

Sub-task dispatches are mirrored into a tiny SQLite log
(`flask_app/subtask_log.py`) so the UI's right-column **Sub-tasks** panel
can render every dispatch with its status (PENDING → PROCESSING → SUCCESS
/ FAILURE) and the sub-agent's final output. The panel does an initial
fetch on chat init and refreshes when the user clicks the refresh button;
individual async chat tasks are polled every 2s by `pollTaskResult`.

Toggle with `USE_SUPERVISOR=1` (default) and pick the routing model with
`SUPERVISOR_MODEL` (default `anthropic/claude-opus-4.6`).