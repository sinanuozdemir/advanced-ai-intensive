# Turn flow walkthrough

This is the step-by-step of a single user message through Forge, end to end.
It is written for someone wiring up the **Electron app** and the **slim TUI**
on top of the same headless backend (`forge serve`).

## Vocabulary

- **Turn** — one user message → one assistant reply. Forge calls one
  `engine.run_task` per turn. A turn ends with a `thread_end` trace event;
  the conversation thread itself keeps going across many turns.
- **Thread** — a sequence of turns sharing a `thread_id`. The LangGraph
  checkpointer keys off `thread_id` so resumption "just works".
- **Session** — a unit of memory bookkeeping. `/done` (or
  `cfg.memory.reflect_on_thread_end`) runs the summarizer over the session
  transcript, writes an episodic entry, and starts a fresh `thread_id`.
- **PlanActPolicy** — decides `(mode, topology)` per turn. Two ship:
  `trajectory_probe` (one cheap LLM call) and `tool_risk_heuristic`
  (regex; zero LLM cost). Both implement
  `forge.agent.plan_act.PlanActPolicy`.
- **Topology** — `solo` (one agent with merged tools) or `supervisor`
  (planner / coder / critic + ephemeral spawns + persistent agents).
- **MCP server** — a subprocess Forge spawns over stdio that exposes tools.
  Built-ins: `fs`, `shell`, `git`, `repo_rag`, `code`. User-added live in
  `<repo>/.forge/mcp_servers.json`.

## Boot

`forge serve` boots once per repo:

1. `ForgePaths.for_repo()` resolves the repo root (`.git` or `.forge`
   walk-up; `FORGE_REPO` env overrides).
2. `ensure_config(paths)` writes a default `.forge/config.toml` if missing.
3. `FastAPI` app comes up on `127.0.0.1:6790`. The engine is **lazy** — it
   doesn't initialize MCP subprocesses, Chroma collections, or
   checkpointer until the first `/api/chat` or `/ws/chat` connection.
4. On first connection, `ForgeEngine.start()`:
   - Loads MCP servers via `forge.mcp.tool_loader.load_mcp_tools`, which
     merges built-in `SERVERS` with entries from `mcp_servers.json` and
     wraps each tool in a `PermissionBroker.gate()` call.
   - Opens the Chroma collections for semantic / episodic memory.
   - Opens the async sqlite checkpointer.
   - Subscribes the WebSocket pubsub (`_broadcast`) to the `Tracer`.

The Electron app and TUI both subscribe to the same engine — they're just
different clients of the same backend.

## Example: user types "hi"

```
[user types in chat]
   |
   v
Electron / TUI sends:  {type: "chat", message: "hi"}  over /ws/chat
   |
   v
server.ws_chat → engine.run_task(task="hi", thread_id=...)
   |
   v
PlanActPolicy.decide(task="hi", history=[])
   - trajectory_probe LLM call returns Decision(mode="act", topology="solo",
     reason="pure greeting, no tools needed")
   - emits trace: policy_decision {mode, topology, reason}
   |
   v
Topology dispatch → solo agent (merged tools list)
   - LangGraph ReAct agent gets a system prompt + history + the user message
   - The model produces an AIMessage with no tool call: "Hi!"
   - emits trace: thread_end {task_id}
   |
   v
server.ws_chat sends:
   {type: "chat_result", thread_id, topology: "solo", planned: false,
    plan_md: null, answer: "Hi!", decision: {mode, topology, reason}}
```

In the Electron UI: a user bubble appears, a tiny `act/solo` chip surfaces
inline above the assistant bubble, then the assistant bubble fills with
"Hi!". In the slim TUI: same sequence, rendered as text in the transcript.

No permission asks. No spawns. Total cost: one routing call + one chat call.

## Example: heavier task

User: "find every place we set `OPENROUTER_API_KEY` and add a fallback
explaining how to use Ollama."

```
PlanActPolicy.decide
   - trajectory_probe sees write-class verbs ("add"), multi-file scope
     ("every place")
   - returns Decision(mode="plan", topology="supervisor")
   |
   v
Planner draft (PLANNER_SYSTEM)
   - reads + repo_rag.hybrid_retrieve("OPENROUTER_API_KEY")
   - drafts 5-step plan as markdown
   - emits trace: plan_drafted {plan_md}
   |
   v
HITL? (cfg.plan_act.hitl_auto_approve)
   - If false: PlanApproveScreen modal (TUI) or plan modal (Electron Chat tab)
   - If approved or auto-approved: continue
   |
   v
Supervisor dispatch
   - SUPERVISOR_ROUTER picks the next worker
   - coder agent edits the files via fs_edit (each call permission-gated)
   - critic agent reviews the diff
   - any of them may `spawn` an ephemeral sub-agent for a sub-task
   - emits trace: agent_spawn, tool_call, tool_result, agent_done x N
   |
   v
Each fs_edit triggers PermissionBroker.gate()
   - Static decision: "ask" by default
   - Server broadcasts permission_request {request_id, tool, args, agent}
   - Electron renders the modal; user clicks "allow"
   - Client sends {type: "permission_response", request_id, approved: true}
   - server.resolve_approval sets the broker's future → the gate returns "allow"
   - If no client responds within approval_timeout_s (default 60s):
     auto-approve fallback (configurable to deny)
   |
   v
Final answer assembled by critic
   - emits trace: thread_end
   - server.ws_chat sends chat_result with the final answer + plan_md
```

## Quick reference

| event type            | who emits                                | client uses for                  |
|-----------------------|------------------------------------------|----------------------------------|
| `ws_hello`            | server on WS connect                     | render "connected" badge         |
| `policy_decision`     | PlanActPolicy                            | the act/plan chip in chat        |
| `plan_drafted`        | planner worker                           | expandable plan block            |
| `agent_spawn`         | supervisor / spawn tool                  | spawn chip in chat               |
| `agent_done`          | supervisor                               | done chip in chat                |
| `tool_call`           | tool_loader's `_wrap_with_gate`          | pill row above the result        |
| `tool_result`         | tool_loader's `_wrap_with_gate`          | ✓ / ✗ icon                       |
| `permission_request`  | server.request_approval                  | the modal                        |
| `permission_timeout`  | server.request_approval                  | fade out a stale modal           |
| `memory_write`        | semantic_write tool                      | tiny chip (or hidden)            |
| `memory_read`         | semantic_read tool                       | tiny chip (or hidden)            |
| `compaction_fired`    | compaction middleware                    | dim "compact" pill               |
| `thread_end`          | engine.run_task                          | turn boundary; clear the buffer  |
| `chat_result`         | server.ws_chat                           | the assistant bubble             |
| `chat_error`          | server.ws_chat                           | red bubble                       |
| `heartbeat_consent_pending` | tool_loader (`_wrap_with_gate`)    | the create-heartbeat consent modal |
| `heartbeat_fired`     | HeartbeatScheduler tick                  | "tick started" chip in Heartbeats tab |
| `heartbeat_complete`  | HeartbeatScheduler tick                  | "tick done" chip + answer preview|
| `heartbeat_failed`    | HeartbeatScheduler tick                  | red chip with error              |
| `heartbeat_skipped`   | HeartbeatScheduler tick                  | "skipped (in flight)" chip       |

## Example: scheduling a heartbeat

User: "every 5 minutes, fetch https://pearson.com/foo and text me if it
changes."

```
PlanActPolicy.decide → act / solo (no codebase work needed)
   |
   v
Main agent inspects its toolbelt
   - it has heartbeat_*, fs_*, shell_exec, …
   - it does NOT have a "send text" MCP tool installed
   - per prompts.MAIN_SYSTEM, it honestly refuses:
     "I don't have a tool that can send SMS. Add an MCP server that
      provides one (e.g. Twilio) and ask again — I won't schedule a
      heartbeat that can't complete its task."
```

If the user later installs an MCP server with a `sms_send` tool, the same
prompt produces a `heartbeat_create` tool call instead:

```
Main agent calls heartbeat_create(
    name="watch-pearson",
    cron="*/5 * * * *",
    task="Fetch https://pearson.com/foo, hash it, compare to last entry
          in your scratchpad, sms_send if changed, then log a dated
          entry.",
    tools=["shell_exec", "user_sms_send",
           "heartbeat_scratchpad_read", "heartbeat_scratchpad_write"],
)
   |
   v
heartbeat MCP server returns {status: "consent_pending", proposed: ...}
   |
   v
tool_loader._wrap_with_gate sees the consent_pending payload and emits
heartbeat_consent_pending {proposed, warnings} on /ws/chat
   |
   v
Electron Chat tab pops the HeartbeatConsentModal:
   - lists the tools the heartbeat would gain auto-allow for
   - shows the proposed cron's next 3 fire times (croniter preview)
   - "Approve" → POST /api/heartbeats with the proposed spec
   - "Cancel" → no-op; the main agent gets back
     {status: "consent_denied"} so it can apologize
   |
   v
On approve: HeartbeatScheduler.reload(spec)
   - PermissionBroker.set_heartbeat_allowlist("watch-pearson", spec.tools)
   - asyncio task starts sleeping until next cron boundary
```

Every tick:

```
sleep until next_fire_at (croniter)
   |
   v
HeartbeatScheduler._tick(slot)
   - thread_id = f"hb-{name}-{uuid4-prefix}"   # unique per tick → fresh run
   - emit heartbeat_fired {name, cron, thread_id, tools}
   |
   v
HeartbeatScheduler._invoke_tick
   - build a fresh per-tick agent via langchain.create_agent
     (no checkpointer — scratchpad is the only memory)
   - narrow tools to spec.tools, agent_name = "heartbeat:watch-pearson"
   - read .forge/heartbeats/watch-pearson.scratchpad.md
   - compose user message:

       Today is 2026-05-12 22:00 PT (ISO: …).

       ### Your scratchpad (watch-pearson.scratchpad.md)
       ```
       - 2026-05-12 21:55 PT: hash abc123, no change.
       - 2026-05-12 21:50 PT: hash abc123, no change.
       …
       ```

       ### Task
       Fetch …

   - agent.ainvoke({messages: [HumanMessage(content=…)]},
                   config={configurable: {thread_id}})
   - the agent calls shell_exec → hashes the body → compares to scratchpad
     → calls heartbeat_scratchpad_write(name, contents=updated body)
     → optionally calls user_sms_send
   |
   v
Append RunRecord to .forge/heartbeats/watch-pearson.runs.jsonl
emit heartbeat_complete {name, duration_ms, answer_preview}
```

Per-tick permission decisions go through the same `PermissionBroker` as
the main agent, but with two twists for the heartbeat agent name:

1. **Outside-allowlist deny.** Any tool not in `spec.tools` is denied at
   the broker layer regardless of the global `[permissions.tools]`
   default. The consented allowlist *is* the heartbeat's authorization.
2. **Ask → allow promotion for in-allowlist tools.** A tool that would
   normally be "ask" auto-promotes to "allow" since the user consented
   to it at create time. "deny" rules in config still win.

This means a heartbeat can't quietly grow new capabilities just by the
LLM deciding to call a new tool — you'd have to delete and recreate the
heartbeat (and re-consent in the modal).

## Where to look in code

- `apps/forge/forge/agent/engine.py` — `ForgeEngine.run_task` orchestrates
  one turn. Read this first; it's the spine.
- `apps/forge/forge/agent/plan_act/` — both policies live here. `base.py`
  defines the protocol.
- `apps/forge/forge/agent/permissions.py` — broker + `gate()`. The approver
  coroutine wired in by `forge.server.ensure_engine` is
  `_ServerState.request_approval`.
- `apps/forge/forge/mcp/tool_loader.py` — how MCP tools become LangChain
  tools wrapped with a permission gate. User servers from
  `forge.mcp.manifest` are merged in here.
- `apps/forge/forge/server.py` — REST + WebSocket. `/ws/chat` is the
  bidirectional channel; `/ws` is the same but without the `chat` /
  `chat_result` messages.
- `apps/forge/electron/src/renderer/src/views/Chat.tsx` — the
  client-side state machine that turns the trace stream into chat bubbles
  with inline cards.
- `apps/forge/forge/heartbeats/` — `spec.py` (Pydantic schema + cron
  validation), `registry.py` (TOML round-trip + scratchpad helpers),
  `scheduler.py` (the asyncio loop + `_invoke_tick` that prepends the
  scratchpad and today's date to the per-tick user message).
- `apps/forge/mcp_servers/heartbeat_server.py` — the MCP tools the main
  agent uses (`heartbeat_create` returns the `consent_pending` payload,
  `heartbeat_scratchpad_read` / `heartbeat_scratchpad_write` manage the
  per-heartbeat markdown notebook).
- `apps/forge/electron/src/renderer/src/views/Heartbeats.tsx` — the
  tab that lists heartbeats, lets you pause/resume/run-now, and shows
  each heartbeat's scratchpad + recent runs.
