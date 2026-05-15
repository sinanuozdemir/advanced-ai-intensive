# research-workflow MCP server

A FastMCP server that exposes the [agent_api](../README.md) workflow as four
tools any MCP client can call: `research`, `get_artifact`, `list_artifacts`,
`health`. The server is a stateless HTTP client — every tool just hits the
agent_api's HTTP surface — so the workflow's observability and scaling stay
in one place.

## Tools

| tool | wraps | notes |
|---|---|---|
| `research(topic, min_grade=4.0, max_iterations=3)` | `POST /research` | Blocks until the judge loop terminates or hits the cap. Returns `{artifact_id, final_grade, iterations, shape, summary}`. |
| `get_artifact(artifact_id)` | `GET /artifacts/{id}` | Full artifact + per-iteration provenance. |
| `list_artifacts(limit=20, offset=0)` | `GET /artifacts` | Pagination over saved artifacts. |
| `health()` | `GET /readyz` | Returns the upstream readiness payload, including which sub-checks (sqlite, llm, workflow) are passing. |

## Configuration

The server reads two env vars:

| var | default | purpose |
|---|---|---|
| `AGENT_API_BASE_URL` | `http://localhost:8090` | Where the agent_api is listening. |
| `AGENT_API_TIMEOUT_S` | `600` | HTTP timeout. Long, because `research` can run minutes. |

These are read from the **MCP client's** descriptor `env` block when the
server is spawned, so each client (Forge, Claude Desktop, Cursor) can point
the same server at a different agent_api deployment.

## Installing into Forge

Forge accepts MCP servers as **JSON descriptors** — a small file specifying
`command`, `args`, and optional `env`. (The legacy "drop a .py" path was
removed; see [`apps/forge/forge/mcp/manifest.py`](../../forge/forge/mcp/manifest.py)
for the rationale.)

### 1. Shape of the descriptor

[`forge_descriptor.json`](./forge_descriptor.json) ships an example. The
shape is:

```json
{
  "command": "python3",
  "args": [
    "/absolute/path/to/apps/agent_api/mcp_server/research_workflow_server.py"
  ],
  "env": {
    "AGENT_API_BASE_URL": "http://localhost:8090",
    "AGENT_API_TIMEOUT_S": "600"
  }
}
```

Two gotchas worth flagging up front, both rooted in how Forge spawns the
subprocess (via `asyncio.create_subprocess_exec`, **not** a shell):

* **Use an absolute path** for `args[0]`. Forge does not expand env vars or
  resolve relative paths from the descriptor location.
* **Use `python3`** (or the absolute path to a specific interpreter), not
  bare `python`. On macOS `python` is usually a shell alias for `python3`
  that doesn't exist as a real binary; `create_subprocess_exec` ignores
  aliases and you'll see `FileNotFoundError: 'python'`.

### 2. Install it

**From the Electron app:** open the MCP tab, drag the descriptor into the
drop zone, click _validate_ (Forge spawns the server, lists its tools, kills
it), name it (`research-workflow` is a good slug), click _install_. Restart
`forge serve` so the live engine picks it up.

**From a script / one-liner:**

```bash
curl -X POST http://127.0.0.1:6790/api/mcp/install \
  -H 'content-type: application/json' \
  -d @<(jq -n --rawfile contents \
          apps/agent_api/mcp_server/forge_descriptor.json \
          '{name:"research-workflow",
            description:"Research workflow over agent_api.",
            contents:$contents}')
```

Either path writes the descriptor to `.forge/mcp_servers/research-workflow.json`
and appends a `kind: "json"` entry to `.forge/mcp_servers.json`. You'll get
a `pending_restart: true` response — restart `forge serve` and the four
tools surface as `userresearch_workflow_research`, `_get_artifact`, etc.

### 3. Why this shape (and not a `.py` upload)

The descriptor approach has two properties the old `.py` drop didn't:

* **Explicit env.** The script needs `AGENT_API_BASE_URL` at runtime, and
  the descriptor is the only place a Forge user has to set it. A vendored
  `.py` had nowhere to put per-deployment config.
* **No executable upload.** Forge isn't accepting source code; it's
  accepting a small instruction to spawn an existing command. That's a much
  saner default for a permission-gated tool runner.

## Quick smoke test (no Forge)

With the agent_api running (`docker compose up` in `apps/agent_api/`), the
server can be tested directly:

```bash
cd apps/agent_api/mcp_server
AGENT_API_BASE_URL=http://localhost:8090 \
  python3 research_workflow_server.py
```

It speaks MCP over stdio; any MCP-aware client (Claude Desktop, MCP CLI,
Forge) can list and call the four tools.

## Why this isn't under `apps/forge/mcp_servers/`

The MCP wrapper is the agent_api's **public client surface**, so it ships
next to the service. Any MCP-aware coding agent — Forge, Claude Code,
opencode, Cursor — can install it. Moving it under `apps/forge/` would
imply Forge owns this workflow; it doesn't.
