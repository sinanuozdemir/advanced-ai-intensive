"""Forge headless backend — FastAPI + WebSocket.

The Electron app and the TUI are both clients of this server. ``forge serve``
launches it on ``127.0.0.1:<config.ui.server_port>`` (default 6790). The
server boots one ``ForgeEngine`` and keeps it alive for the lifetime of the
process — every WebSocket subscriber tees off the same ``Tracer``, every
``POST /api/chat`` reuses the same MCP subprocesses and memory stores.

Endpoints:

- ``GET  /``                             — tiny JSON pointer (no HTML SPA here)
- ``GET  /api/health``                 — liveness probe
- ``GET  /api/workspace``              — repo root, branch, dirty, head sha
- ``GET  /api/config``                 — current ``.forge/config.toml`` (parsed)
- ``GET  /api/config/schema``          — JSON Schema for ``ForgeConfig``
- ``PUT  /api/config``                 — write+reload config (next-thread only)
- ``GET  /api/agents``                 — list persistent agents
- ``PUT  /api/agents/<name>``          — upsert
- ``DELETE /api/agents/<name>``        — remove
- ``GET  /api/memory/semantic?q=...&k=10``
- ``GET  /api/memory/episodic?limit=20``
- ``GET  /api/memory/procedural``
- ``POST /api/chat`` ``{ message }``   — start a chat turn; returns
  ``{ thread_id, topology, planned, answer }`` (synchronous turn for now)
- ``GET  /api/trace?tail=200``         — last N trace events
- ``GET  /api/audit?tail=200``         — last N audit lines
- ``GET  /api/models/health?slug=...`` — verify a model slug actually resolves
  (catches bad prefixes, missing keys, or an offline Ollama before a turn).
- ``GET  /api/mcp``                    — list built-in + user-added MCP servers.
- ``POST /api/mcp/validate``           — spawn a candidate server, list its
  tools, kill it. Body: ``{contents}`` (a JSON descriptor).
- ``POST /api/mcp/install``            — validate + persist a server. Body:
  ``{name, contents, description?}``. Returns ``pending_restart: true``.
- ``POST /api/mcp/reload``             — tear down the in-process engine and
  re-boot it so manifest changes pick up without restarting ``forge serve``.
- ``GET  /api/eval/threads``           — list past per-thread evals (newest first).
- ``GET  /api/eval/threads/{id}``      — one eval (rubric scores + trajectory).
- ``POST /api/eval/threads/{id}/run``  — re-run rubrics for one thread on demand.
- ``GET  /api/eval/rubrics``           — current rubric system prompts.
- ``WS   /ws``                          — bidirectional. Outbound: live trace,
  audit, and ``permission_request`` events. Inbound: clients reply to
  permission asks with ``{type:"permission_response", request_id, approved}``;
  unanswered asks auto-approve after ``approval_timeout_s`` (60s by default).
- ``WS   /ws/chat``                      — bidirectional chat socket. Send
  ``{type:"chat", message, thread_id?, topology?}`` to start a turn; the
  server streams every trace event over the same socket and finishes with
  ``{type:"chat_result", thread_id, answer, planned, plan_md, decision}``.
  Permission responses can be sent on the same socket so the Electron app
  doesn't need to multiplex two WebSockets.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import (
    ForgeConfig,
    ForgeConfigError,
    ensure_config,
    load_config,
    write_config,
)
from .paths import ForgePaths


# ---------------------------------------------------------------------------
# State holder — one engine + tracer subscription per process
# ---------------------------------------------------------------------------


class _ServerState:
    """Lazy holder for the ForgeEngine. We boot it on first chat turn so a
    plain ``GET /api/config`` doesn't pay the MCP startup cost."""

    def __init__(self, paths: ForgePaths) -> None:
        self.paths = paths
        self.cfg: ForgeConfig = load_config(paths)
        self.engine: Any | None = None
        self._engine_lock = asyncio.Lock()
        # Each subscriber is (queue, loop) so cross-loop / cross-thread
        # broadcasts can hand the event off via call_soon_threadsafe.
        self.subscribers: set[tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = set()
        self._sub_lock = threading.Lock()
        # Pending permission asks → futures keyed by request_id. The /ws
        # endpoint inbound loop calls resolve_approval() when the UI replies.
        self.pending_approvals: dict[str, asyncio.Future[bool]] = {}
        self._approvals_lock = asyncio.Lock()
        # Seconds the broker waits for a UI client before falling back to
        # the safe default (auto-approve, per the planned spec — flip to
        # False here to make the headless behavior auto-deny instead).
        self.approval_timeout_s: float = 60.0
        self.approval_timeout_decision: bool = True

    async def ensure_engine(self) -> Any:
        if self.engine is not None:
            return self.engine
        async with self._engine_lock:
            if self.engine is not None:
                return self.engine
            from .agent.engine import ForgeEngine

            engine = await ForgeEngine.start(
                paths=self.paths, cfg=self.cfg, approver=self.request_approval,
            )
            # Tee tracer events to every WS subscriber.
            engine.tracer.subscribe(self._broadcast)
            self.engine = engine
            return engine

    async def shutdown(self) -> None:
        if self.engine is not None:
            with contextlib.suppress(Exception):
                await self.engine.shutdown()
            self.engine = None

    # ---------------------------------------------------------------- WS pubsub

    def add_subscriber(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        loop = asyncio.get_running_loop()
        entry = (q, loop)
        with self._sub_lock:
            self.subscribers.add(entry)
        # Stash the entry on the queue so remove_subscriber can find it.
        q._forge_entry = entry  # type: ignore[attr-defined]
        return q

    def remove_subscriber(self, q: asyncio.Queue) -> None:
        entry = getattr(q, "_forge_entry", None)
        if entry is None:
            return
        with self._sub_lock:
            self.subscribers.discard(entry)

    def _broadcast(self, event: dict) -> None:
        """Tracer hook. Push to every WS queue without blocking the writer.

        Broadcasts may originate from any thread / any loop (the tracer is
        called from whichever coroutine emits). We dispatch onto each
        subscriber's own loop via ``call_soon_threadsafe`` so cross-loop
        broadcasts work too (e.g. starlette TestClient, or future runs that
        spin up engines on a worker thread)."""
        with self._sub_lock:
            subs = list(self.subscribers)
        for q, loop in subs:
            try:
                if loop.is_closed():
                    continue
                loop.call_soon_threadsafe(_safe_put, q, event)
            except RuntimeError:
                continue

    # --------------------------------------------------------- approvals (HITL)

    async def request_approval(
        self, *, tool_name: str, args: dict, agent_name: str, reason: str,
    ) -> bool:
        """Approver coroutine registered with the engine.

        Broadcasts a ``permission_request`` event to every WS subscriber,
        waits for any client to send back a matching ``permission_response``,
        and falls back to ``approval_timeout_decision`` (default: auto-approve)
        after ``approval_timeout_s`` seconds.

        Multiple clients can be watching, but the first response wins —
        subsequent ones are ignored because the future is already resolved.
        """
        request_id = str(uuid.uuid4())
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        async with self._approvals_lock:
            self.pending_approvals[request_id] = fut

        # Broadcast through the regular pubsub so every connected client sees
        # it and so it gets persisted in /api/trace via the same code path.
        self._broadcast({
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "permission_request",
            "request_id": request_id,
            "tool": tool_name,
            "agent": agent_name,
            "reason": reason,
            "args": _truncate_for_ui(args),
            "timeout_s": self.approval_timeout_s,
        })

        try:
            try:
                approved = await asyncio.wait_for(fut, timeout=self.approval_timeout_s)
            except asyncio.TimeoutError:
                approved = self.approval_timeout_decision
                self._broadcast({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "type": "permission_timeout",
                    "request_id": request_id,
                    "tool": tool_name,
                    "agent": agent_name,
                    "approved": approved,
                })
        finally:
            async with self._approvals_lock:
                self.pending_approvals.pop(request_id, None)
        return bool(approved)

    async def resolve_approval(self, request_id: str, approved: bool) -> bool:
        """Called from the WS inbound loop when a client replies. Returns
        True if the request was still pending (i.e. we resolved a future),
        False if it had already timed out / been resolved by someone else."""
        async with self._approvals_lock:
            fut = self.pending_approvals.get(request_id)
        if fut is None or fut.done():
            return False
        fut.set_result(bool(approved))
        return True


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def build_app(paths: ForgePaths) -> FastAPI:
    state = _ServerState(paths)
    app = FastAPI(title="Forge backend", version="0.0.1")
    # Loopback-only by CLI flag, but CORS is still permissive for the
    # Electron renderer (which loads from file://). Same-host only.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root() -> dict[str, Any]:
        """Browser landing — no SPA; Electron loads ``file://…/renderer/index.html``."""
        p = int(state.cfg.ui.server_port)
        return {
            "service": "forge",
            "hint": (
                "This port is JSON API + WebSocket only. "
                "Run the Electron app (apps/forge/electron npm run start) for the GUI."
            ),
            "repo_root": str(state.paths.repo_root),
            "api": {"health": "/api/health", "chat": "POST /api/chat", "ws": "/ws"},
            "listening_on": p,
        }

    # ------------------------------------------------------------------ health

    @app.get("/api/health")
    async def health() -> dict:
        return {
            "ok": True,
            "repo_root": str(state.paths.repo_root),
            "version": "0.0.1",
            "ts": datetime.now(timezone.utc).isoformat(),
            "engine_started": state.engine is not None,
        }

    # --------------------------------------------------------------- workspace

    @app.get("/api/workspace")
    async def workspace() -> dict:
        """Lightweight workspace header for the Electron app's top strip.

        Returns ``{repo_root, branch, dirty, head}`` so the UI can render the
        "you're in <repo> @ <branch>" line and a dirty-tree dot without
        booting the engine or paying the trace startup cost.

        Non-repos return ``branch=None`` / ``dirty=False`` instead of erroring.
        """
        return await asyncio.to_thread(_workspace_snapshot, state.paths.repo_root)

    # ------------------------------------------------------------------ config

    @app.get("/api/config")
    async def get_config() -> dict:
        return state.cfg.model_dump(mode="json")

    @app.get("/api/config/schema")
    async def get_config_schema() -> dict:
        """JSON Schema for ForgeConfig — the Electron settings tab generates
        its form straight from this. The ``live`` extra in each field tells
        the UI which fields are hot-reloadable vs need a restart."""
        return ForgeConfig.model_json_schema()

    @app.put("/api/config")
    async def put_config(payload: dict) -> dict:
        try:
            new_cfg = ForgeConfig.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid config: {exc}")
        try:
            write_config(state.paths, new_cfg)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"write failed: {exc}")

        # Detect what actually changed so we can decide whether to rebuild
        # the live agent. The previous behavior wrote the new cfg to disk
        # but never propagated it into the running engine, so model swaps
        # silently did nothing until the next ``forge serve`` restart.
        old_cfg = state.cfg
        old_models = old_cfg.models.model_dump()
        new_models = new_cfg.models.model_dump()
        models_changed = old_models != new_models
        changed_model_fields = sorted(
            k for k in new_models if old_models.get(k) != new_models.get(k)
        )

        state.cfg = new_cfg
        rebuilt = False
        if state.engine is not None:
            # Engine owns its own ``cfg`` reference (passed by value at
            # ``ForgeEngine.start``). Keep them in sync so the next turn
            # uses the latest knobs even for "live=True" fields like
            # ``memory.semantic_k`` that don't require a rebuild.
            state.engine.cfg = new_cfg
            if models_changed:
                # The main agent is a ``create_agent`` instance built
                # around ``get_llm(cfg.models.default_agent)`` — the LLM
                # is captured at build time. Rebuild so the next turn
                # actually uses the new slug.
                try:
                    await state.engine.rebuild_main()
                    rebuilt = True
                except Exception as exc:  # noqa: BLE001
                    # Don't fail the save — the config is on disk and a
                    # restart will pick it up. Just surface the issue.
                    raise HTTPException(
                        status_code=500,
                        detail=f"config saved but agent rebuild failed: {exc}",
                    )
        return {
            "ok": True,
            "config": new_cfg.model_dump(mode="json"),
            "agent_rebuilt": rebuilt,
            "changed_model_fields": changed_model_fields,
        }

    # ----------------------------------------------------------- persistent agents

    @app.get("/api/agents")
    async def list_agents() -> list[dict]:
        from .agent.agents_registry import load_persistent_agents
        rows = []
        for entry in load_persistent_agents(state.paths):
            rows.append({
                "name": entry.spec.name,
                "description": entry.spec.description,
                "model": entry.spec.model,
                "tools": list(entry.spec.tools),
                "system_prompt": entry.spec.system_prompt,
                "toml_path": str(entry.toml_path),
            })
        return rows

    @app.put("/api/agents/{name}")
    async def put_agent(name: str, payload: dict) -> dict:
        from .agent.agents_registry import (
            PersistentAgentSpec,
            write_persistent_agent,
        )
        payload = {**payload, "name": name}
        try:
            spec = PersistentAgentSpec.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid agent: {exc}")
        target = write_persistent_agent(state.paths, spec)
        if state.engine is not None:
            state.engine.broker.set_persistent_allowlist(spec.name, spec.tools)
            # Rebuild the main agent so its tool list picks up the new
            # ``delegate_to_<name>`` tool without an engine restart.
            await state.engine.rebuild_main()
        return {"ok": True, "toml_path": str(target)}

    @app.delete("/api/agents/{name}")
    async def delete_agent(name: str) -> dict:
        from .agent.agents_registry import delete_persistent_agent
        removed = delete_persistent_agent(state.paths, name)
        if not removed:
            raise HTTPException(status_code=404, detail=f"agent {name} not found")
        if state.engine is not None:
            state.engine.broker.persistent_allowlists.pop(name, None)
            # Drop the now-stale ``delegate_to_<name>`` from the main agent.
            await state.engine.rebuild_main()
        return {"ok": True}

    # ----------------------------------------------------------------- tools

    @app.get("/api/tools")
    async def list_tools() -> dict:
        """Return the live tool inventory grouped by originating MCP server.

        Used by the Agents tab's tools picker so users don't have to type
        raw tool names. Response shape:

            {"servers": [{"name": "fs", "tools": [{"name": "fs_read",
              "description": "..."}, ...]}, ...]}
        """
        engine = await state.ensure_engine()
        loaded = engine.loaded_tools
        if loaded is None:
            return {"servers": []}
        groups: dict[str, list[dict]] = {}
        for row in loaded.inventory():
            server = row.get("server") or "other"
            groups.setdefault(server, []).append({
                "name": row["name"],
                "description": row.get("description", ""),
            })
        servers = [
            {"name": s, "tools": sorted(ts, key=lambda r: r["name"])}
            for s, ts in sorted(groups.items())
        ]
        return {"servers": servers}

    # ------------------------------------------------------------------ memory

    @app.get("/api/memory/semantic")
    async def memory_semantic(q: str | None = None, k: int = 10) -> list[dict]:
        engine = await state.ensure_engine()
        stores = engine.stores
        if q:
            # Search results stay similarity-ranked — sorting them by date
            # would defeat the point of the query box.
            hits = stores.semantic.search(q, k=max(1, min(int(k), 50)))
            return [
                {"text": h.text, "score": float(h.score),
                 "thread_id": getattr(h, "thread_id", ""),
                 "created_at": getattr(h, "created_at", ""),
                 "id": getattr(h, "id", "")}
                for h in hits
            ]
        # No query => browse mode. Newest first so the Memory tab shows
        # the most recently captured fact at the top.
        rows = stores.semantic.all(limit=max(1, min(int(k), 200)))
        rows = sorted(
            rows, key=lambda r: getattr(r, "created_at", "") or "", reverse=True,
        )
        return [
            {"text": r.text, "score": 0.0,
             "thread_id": getattr(r, "thread_id", ""),
             "created_at": getattr(r, "created_at", ""),
             "id": getattr(r, "id", "")}
            for r in rows
        ]

    @app.delete("/api/memory/semantic/{record_id}")
    async def memory_semantic_delete(record_id: str) -> dict:
        """Delete one semantic memory by ``id``.

        Returns ``{"ok": True}`` on a successful removal, ``404`` if the
        store doesn't acknowledge the deletion. The store is a thin
        wrapper around Chroma so the call also drops the row's vector
        embedding, not just the metadata.
        """
        if not record_id.strip():
            raise HTTPException(status_code=400, detail="record_id is required")
        engine = await state.ensure_engine()
        ok = engine.stores.semantic.delete(record_id)
        if not ok:
            raise HTTPException(status_code=404, detail="not found")
        return {"ok": True, "id": record_id}

    @app.get("/api/memory/episodic")
    async def memory_episodic(limit: int = 20) -> list[dict]:
        engine = await state.ensure_engine()
        rows = engine.stores.episodic.all(limit=max(1, min(int(limit), 500)))
        # Sort by ``updated_at`` desc so recently-refined episodes float
        # to the top (reflection upserts one row per thread per turn).
        # ``updated_at`` falls back to ``created_at`` for legacy rows
        # written before the upsert path landed.
        rows = sorted(
            rows,
            key=lambda r: (r.updated_at or r.created_at or ""),
            reverse=True,
        )
        return [
            {"summary": r.summary, "thread_id": r.thread_id,
             "score": float(r.score), "created_at": r.created_at,
             "updated_at": r.updated_at or r.created_at,
             "id": getattr(r, "id", "")}
            for r in rows
        ]

    @app.get("/api/memory/procedural")
    async def memory_procedural() -> list[dict]:
        engine = await state.ensure_engine()
        rows = engine.stores.procedural.all()
        # Newest first. The store-level ``all()`` returns score-desc order,
        # which is fine for prompt injection ranking but not for browsing.
        rows = sorted(
            rows, key=lambda r: r.created_at or "", reverse=True,
        )
        return [
            {"name": r.name, "fragment": r.fragment,
             "when_to_use": r.when_to_use, "usage_count": int(r.usage_count),
             "score": float(r.score), "created_at": r.created_at}
            for r in rows
        ]

    # ------------------------------------------------------------------ chat

    @app.post("/api/chat")
    async def chat(payload: dict) -> dict:
        message = (payload or {}).get("message")
        if not isinstance(message, str) or not message.strip():
            raise HTTPException(status_code=400, detail="message is required")
        thread_id = (payload or {}).get("thread_id")  # optional resume key
        plan_mode = bool((payload or {}).get("plan_mode", False))
        engine = await state.ensure_engine()
        try:
            result = await engine.run_task(
                message, thread_id=thread_id, plan_mode=plan_mode,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=repr(exc))
        return {
            "thread_id": result.task_id,
            "topology": result.topology,
            "planned": bool(result.planned),
            "answer": result.answer,
        }

    # ---------------------------------------------------------- trace / audit

    @app.get("/api/trace", response_class=PlainTextResponse)
    async def trace_tail(tail: int = 200) -> str:
        return _tail_jsonl(state.paths.trace_jsonl, tail)

    @app.get("/api/audit", response_class=PlainTextResponse)
    async def audit_tail(tail: int = 200) -> str:
        return _tail_jsonl(state.paths.audit_jsonl, tail)

    # ----------------------------------------------------------------- MCP

    @app.get("/api/mcp")
    async def mcp_list() -> dict:
        """List built-in and user-added MCP servers (no engine boot)."""
        from .mcp import list_servers_for_api
        return list_servers_for_api(state.paths)

    @app.post("/api/mcp/validate")
    async def mcp_validate(payload: dict) -> dict:
        """Spawn a candidate MCP server and report its advertised tools.

        Body: ``{contents: str}`` — the raw text of a JSON descriptor
        (``{"command": ..., "args": [...], "env": {...}}``). Legacy
        ``kind: "python"`` payloads are rejected so callers learn about the
        new shape rather than silently falling back.
        Returns ``{ok, tools, error}``.
        """
        from .mcp import validate_server

        contents = (payload or {}).get("contents")
        kind = (payload or {}).get("kind", "json")
        if kind != "json":
            raise HTTPException(
                status_code=400,
                detail=(
                    "only JSON descriptors are supported; the .py install "
                    "path was removed in favor of explicit command + env"
                ),
            )
        if not isinstance(contents, str) or not contents.strip():
            raise HTTPException(status_code=400, detail="contents is required")
        return await validate_server(paths=state.paths, contents=contents)

    @app.post("/api/mcp/install")
    async def mcp_install(payload: dict) -> dict:
        """Validate + persist a new MCP server. Returns the manifest entry.

        Body: ``{name, contents, description?}`` — ``contents`` is a JSON
        descriptor. ``pending_restart`` is always ``True`` in the response —
        the live engine won't pick up the new server until the next
        ``forge serve`` boot.
        """
        from .mcp import install_server

        name = (payload or {}).get("name")
        contents = (payload or {}).get("contents")
        description = (payload or {}).get("description", "")
        kind = (payload or {}).get("kind", "json")
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(status_code=400, detail="name is required")
        if kind != "json":
            raise HTTPException(
                status_code=400,
                detail=(
                    "only JSON descriptors are supported; the .py install "
                    "path was removed in favor of explicit command + env"
                ),
            )
        if not isinstance(contents, str) or not contents.strip():
            raise HTTPException(status_code=400, detail="contents is required")
        try:
            return await install_server(
                paths=state.paths,
                name=name.strip(),
                contents=contents,
                description=str(description or ""),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/mcp/{name}")
    async def mcp_uninstall(name: str) -> dict:
        """Uninstall a user-added MCP server.

        Deletes the descriptor + the manifest entry. The running engine
        still holds the old tools — the response sets ``pending_restart``
        so the UI can prompt the user to hit reload. Built-in servers
        return 400 since they ship with Forge and aren't removable.
        """
        from .mcp import uninstall_server
        try:
            return uninstall_server(paths=state.paths, name=name)
        except ValueError as exc:
            # 400 for "built-in / not found" — same shape as install errors.
            raise HTTPException(status_code=400, detail=str(exc))

    # -------------------------------------------------------------- threads

    @app.get("/api/threads")
    async def list_threads(limit: int = 50) -> dict:
        """List past chat threads (newest first), reconstructed from the
        trace JSONL."""
        rows = _aggregate_chat_threads(
            state.paths.trace_jsonl, limit=max(1, min(limit, 500)),
        )
        return {"threads": rows, "count": len(rows)}

    @app.get("/api/threads/{thread_id}")
    async def get_thread_transcript(thread_id: str) -> dict:
        """Full transcript for one thread: a list of turns, each
        ``{user, assistant, trace[]}``, in chronological order. The trace
        shape matches what the Chat view builds live so the renderer can
        treat history and live turns the same way."""
        turns = _load_thread_transcript(state.paths.trace_jsonl, thread_id)
        if not turns:
            raise HTTPException(status_code=404, detail="thread not found")
        return {"thread_id": thread_id, "turns": turns}

    # ----------------------------------------------------------------- eval

    @app.get("/api/eval/threads")
    async def eval_list_threads(limit: int = 50, offset: int = 0) -> dict:
        """List recent per-thread evals, newest first."""
        from .eval.thread_eval import list_thread_evals

        rows = list_thread_evals(
            state.paths, limit=max(1, min(limit, 500)), offset=max(0, offset),
        )
        return {"evals": rows, "count": len(rows)}

    @app.get("/api/eval/threads/{thread_id}")
    async def eval_get_thread(thread_id: str) -> dict:
        """Most recent eval for a single thread. 404 if no eval has run yet."""
        from .eval.thread_eval import get_thread_eval

        row = get_thread_eval(state.paths, thread_id)
        if row is None:
            raise HTTPException(status_code=404, detail="no eval for thread")
        return row

    @app.delete("/api/eval/threads/{thread_id}")
    async def eval_delete_thread(thread_id: str) -> dict:
        """Remove every stored eval row for one thread. 200 even if there
        was nothing to delete (idempotent so the UI doesn't have to care)."""
        from .eval.thread_eval import delete_thread_eval

        removed = delete_thread_eval(state.paths, thread_id)
        state._broadcast({
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "thread_eval_deleted",
            "task_id": thread_id,
            "removed": removed,
        })
        return {"thread_id": thread_id, "removed": removed}

    @app.post("/api/eval/threads/{thread_id}/run")
    async def eval_run_thread(thread_id: str) -> dict:
        """Re-run the rubrics for one thread (e.g. after switching judge
        model). Synchronous — returns the new eval row when done."""
        from .eval.thread_eval import evaluate_thread

        try:
            rec = await asyncio.to_thread(
                evaluate_thread,
                paths=state.paths, cfg=state.cfg, thread_id=thread_id,
            )
        except RuntimeError as exc:
            # No trace events for that thread (bogus id, or the trace file
            # got cleared). 404 is more honest than 500 here.
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500, detail=f"eval failed: {exc!r}",
            )
        # Broadcast so any open Eval tab refreshes.
        state._broadcast({
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "thread_eval_ready",
            "task_id": thread_id,
            "outcome_overall": (rec.outcome or {}).get("overall"),
            "trajectory_overall": (rec.trajectory or {}).get("overall"),
            "error": rec.error or "",
        })
        return rec.to_jsonable()

    @app.get("/api/eval/rubrics")
    async def eval_rubrics() -> dict:
        """Expose the rubric prompts and config so the UI can show users
        exactly what the judge is being asked. Transparency = trust."""
        from .eval.thread_eval import rubric_prompts

        prompts = rubric_prompts()
        return {
            "prompts": prompts,
            "config": {
                "auto_evaluate_threads": state.cfg.eval.auto_evaluate_threads,
                "outcome_judge_model": (
                    state.cfg.eval.outcome_judge_model or state.cfg.models.judge
                ),
                "trajectory_judge_model": (
                    state.cfg.eval.trajectory_judge_model or state.cfg.models.judge
                ),
            },
        }

    @app.post("/api/mcp/reload")
    async def mcp_reload() -> dict:
        """Tear the engine down and re-boot it so manifest changes apply.

        Bake-at-boot is fine for the steady-state but it makes the install
        UX miserable — a user just installed a server, validated it, and
        the only way to use it is to kill ``forge serve`` and rerun. So we
        offer an in-process reload: shut the engine down, drop the MCP
        client (which kills its stdio subprocesses), then ``ensure_engine``
        on the next read.

        Caveats the UI should know about:

        * Any in-flight chat turn dies. We don't quiesce. The teaching
          claim is "reload between turns, not during one". A future
          refinement would refuse the reload if there are open futures
          on the tracer.
        * WS subscribers stay connected (they're owned by ``state``, not
          the engine), but they'll see a gap in the event stream.
        """
        before = (
            len(state.engine.tools)
            if state.engine is not None
            else 0
        )
        await state.shutdown()
        engine = await state.ensure_engine()
        return {
            "ok": True,
            "tool_count_before": before,
            "tool_count_after": len(engine.tools),
            "tools": [t.name for t in engine.tools],
        }

    # ------------------------------------------------------------ model health

    @app.get("/api/models/health")
    async def model_health(slug: str) -> dict:
        """Probe a model slug with a 1-token ping.

        Useful from the Settings tab to confirm:
        * an OpenRouter slug actually resolves (and the key is set), and
        * an ``ollama/*`` slug can reach the local Ollama server.

        Returns ``{ok, provider, slug, latency_ms, error?}``.
        """
        return await _probe_model(slug)

    @app.get("/api/models/catalog")
    async def model_catalog() -> dict:
        """List the slugs the Settings model picker offers as defaults.

        - ``openrouter``: every slug that has a price-table entry in
          ``shared.openrouter_llm`` (curated list, sorted by vendor).
        - ``ollama``: live ``GET ${OLLAMA_HOST}/api/tags`` so the dropdown
          reflects whatever the user has actually pulled. If Ollama isn't
          running we return an empty list + an ``ollama_error``.
        """
        return await _model_catalog()

    # ------------------------------------------------------------------ WS

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        """Bidirectional event channel.

        Outbound: every tracer event + permission_request broadcast lands on
        this socket as JSON.

        Inbound: the only message types the server reacts to are
        ``{"type": "permission_response", "request_id": str, "approved": bool}``
        and ``{"type": "ping"}``. Anything else is ignored (the UI is free to
        echo unrelated events for its own bookkeeping)."""
        await state.ensure_engine()
        await websocket.accept()
        q = state.add_subscriber()
        await websocket.send_json({
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "ws_hello",
            "repo_root": str(state.paths.repo_root),
        })

        async def _out() -> None:
            while True:
                event = await q.get()
                await websocket.send_json(event)

        async def _in() -> None:
            while True:
                msg = await websocket.receive_json()
                kind = (msg or {}).get("type")
                if kind == "permission_response":
                    rid = msg.get("request_id")
                    approved = bool(msg.get("approved", False))
                    if isinstance(rid, str):
                        await state.resolve_approval(rid, approved)
                # Other inbound types are ignored on purpose — keeps the
                # protocol forward-compatible with whatever UI events the
                # client wants to round-trip for its own state.

        out_task = asyncio.create_task(_out())
        in_task = asyncio.create_task(_in())
        try:
            done, pending = await asyncio.wait(
                {out_task, in_task}, return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, WebSocketDisconnect):
                    raise exc
        except WebSocketDisconnect:
            pass
        finally:
            for task in (out_task, in_task):
                if not task.done():
                    task.cancel()
            state.remove_subscriber(q)

    # ----------------------------------------------------------------- WS chat

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket) -> None:
        """Bidirectional chat socket.

        Lifecycle:

        1. Client connects. Server sends ``ws_hello``.
        2. Client sends ``{type:"chat", message, thread_id?, plan_mode?}``.
        3. Server runs ``engine.run_task`` in a background task. Every tracer
           event broadcast during the turn is forwarded over this socket too
           (same pubsub as ``/ws``). The Electron app uses those events to
           render tool-call cards / agent_spawn chips inline in chat.
        4. When the task finishes, the server sends a single
           ``{type:"chat_result", thread_id, answer, planned}`` message.
           The client may then send another ``chat`` message.
        5. Inbound ``permission_response`` messages are routed to the broker
           the same way ``/ws`` does — so chat can answer its own asks
           without a second socket.
        """
        await state.ensure_engine()
        await websocket.accept()
        q = state.add_subscriber()
        await websocket.send_json({
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "ws_hello",
            "channel": "chat",
            "repo_root": str(state.paths.repo_root),
        })

        # One outstanding turn at a time per socket. The UI is welcome to
        # send a second chat while the first is running; we'll reply with an
        # error rather than racing two engine.run_task calls on the same
        # thread_id.
        active_turn: asyncio.Task | None = None

        async def _send_event(event: dict) -> None:
            await websocket.send_json(event)

        async def _out() -> None:
            while True:
                event = await q.get()
                await _send_event(event)

        async def _run_turn(payload: dict) -> None:
            engine = await state.ensure_engine()
            message = payload.get("message")
            thread_id = payload.get("thread_id")
            plan_mode = bool(payload.get("plan_mode", False))
            try:
                result = await engine.run_task(
                    message, thread_id=thread_id, plan_mode=plan_mode,
                )
            except Exception as exc:  # noqa: BLE001
                await _send_event({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "type": "chat_error",
                    "error": repr(exc),
                })
                return
            await _send_event({
                "ts": datetime.now(timezone.utc).isoformat(),
                "type": "chat_result",
                "thread_id": result.task_id,
                "topology": result.topology,
                "planned": bool(result.planned),
                "answer": result.answer,
            })
            # Broadcast (not just to this socket) so other open Chat tabs
            # — and the sidebar inside the same tab — refresh the thread
            # list without having to poll.
            state._broadcast({
                "ts": datetime.now(timezone.utc).isoformat(),
                "type": "thread_list_changed",
                "thread_id": result.task_id,
            })

        async def _in() -> None:
            nonlocal active_turn
            while True:
                msg = await websocket.receive_json()
                kind = (msg or {}).get("type")
                if kind == "chat":
                    message = msg.get("message")
                    if not isinstance(message, str) or not message.strip():
                        await _send_event({
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "type": "chat_error",
                            "error": "message is required and must be a non-empty string",
                        })
                        continue
                    if active_turn is not None and not active_turn.done():
                        await _send_event({
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "type": "chat_error",
                            "error": "a turn is already in flight on this socket",
                        })
                        continue
                    active_turn = asyncio.create_task(_run_turn(msg))
                elif kind == "permission_response":
                    rid = msg.get("request_id")
                    approved = bool(msg.get("approved", False))
                    if isinstance(rid, str):
                        await state.resolve_approval(rid, approved)
                # Anything else is intentionally ignored.

        out_task = asyncio.create_task(_out())
        in_task = asyncio.create_task(_in())
        try:
            done, _pending = await asyncio.wait(
                {out_task, in_task}, return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, WebSocketDisconnect):
                    raise exc
        except WebSocketDisconnect:
            pass
        finally:
            for task in (out_task, in_task):
                if not task.done():
                    task.cancel()
            if active_turn is not None and not active_turn.done():
                # Let the turn finish on its own — cancelling mid-tool-call
                # would leak MCP state. We just stop forwarding events.
                pass
            state.remove_subscriber(q)

    # ------------------------------------------------------------------ lifecycle

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        await state.shutdown()

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_for_ui(args: dict, limit: int = 512) -> dict:
    """Shrink long string args before they ride a WS event.

    The audit log truncates separately (and at a larger limit); this one is
    just so the permission modal doesn't try to render a 20k-line patch.
    """
    out: dict[str, Any] = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > limit:
            out[k] = v[:limit] + f"... [truncated {len(v) - limit} chars]"
        else:
            out[k] = v
    return out


def _safe_put(q: asyncio.Queue, event: dict) -> None:
    """Drop-oldest queue push, run from the queue's own loop."""
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        try:
            _ = q.get_nowait()
            q.put_nowait(event)
        except Exception:  # noqa: BLE001
            pass


def _workspace_snapshot(repo_root: Path) -> dict[str, Any]:
    """Best-effort git snapshot for the Electron workspace header.

    Uses ``git`` via subprocess (not GitPython) to keep deps thin. All branches
    swallow errors and return safe defaults so a missing/locked git binary or
    a non-repo directory never breaks the UI.
    """
    import subprocess

    out: dict[str, Any] = {
        "repo_root": str(repo_root),
        "branch": None,
        "head": None,
        "dirty": False,
        "is_git": False,
    }
    if not (repo_root / ".git").exists():
        return out

    def _git(*args: str) -> str | None:
        try:
            cp = subprocess.run(
                ["git", *args],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if cp.returncode != 0:
            return None
        return cp.stdout.strip()

    out["is_git"] = True
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        # Detached HEAD — surface short sha as the "branch" so the UI has
        # something to show ("(detached @ abc1234)" is up to the renderer).
        sha = _git("rev-parse", "--short", "HEAD")
        out["branch"] = f"(detached @ {sha})" if sha else "(detached)"
    else:
        out["branch"] = branch
    out["head"] = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain")
    out["dirty"] = bool(status)
    return out


def _tail_jsonl(path: Path, tail: int) -> str:
    if not path.is_file():
        return ""
    tail = max(1, min(int(tail), 5000))
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    return "".join(lines[-tail:])


# ---------------------------------------------------------------------------
# Thread aggregation — fed to the Chat sidebar
# ---------------------------------------------------------------------------
#
# We treat the trace JSONL as the source of truth for past conversations
# (LangGraph's checkpointer SQLite has the raw message arrays, but the
# trace also has tool calls, plans, spawns, etc. — which is exactly what
# we want to surface in the UI). One pass over the file is fine: it's a
# single-user dev tool, the file is small, and adding an index would
# bring complexity we don't need yet.


def _is_chat_thread_id(tid: str) -> bool:
    """Reserved for future non-chat threads — currently all engine threads
    are user chat threads, so this is a passthrough."""
    return bool(tid)


def _aggregate_chat_threads(trace_path: Path, limit: int) -> list[dict]:
    """Walk the trace file once and emit a per-thread summary.

    Each row: ``{thread_id, title, last_ts, first_ts, turns, last_answer,
    ok}``. ``title`` is the first user task we saw (the conversation's
    natural label). ``turns`` counts the number of ``thread_start``
    events — i.e. how many user messages the thread has."""
    if not trace_path.is_file():
        return []

    # Aggregator keyed by thread_id. Order-preserving so the first task
    # we see wins as the title even if later turns have shorter prompts.
    agg: dict[str, dict] = {}
    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = ev.get("task_id")
            if not isinstance(tid, str) or not _is_chat_thread_id(tid):
                continue
            row = agg.get(tid)
            if row is None:
                row = {
                    "thread_id": tid,
                    "title": "",
                    "first_ts": ev.get("ts") or "",
                    "last_ts": ev.get("ts") or "",
                    "turns": 0,
                    "last_answer": "",
                    "ok": True,
                }
                agg[tid] = row
            row["last_ts"] = ev.get("ts") or row["last_ts"]
            t = ev.get("type")
            if t == "thread_start":
                row["turns"] += 1
                if not row["title"]:
                    row["title"] = str(ev.get("task") or "")[:120]
            elif t == "agent_done" and ev.get("agent_name") == "main":
                row["last_answer"] = str(ev.get("result") or "")
            elif t == "thread_end":
                # Latest ok flag wins — a later successful turn shouldn't
                # be poisoned by an earlier failure.
                row["ok"] = bool(ev.get("ok", True))

    threads = list(agg.values())
    threads.sort(key=lambda r: r.get("last_ts") or "", reverse=True)
    return threads[:limit]


def _load_thread_transcript(trace_path: Path, thread_id: str) -> list[dict]:
    """Reconstruct the chat transcript for one thread as a list of turns.

    Each turn: ``{user, assistant, ok, error, trace: [...]}``. ``trace``
    entries use the same discriminated-shape the Chat view builds live
    (kind in ``policy|plan|spawn|agent_done|tool|compaction``), so the
    renderer can treat live and historical turns uniformly.

    Pairing of ``tool_call`` -> ``tool_result`` mirrors the FIFO-by-name
    heuristic the live view uses. Same caveats apply (heavy parallelism
    can blur edges); fine for human consumption."""
    if not trace_path.is_file():
        return []

    turns: list[dict] = []
    cur: dict | None = None
    pending_tools: list[dict] = []  # indices into the current turn's trace

    def _start_turn(task: str) -> dict:
        return {
            "user": task,
            "assistant": "",
            "ok": True,
            "error": "",
            "trace": [],
        }

    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("task_id") != thread_id:
                continue
            t = ev.get("type")

            if t == "thread_start":
                # Flush an unfinished previous turn (rare — engine always
                # emits thread_end, but be defensive against crashes).
                if cur is not None:
                    turns.append(cur)
                cur = _start_turn(str(ev.get("task") or ""))
                pending_tools = []
                continue

            if cur is None:
                # Trace events before the first thread_start — skip.
                continue
            trace = cur["trace"]

            # Reflector events run asynchronously after ``thread_end`` and
            # are shown in the Chat view's side ReflectionPanel, not in
            # the assistant bubble. Skip them during transcript replay
            # so revisiting a thread doesn't contaminate old turns with
            # reflection trace (especially because async timing can land
            # reflector events anywhere between two turns).
            if ev.get("agent_name") == "reflector":
                continue

            if t == "policy_decision":
                trace.append({
                    "kind": "policy",
                    "mode": str(ev.get("mode") or ""),
                    "topology": str(ev.get("topology") or ""),
                    "reason": str(ev.get("reason") or ""),
                })
            elif t == "plan_drafted":
                head = ""
                for line2 in str(ev.get("plan_md") or "").splitlines():
                    if line2.strip():
                        head = line2.strip()[:160]
                        break
                trace.append({"kind": "plan", "head": head})
            elif t == "agent_spawn":
                trace.append({
                    "kind": "spawn",
                    "name": str(ev.get("agent_name") or ""),
                    "agentKind": str(ev.get("kind") or ""),
                })
            elif t == "agent_done":
                name = str(ev.get("agent_name") or "")
                trace.append({"kind": "agent_done", "name": name})
                if name == "main":
                    cur["assistant"] = str(ev.get("result") or cur["assistant"])
            elif t == "tool_call":
                entry = {
                    "kind": "tool",
                    "name": str(ev.get("tool") or ""),
                    "agent": str(ev.get("agent_name") or ""),
                    "args": dict(ev.get("args") or {}),
                    "status": "pending",
                }
                trace.append(entry)
                pending_tools.append(entry)
            elif t == "tool_result":
                tool_name = str(ev.get("tool") or "")
                ok = bool(ev.get("ok", True))
                preview = str(ev.get("preview") or "")
                # FIFO: settle the oldest pending with a matching name.
                for i, e in enumerate(pending_tools):
                    if e["name"] == tool_name:
                        e["status"] = "ok" if ok else "error"
                        e["preview"] = preview
                        pending_tools.pop(i)
                        break
                else:
                    # Result without a call — still record it so the
                    # transcript shows it.
                    trace.append({
                        "kind": "tool",
                        "name": tool_name,
                        "agent": str(ev.get("agent_name") or ""),
                        "args": {},
                        "status": "ok" if ok else "error",
                        "preview": preview,
                    })
            elif t == "compaction_fired":
                trace.append({
                    "kind": "compaction",
                    "strategy": str(ev.get("strategy") or ""),
                })
            elif t == "procedural_triggered":
                trace.append({
                    "kind": "procedural",
                    "skills": list(ev.get("skills") or []),
                    "judgeModel": str(ev.get("judge_model") or ""),
                })
            elif t == "model_in_use":
                trace.append({
                    "kind": "model",
                    "model": str(ev.get("model") or ""),
                    "role": str(ev.get("role") or ""),
                    "summarizer": str(ev.get("summarizer") or ""),
                    "judge": str(ev.get("judge") or ""),
                })
            elif t == "thread_end":
                cur["ok"] = bool(ev.get("ok", True))
                cur["error"] = str(ev.get("error") or ev.get("reason") or "")
                turns.append(cur)
                cur = None
                pending_tools = []

    # Tail flush in case the trace was cut mid-turn.
    if cur is not None:
        turns.append(cur)
    return turns


async def _probe_model(slug: str) -> dict[str, Any]:
    """Build the LLM for ``slug`` and send a 1-token ping.

    Lives next to the other server helpers so the route stays a one-liner.
    Run as ``asyncio.to_thread`` because ``ChatOpenAI.invoke`` /
    ``ChatOllama.invoke`` are sync and blocking on the network.
    """
    import time

    if not isinstance(slug, str) or not slug.strip():
        return {"ok": False, "error": "slug is required"}
    slug = slug.strip()

    try:
        from shared import get_llm
        from shared.ollama_llm import is_ollama_slug
    except ImportError as exc:
        return {"ok": False, "error": f"shared package not importable: {exc}"}

    provider = "ollama" if is_ollama_slug(slug) else "openrouter"

    def _ping() -> tuple[bool, str | None, int]:
        from langchain_core.messages import HumanMessage  # noqa: PLC0415

        t0 = time.time()
        try:
            llm = get_llm(slug, max_tokens=1)
            _ = llm.invoke([HumanMessage(content="ping")])
            return True, None, int((time.time() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}", int((time.time() - t0) * 1000)

    ok, err, latency_ms = await asyncio.to_thread(_ping)
    out: dict[str, Any] = {
        "ok": ok,
        "provider": provider,
        "slug": slug,
        "latency_ms": latency_ms,
    }
    if err:
        out["error"] = err
    return out


async def _model_catalog() -> dict[str, Any]:
    """Assemble the dropdown payload for ``GET /api/models/catalog``.

    Two sources:

    * **OpenRouter** — read straight from the curated ``_PRICE_PER_M_TOKENS``
      table in ``shared.openrouter_llm`` so the UI matches what the
      provider actually understands (and what the cost tracker has prices
      for). Sorted by ``vendor/model``.
    * **Ollama** — probe the local server's ``GET /api/tags`` endpoint.
      We deliberately don't hardcode a list because the user's installed
      models vary; the probe is short-timeout so a missing Ollama doesn't
      block the Settings page.
    """
    openrouter: list[str] = []
    try:
        from shared.openrouter_llm import _PRICE_PER_M_TOKENS  # type: ignore[attr-defined]
        openrouter = sorted(_PRICE_PER_M_TOKENS.keys())
    except Exception as exc:  # noqa: BLE001
        openrouter = []
        _or_err: str | None = f"{type(exc).__name__}: {exc}"
    else:
        _or_err = None

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama: list[str] = []
    ollama_err: str | None = None

    def _probe_ollama() -> tuple[list[str], bool, str | None]:
        try:
            import urllib.error
            import urllib.request

            req = urllib.request.Request(
                f"{ollama_host.rstrip('/')}/api/tags",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            models = payload.get("models") or []
            names: list[str] = []
            for m in models:
                name = (m or {}).get("name") or (m or {}).get("model")
                if isinstance(name, str) and name:
                    names.append(name)
            return sorted(set(names)), True, None
        except urllib.error.URLError as e:
            return [], False, f"ollama unreachable: {e.reason}"
        except Exception as e:  # noqa: BLE001
            return [], False, f"{type(e).__name__}: {e}"

    ollama, ollama_reachable, ollama_err = await asyncio.to_thread(_probe_ollama)

    return {
        "openrouter": openrouter,
        "openrouter_error": _or_err,
        "ollama": ollama,
        "ollama_host": ollama_host,
        # ``available`` = the server is reachable; the dropdown may still
        # be empty (user hasn't pulled any models yet) which the UI shows
        # as a "no models pulled" hint instead of an error.
        "ollama_available": ollama_reachable,
        "ollama_error": ollama_err,
    }


def run_server(*, host: str, port: int, paths: ForgePaths) -> None:
    """Boot the FastAPI server. ``forge serve`` calls this."""
    import uvicorn

    paths.ensure()
    # Stamp the trace file early so the WS subscribers see a startup event.
    ensure_config(paths)
    app = build_app(paths)
    print(f"forge serve: http://{host}:{port}  (repo={paths.repo_root})")
    uvicorn.run(app, host=host, port=port, log_level="info")


__all__ = ["build_app", "run_server"]
