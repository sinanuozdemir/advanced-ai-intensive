"""SDR supervisor: orchestrates the lead-gen / qualifier / email sub-agents.

The supervisor’s tools are:

  - HubSpot MCP tools (read/search contacts, etc.) so you can answer direct
    CRM questions without delegating when the user only wants to inspect data.
  - delegate_to_lead_gen / delegate_to_qualifier / delegate_to_email — each
    enqueues a Celery `process_chat_task` against the matching per-config
    agent and returns the new task_id immediately (fire-and-forget).
  - read_subtask_result — looks up a Celery task's status and (if ready) its
    output, so the supervisor can chain (e.g. feed lead-gen's output into the
    qualifier on a follow-up turn).

Sub-task lifecycle is mirrored into `subtask_log.SubtaskLog` so the UI can
render a live panel for the conversation without polling Celery directly.

Activate by setting `USE_SUPERVISOR=1` (default in `.env.example`).
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from celery_app import process_chat_task, get_task_result
from subtask_log import SubtaskLog
from memory_agent import (
    MemoryAgent,
    MEM_SYSTEM_PROMPT_SUFFIX,
    build_semantic_tools,
    _render_recent_episodic,
)


SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", "anthropic/claude-opus-4.6")

SUB_AGENT_CONFIGS = {
    "lead_gen": "lead_gen_config.json",
    "qualifier": "qualifying_agent.json",
    "email": "email_agent.json",
}


_SUPERVISOR_PROMPT = """\
You are the SDR supervisor. You coordinate a small team of specialist sub-agents
and can inspect HubSpot yourself when the user only needs CRM facts (contacts,
lead status, etc.).

Your team:
- lead_gen: finds prospective leads (universities, professors teaching ML/AI/DS)
  and writes them into HubSpot with research notes. New contacts must be left in
  HubSpot lead status **New**.
- qualifier: for leads in **New**, scores fit for the book; sets **Open** if
  qualified or **Unqualified** if not.
- email: sends first outreach only to leads in **Open**; after a successful
  send, sets lead status to **Connected**.

Your tools:
- HubSpot MCP tools: use these to look up contacts, notes, or pipeline state when
  the user asks "what's in HubSpot", "show me this lead", etc. Do not use them
  to replace whole workflows that belong to the specialists below — still
  delegate heavy lead gen, research scraping, qualification, and email sending.
- delegate_to_lead_gen(query): hand off a lead-generation task. Returns the
  sub-task id. NON-BLOCKING — the sub-task runs in the background.
- delegate_to_qualifier(query): hand off a qualification task. Same shape.
- delegate_to_email(query): hand off an email/outreach task. Same shape.
- read_subtask_result(task_id): look up a sub-task's status and (if ready) its
  output. Use this when the user asks you to chain on a previous sub-task or
  asks for its status.
- search_subtasks(query?, agent_slug?, status?, limit?): search ALL prior
  sub-tasks across past sessions (persistent SQLite log). Use this when the
  user asks about earlier delegations ("what leads did we find last week",
  "did email ever go out for X", etc.). Returns newest-first.

Rules:
- Prefer HubSpot MCP tools for quick read-only questions; delegate when the user
  wants searching the web, bulk creation, qualification, or sending email.
- After delegating, REPLY TO THE USER. Do not call read_subtask_result in the
  same turn just to wait for completion — the UI shows them a live panel of
  every sub-task you dispatched.
- When the user asks you to act on a previous sub-task's output (e.g. "qualify
  the leads you just found"), call read_subtask_result on the relevant task_id
  first, then dispatch the next delegate with the concrete inputs inlined.
- If a request can't be handled by HubSpot lookups or your three sub-agents, say so plainly. Do
  not invent tools.
- Be concise. Tell the user which sub-agents you delegated to and the task_ids
  so they can match them against the panel.
"""


class SupervisorAgent(MemoryAgent):
    """Inherits MemoryAgent's memory + checkpointing; HubSpot MCP + delegate tools."""

    def __init__(self, config_path: str = "supervisor"):
        super().__init__(config_path=config_path, memory_scope="supervisor")
        self.config = {
            "name": "SDR Supervisor",
            "description": "Orchestrates lead-gen, qualifier, and email sub-agents.",
            "agent_settings": {
                "model": SUPERVISOR_MODEL,
                "temperature": 0.2,
                "max_tokens": 4000,
            },
            "mcp_servers": {
                "hubspot-server": {
                    "enabled": True,
                    "description": "HubSpot CRM read/write for supervisor lookups",
                    "command": "docker",
                    "args": [
                        "exec",
                        "-i",
                        "hubspot-mcp-server",
                        "python",
                        "/app/mcp_server.py",
                    ],
                    "transport": "stdio",
                },
            },
        }
        self._subtask_log = SubtaskLog()
        self._active_conversation_id: str | None = None

    def load_config(self) -> Dict[str, Any]:
        return self.config

    def _get_system_prompt(self) -> str:
        parts: list[str] = [_SUPERVISOR_PROMPT]
        skills_block = self.procedural.render_for_system_prompt(n=5)
        if skills_block:
            parts.append(skills_block)
        episodic_block = _render_recent_episodic(self.episodic, n=5)
        if episodic_block:
            parts.append(episodic_block)
        parts.append(MEM_SYSTEM_PROMPT_SUFFIX)
        return "\n\n".join(parts)

    def _build_delegate_tool(self, slug: str, config_path: str):
        log = self._subtask_log
        # Pull the active conversation_id at *call* time — the same supervisor
        # instance is reused across chats, but each delegate invocation needs
        # to be recorded against the chat that triggered it.
        agent = self

        @tool(
            f"delegate_to_{slug}",
            description=(
                f"Delegate a {slug} task to the {slug} sub-agent. "
                "Returns immediately with the Celery task_id; the sub-task runs "
                "in the background. Args: query (str) — plain-language "
                "instruction for the sub-agent."
            ),
        )
        def _delegate(query: str) -> str:
            conv_id = agent._active_conversation_id or "default"
            sub_thread_id = f"{conv_id}--{slug}--{uuid.uuid4().hex[:8]}"
            try:
                task = process_chat_task.delay(query, sub_thread_id, config_path)
                log.record(
                    task_id=task.id,
                    conversation_id=conv_id,
                    agent_slug=slug,
                    agent_config=config_path,
                    message=query,
                    status="PENDING",
                    thread_id=sub_thread_id,
                )
                return f"Dispatched {slug} sub-task. task_id={task.id}"
            except Exception as exc:  # noqa: BLE001
                return f"Failed to dispatch {slug} sub-task: {exc}"

        return _delegate

    def _build_read_tool(self):
        log = self._subtask_log

        @tool(
            "read_subtask_result",
            description=(
                "Look up the status and (if ready) result of a previously "
                "dispatched sub-task. Args: task_id (str) — the id returned "
                "by a delegate_* tool. Returns one of: 'PENDING', "
                "'PROCESSING', 'SUCCESS\\n<output>', 'FAILURE\\n<error>'."
            ),
        )
        def _read(task_id: str) -> str:
            r = get_task_result(task_id)
            state = r.get("state", "UNKNOWN")
            if state == "SUCCESS":
                payload = r.get("result") or {}
                response = payload.get("response", "") if isinstance(payload, dict) else ""
                log.update_status(task_id, "SUCCESS", result_summary=response)
                return f"SUCCESS\n{response}"
            if state in ("FAILURE", "ERROR"):
                err = r.get("error")
                if not err:
                    payload = r.get("result") or {}
                    err = payload.get("error", "unknown") if isinstance(payload, dict) else "unknown"
                log.update_status(task_id, "FAILURE", result_summary=str(err))
                return f"FAILURE\n{err}"
            log.update_status(task_id, state)
            return state

        return _read

    def _build_search_subtasks_tool(self):
        log = self._subtask_log

        @tool(
            "search_subtasks",
            description=(
                "Search past sub-tasks across ALL conversations (persistent). "
                "Use this when the user asks about prior delegations, "
                "dispatched leads, or earlier sessions. Args: "
                "query (str, optional) — substring matched against the "
                "original sub-task message and the result_summary; "
                "agent_slug (str, optional) — one of 'lead_gen', 'qualifier', "
                "'email'; status (str, optional) — 'PENDING'/'PROCESSING'/"
                "'SUCCESS'/'FAILURE'; limit (int, optional, default 10). "
                "Returns up to `limit` rows newest-first as compact text."
            ),
        )
        def _search(
            query: str = "",
            agent_slug: str = "",
            status: str = "",
            limit: int = 10,
        ) -> str:
            rows = log.search(
                query=query or None,
                agent_slug=agent_slug or None,
                status=status or None,
                limit=max(1, min(int(limit or 10), 50)),
            )
            # Reconcile any non-terminal rows against Celery so we don't
            # report PENDING/PROCESSING for tasks that already finished in
            # another conversation. The /api/subtasks endpoint does the
            # same thing on every poll for the active conversation.
            for r in rows:
                if r.get("status") not in ("PENDING", "PROCESSING"):
                    continue
                try:
                    cel = get_task_result(r["task_id"])
                except Exception:
                    continue
                state = cel.get("state", r["status"])
                if state == r["status"]:
                    continue
                summary = None
                if state == "SUCCESS":
                    payload = cel.get("result") or {}
                    if isinstance(payload, dict):
                        summary = payload.get("response", "")
                elif state in ("FAILURE", "ERROR"):
                    summary = cel.get("error")
                    if not summary:
                        payload = cel.get("result") or {}
                        if isinstance(payload, dict):
                            summary = payload.get("error", "unknown")
                log.update_status(r["task_id"], state, result_summary=summary)
                r["status"] = state
                if summary is not None:
                    r["result_summary"] = summary
            if not rows:
                return "(no matching sub-tasks)"
            out: list[str] = []
            for r in rows:
                summary = (r.get("result_summary") or "").replace("\n", " ")
                if len(summary) > 240:
                    summary = summary[:240] + "…"
                msg = (r.get("message") or "").replace("\n", " ")
                if len(msg) > 200:
                    msg = msg[:200] + "…"
                out.append(
                    f"- [{r['created_at']}] {r['agent_slug']} {r['status']} "
                    f"task_id={r['task_id']} conv={r['conversation_id']}\n"
                    f"  prompt: {msg}\n"
                    f"  result: {summary or '(none yet)'}"
                )
            return "\n".join(out)

        return _search

    async def initialize(self):
        if self.initialized:
            return

        await self._ensure_checkpointer()

        enabled_servers = self.get_enabled_mcp_servers()
        if not enabled_servers:
            raise ValueError("Supervisor requires at least the HubSpot MCP server")

        print("🔌 Supervisor: connecting HubSpot MCP…")
        self.mcp_client = MultiServerMCPClient(enabled_servers)
        hubspot_tools = list(await self.mcp_client.get_tools())
        print(f"✅ Supervisor: loaded {len(hubspot_tools)} HubSpot MCP tools")

        memory_tools = build_semantic_tools(
            self.semantic,
            lambda: self._active_conversation_id or "default",
        )
        delegate_tools = [
            self._build_delegate_tool(slug, cfg)
            for slug, cfg in SUB_AGENT_CONFIGS.items()
        ]
        self.tools = [
            *hubspot_tools,
            *memory_tools,
            *delegate_tools,
            self._build_read_tool(),
            self._build_search_subtasks_tool(),
        ]

        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model=SUPERVISOR_MODEL,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=self.config["agent_settings"]["temperature"],
            max_tokens=self.config["agent_settings"]["max_tokens"],
        )
        prompt = self._get_system_prompt()
        self.agent = create_agent(
            llm,
            self.tools,
            system_prompt=prompt,
            checkpointer=self.checkpointer,
        )
        self.initialized = True
        print(
            f"✅ SDR Supervisor initialized (model={SUPERVISOR_MODEL}, "
            f"{len(hubspot_tools)} HubSpot tools + {len(memory_tools)} memory tools + delegates)"
        )

    async def chat(self, message: str, conversation_id: str = None) -> Dict[str, Any]:
        # Tools read this attribute to attribute their dispatches to the
        # correct conversation_id.
        self._active_conversation_id = conversation_id or "default"
        try:
            return await super().chat(message, conversation_id)
        finally:
            self._active_conversation_id = None
