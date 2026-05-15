"""Forge engine — the top-level "run a task" entry point.

Wires the pieces together once at startup and exposes ``ForgeEngine.run_task``
to the TUI, the FastAPI server, and the eval harness.

Lifecycle:

    engine = await ForgeEngine.start(paths, cfg)         # boot once
    result = await engine.run_task("write a hello.py")    # per-turn
    await engine.shutdown()
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..config import ForgeConfig
from ..memory import MemoryStores, build_memory_tools
from ..paths import ForgePaths
from ..trace import Tracer
from .permissions import PermissionBroker
from .solo import ForgeSoloAgent, build_forge_solo

_log = logging.getLogger("forge.engine")


def _write_tool_snapshot(paths, loaded, cfg) -> None:
    """Write ``.forge/loaded_tools.json`` for out-of-process consumers.

    Includes every gated tool name plus its static gate decision
    (allow/ask/deny). Kept around because future out-of-process MCP
    servers may want to introspect what's loaded without re-spawning the
    full stack.
    """
    import json
    from datetime import datetime, timezone

    rows = []
    for t in loaded.tools:
        gate = cfg.permissions.tools.get(t.name) or cfg.permissions.default
        rows.append({"name": t.name, "gate": gate})
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tools": rows,
    }
    target = paths.forge_dir / "loaded_tools.json"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    except OSError:
        pass


@dataclass
class TaskResult:
    task_id: str
    answer: str
    # Always ``"main"`` since the supervisor topology was collapsed into
    # the main agent (workers became delegate tools). Kept for back-compat
    # with the eval row schema and the Electron UI's topology pill.
    topology: str = "main"
    messages: list[Any] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    planned: bool = False
    plan_md: str | None = None


@dataclass
class ForgeEngine:
    paths: ForgePaths
    cfg: ForgeConfig
    tracer: Tracer
    broker: PermissionBroker
    stores: MemoryStores
    mcp_client: Any
    tools: list[Any]
    memory_tools: list[Any]
    loaded_tools: Any = None  # forge.mcp.LoadedTools
    # The one and only Forge agent — built at engine start with the full
    # tool list (MCP + memory + delegate_to_* + spawn). Rebuilt on demand
    # via ``rebuild_main`` when persistent agents change between turns.
    solo: ForgeSoloAgent | None = None
    _semantic_seeded_threads: set[str] = field(default_factory=set)

    @classmethod
    async def start(
        cls,
        *,
        paths: ForgePaths,
        cfg: ForgeConfig,
        approver: Any | None = None,
    ) -> "ForgeEngine":
        paths.ensure()
        tracer = Tracer(paths.trace_jsonl, enabled=cfg.trace.enabled)
        broker = PermissionBroker(paths=paths, cfg=cfg.permissions)
        broker.set_approver(approver)
        stores = MemoryStores.for_paths(paths)
        # Boot MCP servers.
        from ..mcp import load_mcp_tools
        loaded = await load_mcp_tools(
            paths=paths, broker=broker, agent_name="main", tracer=tracer,
        )
        memory_tools = build_memory_tools(
            stores, tracer=tracer, thread_id="boot",
            semantic_k=cfg.memory.semantic_k,
        ) if cfg.memory.enabled else []
        # Register each persistent agent's tool allowlist with the broker so
        # the "deny-by-default for non-main, non-ephemeral" policy can match.
        from .agents_registry import load_persistent_agents
        for entry in load_persistent_agents(paths):
            broker.set_persistent_allowlist(entry.spec.name, entry.spec.tools)
        # Stamp a JSON snapshot of the loaded toolset so out-of-process MCP
        # servers can sanity-check tool names without re-spawning the whole
        # MCP stack.
        _write_tool_snapshot(paths, loaded, cfg)
        engine = cls(
            paths=paths, cfg=cfg, tracer=tracer, broker=broker, stores=stores,
            mcp_client=loaded.client, tools=loaded.tools,
            loaded_tools=loaded, memory_tools=memory_tools,
        )
        engine.solo = await build_forge_solo(
            paths=paths, cfg=cfg, tools=loaded.tools,
            memory_tools=memory_tools, tracer=tracer,
            loaded_tools=loaded,
        )
        return engine

    async def rebuild_main(self, *, plan_mode: bool = False) -> None:
        """Rebuild the main agent. Called when persistent agents change
        between turns so newly-saved ``delegate_to_<name>`` tools become
        visible without restarting the engine. Also used by the next-turn
        plan-mode toggle (re-mounts the system prompt addendum)."""
        self.solo = await build_forge_solo(
            paths=self.paths, cfg=self.cfg, tools=self.tools,
            memory_tools=self.memory_tools, tracer=self.tracer,
            loaded_tools=self.loaded_tools, plan_mode=plan_mode,
        )

    async def run_task(
        self,
        task: str,
        *,
        thread_id: str | None = None,
        plan_mode: bool = False,
        history: list[Any] | None = None,
        reflect: bool = True,
    ) -> TaskResult:
        """Run a task end-to-end through the main agent.

        Args:
            plan_mode: When True, the main agent's system prompt is
                augmented with an addendum nudging it to call
                ``delegate_to_planner`` before any write-class tool.
                Maps to the old ``mode="plan"`` flag — see
                :data:`forge.agent.prompts.PLAN_MODE_ADDENDUM`.
            reflect: When True (eval / single-shot callers), episodic /
                procedural reflection runs if
                ``memory.reflect_on_thread_end`` is set. The TUI sets
                False and calls :meth:`reflect_conversation` once on
                ``/done`` so short chat turns don't each burn a
                summarizer pass.
        """
        task_id = thread_id or f"t-{uuid.uuid4().hex[:10]}"
        from ..trace import current_task
        with current_task(task_id):
            return await self._run_inner(
                task=task, task_id=task_id, plan_mode=plan_mode,
                history=history, reflect=reflect,
            )

    async def _run_inner(
        self,
        *,
        task: str,
        task_id: str,
        plan_mode: bool,
        history: list[Any] | None,
        reflect: bool,
    ) -> TaskResult:
        del history  # main-agent loop owns its own history via checkpointer
        if plan_mode:
            # Plan-mode is a per-turn behavioral hint, not a separate
            # topology. Rebuild the agent with the addendum mounted.
            await self.rebuild_main(plan_mode=True)
        self.tracer.emit(
            "thread_start", task_id=task_id, task=task, topology="main",
        )
        # Echo which model is actually driving this turn. Critical for
        # diagnosing "I switched the model and nothing changed" — both
        # the server log and the trace stream now name the live slug.
        active_model = self.cfg.models.default_agent
        self.tracer.emit(
            "model_in_use", task_id=task_id,
            model=active_model, role="main_agent",
            summarizer=self.cfg.models.summarizer,
            judge=self.cfg.models.judge,
        )
        _log.info(
            "turn start: task_id=%s main_agent=%s summarizer=%s judge=%s plan_mode=%s",
            task_id, active_model, self.cfg.models.summarizer,
            self.cfg.models.judge, plan_mode,
        )
        try:
            self.memory_tools = build_memory_tools(
                self.stores, tracer=self.tracer, thread_id=task_id,
                semantic_k=self.cfg.memory.semantic_k,
            ) if self.cfg.memory.enabled else []
            executable_task = task
            pfx = self._semantic_thread_start_seed(task_id)
            if pfx:
                executable_task = pfx + executable_task
            skill_pfx = self._procedural_skill_preamble(task, task_id=task_id)
            if skill_pfx:
                executable_task = skill_pfx + executable_task
            assert self.solo is not None
            answer, msgs = await self.solo.ainvoke(
                executable_task, thread_id=task_id,
            )
            result = TaskResult(
                task_id=task_id, answer=answer, topology="main",
                messages=msgs, planned=plan_mode,
            )
            # Emit ``thread_end`` BEFORE reflection so the chat_result
            # frame returns to the UI immediately. Reflection then runs
            # in the background and emits its own ``agent_spawn`` /
            # ``tool_call`` events that the UI routes into a side panel
            # (any event with ``agent_name == "reflector"``).
            self.tracer.emit("thread_end", task_id=task_id, ok=True)
            if reflect:
                self._maybe_reflect(task_id=task_id, result=result)
            self._maybe_schedule_thread_eval(task_id)
            return result
        except Exception as exc:  # noqa: BLE001
            self.tracer.emit(
                "thread_end", task_id=task_id, ok=False, error=repr(exc),
            )
            raise
        finally:
            if plan_mode:
                # Drop the addendum so the next turn doesn't inherit it.
                await self.rebuild_main(plan_mode=False)

    def _maybe_schedule_thread_eval(self, thread_id: str) -> None:
        """Fire-and-forget the per-thread eval. The eval reads the trace
        from disk, calls two LLMs, and appends to thread_evals.jsonl. We
        don't await it: the chat turn should return as soon as the answer
        is ready, and the eval is purely observational.

        Two guard rails to avoid grading turns that aren't worth grading
        (each one cost two LLM calls, so noise hurts):

        * **Degenerate-turn skip** — if the thread has no final answer AND
          no tool calls at all, there's nothing to judge. This catches the
          "idk just try to fid it" case where the user sent a vague
          one-liner and the agent didn't really do anything yet.
        * **No-progress skip** — if the most recent stored eval for this
          thread already covered the current ``(final_answer, n_tools)``
          pair, re-eval'ing would just produce a duplicate row. We skip.

        Failures are caught and emitted as a ``thread_eval_failed`` trace
        event so they show up in the UI without nuking the chat path.
        """
        if not self.cfg.eval.auto_evaluate_threads:
            return

        skip_reason = self._eval_skip_reason(thread_id)
        if skip_reason:
            # Emit a low-noise trace so the UI can show "skipped: <why>" if
            # we ever surface it. For now it's mainly for debug visibility.
            self.tracer.emit(
                "thread_eval_skipped",
                task_id=thread_id,
                reason=skip_reason,
            )
            return

        async def _run() -> None:
            from ..eval.thread_eval import evaluate_thread

            try:
                rec = await asyncio.to_thread(
                    evaluate_thread,
                    paths=self.paths, cfg=self.cfg, thread_id=thread_id,
                )
                self.tracer.emit(
                    "thread_eval_ready",
                    task_id=thread_id,
                    outcome_overall=(rec.outcome or {}).get("overall"),
                    trajectory_overall=(rec.trajectory or {}).get("overall"),
                    error=rec.error or "",
                )
            except Exception as exc:  # noqa: BLE001
                self.tracer.emit(
                    "thread_eval_failed",
                    task_id=thread_id,
                    error=repr(exc),
                )

        try:
            asyncio.get_running_loop().create_task(_run())
        except RuntimeError:
            # No running loop (eval runner / TUI batch mode). Run inline
            # so the caller can still trigger evals in non-async contexts;
            # synchronous fallback is rare for chat turns.
            asyncio.run(_run())

    def _eval_skip_reason(self, thread_id: str) -> str:
        """Return a short reason string when the thread isn't worth eval'ing
        right now, or an empty string when it is."""
        from ..eval.thread_eval import (
            get_thread_eval,
            load_thread_events,
            thread_summary_from_events,
        )

        events = load_thread_events(self.paths, thread_id)
        if not events:
            return "no events for thread"
        summary = thread_summary_from_events(events)
        answer = (summary.get("final_answer") or "").strip()
        trajectory = summary.get("trajectory") or []
        if not answer and not trajectory:
            return "no answer and no tool calls"

        # Dedup: did the most recent stored eval already cover this state?
        last = get_thread_eval(self.paths, thread_id)
        if last is not None:
            prev_answer = (last.get("final_answer") or "").strip()
            prev_tool_calls = last.get("tool_calls") or []
            if prev_answer == answer and len(prev_tool_calls) == len(trajectory):
                return "thread state unchanged since last eval"
        return ""

    def _semantic_thread_start_seed(self, task_id: str) -> str:
        """Once per ``thread_id``, prepend a memory block to the executable task.

        Two tiers are injected, in priority order:
          1. Semantic memories matching a broad "durable user/project facts" query.
          2. The N most recent episodic summaries (so the agent knows what we
             last talked about without the user having to paste history).

        Procedural skills are NOT injected here — they fire just-in-time
        per turn via :meth:`_procedural_skill_preamble`, gated by a cosine
        similarity threshold against each skill's ``when_to_use`` cue.

        Later turns in the same thread get no automatic semantic/episodic
        injection — the model is expected to call ``semantic_read`` when it
        needs narrower recall.
        """
        mc = self.cfg.memory
        if not mc.enabled:
            return ""
        if task_id in self._semantic_seeded_threads:
            return ""
        self._semantic_seeded_threads.add(task_id)

        sections: list[str] = []

        # Tier 1: broad semantic recall
        if mc.semantic_thread_start_k > 0:
            k = mc.semantic_thread_start_k
            broad = (
                "User preferences identity habits durable facts about the user "
                "or this project worth remembering across sessions"
            )
            try:
                seen: set[str] = set()
                lines: list[str] = []
                for hit in self.stores.semantic.search(broad, k=k):
                    txt = (hit.text or "").strip()
                    if not txt or txt in seen:
                        continue
                    seen.add(txt)
                    lines.append(txt)
                    if len(lines) >= k:
                        break
                if lines:
                    bullets = "\n".join(f"- {t}" for t in lines)
                    sections.append(
                        "#### Semantic memory (durable user/project facts)\n"
                        f"{bullets}"
                    )
                    self.tracer.emit(
                        "memory_read", store="semantic",
                        query="(thread-start seed)", hits=len(lines),
                        source="thread_start_seed",
                    )
            except Exception:  # noqa: BLE001
                pass

        # Tier 2: most recent episodic summaries
        if mc.episodic_k > 0:
            try:
                recent = self.stores.episodic.all(limit=max(mc.episodic_k * 2, 10))
                ep_lines = [
                    f"- ({e.thread_id[:10]}) {e.summary.strip()[:240]}"
                    for e in recent[: mc.episodic_k]
                    if (e.summary or "").strip()
                ]
                if ep_lines:
                    sections.append(
                        "#### Recent episodes (what we worked on lately)\n"
                        + "\n".join(ep_lines)
                    )
                    self.tracer.emit(
                        "memory_read", store="episodic",
                        query="(thread-start seed)", hits=len(ep_lines),
                        source="thread_start_seed",
                    )
            except Exception:  # noqa: BLE001
                pass

        if not sections:
            return ""
        body = "\n\n".join(sections)
        return (
            "### Long-term memory (thread seed — first turn only)\n"
            "These are reminders surfaced from previous sessions. For more or "
            "narrower recall, call `semantic_read` with a focused query.\n\n"
            f"{body}\n\n---\n\n"
        )

    def _procedural_skill_preamble(
        self, user_message: str, *, task_id: str,
    ) -> str:
        """Per-turn just-in-time recall of procedural skills.

        Two-stage gating:
          1. Cheap cosine recall — shortlist the top
             ``memory.procedural_candidate_pool`` skills whose
             ``when_to_use`` cues are most similar to the user's message.
          2. LLM judge — ``models.procedural_judge`` receives every
             candidate and returns ``{reasoning, keep}`` per skill via
             structured output. Only ``keep=True`` skills are injected,
             capped at ``memory.skill_inject_count``.

        Emits a ``procedural_triggered`` event including each kept skill's
        reasoning so the chat UI can render an auditable chip.
        """
        mc = self.cfg.memory
        if not mc.enabled or mc.skill_inject_count <= 0:
            return ""
        text = (user_message or "").strip()
        if not text:
            return ""

        # ---- stage 1: cosine recall ----
        try:
            candidates = self.stores.procedural.search_when(
                text, k=mc.procedural_candidate_pool, min_score=0.0,
            )
        except Exception:  # noqa: BLE001
            return ""
        if not candidates:
            return ""

        # ---- stage 2: LLM relevance judge ----
        verdicts = self._judge_procedural_relevance(text, candidates)
        if not verdicts:
            return ""
        kept = [
            (skill, score, reasoning)
            for (skill, score, reasoning, keep) in verdicts
            if keep
        ][: mc.skill_inject_count]
        if not kept:
            return ""

        skill_lines: list[str] = []
        triggered: list[dict[str, Any]] = []
        for skill, score, reasoning in kept:
            frag = (skill.fragment or "").strip().replace("\n", " ")
            when = (skill.when_to_use or "").strip()
            cue = f" _when: {when}_" if when else ""
            skill_lines.append(
                f"- **{skill.name}** — {frag}{cue}"
            )
            triggered.append({
                "name": skill.name,
                "score": float(score),
                "when_to_use": when,
                "fragment": frag,
                "reasoning": reasoning,
            })

        self.tracer.emit(
            "procedural_triggered",
            task_id=task_id,
            skills=triggered,
            user_message_preview=text[:200],
            judge_model=self.cfg.models.procedural_judge,
        )

        return (
            "### Procedural skill"
            + ("s" if len(skill_lines) > 1 else "")
            + " triggered for this turn\n"
            "These skills were judged relevant to your current message by "
            "the procedural-relevance LLM. Apply them when they fit.\n"
            + "\n".join(skill_lines)
            + "\n\n---\n\n"
        )

    def _judge_procedural_relevance(
        self,
        user_message: str,
        candidates: list[tuple[Any, float]],
    ) -> list[tuple[Any, float, str, bool]]:
        """Ask the procedural-judge LLM whether each shortlisted skill is
        relevant to ``user_message``. Returns ``[(skill, cosine_score,
        reasoning, keep), ...]`` in the original order. On any error the
        function returns an empty list so the engine falls through to "no
        skill injected" rather than dumping irrelevant guidance.
        """
        from pydantic import BaseModel as _PdBase, Field as _PdField

        class _SkillVerdict(_PdBase):
            skill_name: str = _PdField(
                description="The exact `name` of the skill being judged."
            )
            reasoning: str = _PdField(
                description=(
                    "One concise sentence: why this skill is or isn't "
                    "relevant to the user's current message."
                )
            )
            keep: bool = _PdField(
                description=(
                    "True if injecting this skill's fragment into the "
                    "agent's context would meaningfully help this turn. "
                    "Be strict: prefer false when in doubt."
                )
            )

        class _Verdicts(_PdBase):
            verdicts: list[_SkillVerdict]

        bullet_lines = []
        for i, (skill, score) in enumerate(candidates, 1):
            name = skill.name
            when = (skill.when_to_use or "").strip()
            frag = (skill.fragment or "").strip()
            bullet_lines.append(
                f"{i}. name: {name}\n"
                f"   when_to_use: {when}\n"
                f"   fragment: {frag}"
            )
        candidates_blob = "\n\n".join(bullet_lines)

        system_prompt = (
            "You decide which previously-learned procedural skills are "
            "relevant to a user's CURRENT message. For each candidate, "
            "reason briefly about whether the skill's 'when_to_use' cue "
            "applies to this message, then set keep=true ONLY if injecting "
            "the skill's fragment would meaningfully help the agent respond "
            "well. Be strict: prefer keep=false when in doubt. Return one "
            "verdict per input skill, in the input order, using each skill's "
            "exact name."
        )
        user_prompt = (
            f"User's current message:\n\"{user_message[:1500]}\"\n\n"
            f"Candidate skills ({len(candidates)}):\n\n{candidates_blob}"
        )

        try:
            from shared import get_structured_llm
            llm = get_structured_llm(
                self.cfg.models.procedural_judge, _Verdicts,
            )
            res = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
        except Exception as exc:  # noqa: BLE001
            self.tracer.emit(
                "memory_read", store="procedural",
                error=repr(exc), source="judge_failed",
            )
            return []

        by_name = {v.skill_name: v for v in (res.verdicts or [])}
        out: list[tuple[Any, float, str, bool]] = []
        for skill, score in candidates:
            v = by_name.get(skill.name)
            if v is None:
                # Judge omitted this skill; treat as drop.
                out.append((skill, float(score), "", False))
            else:
                out.append(
                    (skill, float(score), v.reasoning, bool(v.keep))
                )
        return out

    def reflect_conversation(
        self, *, thread_id: str, messages: list[Any],
    ) -> None:
        """Run episodic + procedural reflection on an explicit transcript.

        Used by the TUI ``/done`` flow. Synchronous because the TUI's
        ``/done`` handler is itself a discrete user action — the user
        is asking to wait. The Electron chat path uses
        :meth:`_maybe_reflect` instead, which runs the same logic in
        the background so the user-visible chat turn isn't blocked.
        """
        if not self.cfg.memory.enabled or not self.cfg.memory.reflect_on_thread_end:
            return
        if not messages:
            return
        try:
            from ..memory import reflect_main_thread
            reflect_main_thread(
                stores=self.stores, tracer=self.tracer, thread_id=thread_id,
                messages=messages, model_slug=self.cfg.models.summarizer,
            )
        except Exception as exc:  # noqa: BLE001
            self.tracer.emit("memory_write", store="episodic", error=repr(exc))

    def _maybe_reflect(self, *, task_id: str, result: TaskResult) -> None:
        """Fire-and-forget end-of-turn reflection on the MAIN transcript.

        Ephemeral and persistent agents do not get their own reflection
        (deferred per the locked decisions). Reflection runs as a
        background task so the user-visible ``chat_result`` returns the
        instant the main agent finishes — the reflector's
        ``agent_spawn`` / ``tool_call`` / ``agent_done`` events stream
        out the same trace WS the Chat view already subscribes to. The
        Chat view routes ``agent_name="reflector"`` events into a
        side panel instead of the assistant bubble.
        """
        if not result.messages:
            return
        if not self.cfg.memory.enabled or not self.cfg.memory.reflect_on_thread_end:
            return

        # Snapshot the inputs we need so the background task is
        # independent of the engine's mutable state. ``messages`` is
        # already a finalized list owned by the just-completed turn.
        captured_messages = list(result.messages)
        captured_thread_id = task_id

        async def _run() -> None:
            try:
                from ..memory import reflect_main_thread
                await asyncio.to_thread(
                    reflect_main_thread,
                    stores=self.stores,
                    tracer=self.tracer,
                    thread_id=captured_thread_id,
                    messages=captured_messages,
                    model_slug=self.cfg.models.summarizer,
                )
            except Exception as exc:  # noqa: BLE001
                # Mirrors the synchronous path's error reporting so the
                # UI still sees something if reflection blows up.
                self.tracer.emit(
                    "memory_write", store="episodic", error=repr(exc),
                    thread_id=captured_thread_id,
                )

        try:
            asyncio.get_running_loop().create_task(_run())
        except RuntimeError:
            # No running loop (CLI/test contexts) — fall back to the
            # synchronous path so reflection still happens.
            self.reflect_conversation(
                thread_id=task_id, messages=result.messages,
            )

    async def shutdown(self) -> None:
        """Close MCP server subprocesses and the checkpointer connection."""
        try:
            if self.mcp_client is not None and hasattr(self.mcp_client, "__aexit__"):
                await self.mcp_client.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
        if self.solo is not None and hasattr(self.solo.checkpointer, "conn"):
            try:
                await self.solo.checkpointer.conn.close()
            except Exception:  # noqa: BLE001
                pass


__all__ = ["ForgeEngine", "TaskResult"]
