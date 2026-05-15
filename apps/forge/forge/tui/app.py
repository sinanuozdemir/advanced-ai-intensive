"""Forge Textual TUI — minimal companion to the Electron app.

This is a chat-only TUI. Rich observability (audit panel, agent CRUD,
memory, settings) lives in the Electron app (``apps/forge/electron``);
the TUI exists for fast local iteration and for environments where
launching Electron is impractical.

Layout:

    ┌── header (repo) ────────────────────┐
    │ chat transcript                     │
    │   you / forge / (compact trace)     │
    │                                     │
    │ input  > _                          │
    └─ status strip ──────────────────────┘

Slash commands (parsed in ``on_input_submitted``):

    /plan <task>   — run one turn with plan-mode addendum mounted
                     (nudges the agent to call delegate_to_planner first)
    /done          — reflect on the whole session (summarizer → episodic /
                     procedural) and start a fresh session id
    /help          — list commands
    /quit          — exit
"""
from __future__ import annotations

import uuid
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, RichLog, Static

from langchain_core.messages import AIMessage, HumanMessage

from ..agent.engine import ForgeEngine
from ..agent.permissions import ApproverProtocol
from ..config import ForgeConfig
from ..paths import ForgePaths


_FORGE_HELP = """\
Forge TUI — slash commands:
  /plan <task>   one turn with plan-mode addendum (nudges delegate_to_planner first)
  /done          run session-end reflection (memory) and reset session
  /help          this list
  /quit          exit Forge

For audit, agents, memory and settings views, launch the Electron app:
  cd apps/forge/electron && npm run start
"""


class ToolGateScreen(ModalScreen[bool]):
    """Yes/No modal shown when a tool is gated as 'ask'."""

    BINDINGS = [
        Binding("y", "approve", "approve"),
        Binding("n", "reject", "reject"),
        Binding("escape", "reject", "reject"),
    ]

    def __init__(self, *, tool_name: str, args: dict, agent_name: str, reason: str) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.args = args
        self.agent_name = agent_name
        self.reason = reason

    def compose(self) -> ComposeResult:
        import json
        with Vertical():
            yield Static(
                f"[b][yellow]Tool permission requested[/]\n\n"
                f"agent : {self.agent_name}\n"
                f"tool  : {self.tool_name}\n"
                f"reason: {self.reason}\n\n"
                f"args:\n{json.dumps(self.args, indent=2, default=str)[:1200]}",
                id="gate-text",
            )
            with Horizontal():
                yield Static("[Y] allow   [N] deny", id="gate-help")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_reject(self) -> None:
        self.dismiss(False)


class ForgeTUI(App):
    """Slim chat TUI. One transcript, one input, one status strip.

    For audit / plan / agents / memory / settings, use the Electron app —
    this TUI is just for "I'm SSHed in and need to ask Forge something".
    """

    CSS_PATH = "styles.tcss"
    BINDINGS = [
        Binding("ctrl+c", "quit", "quit"),
        Binding("ctrl+l", "clear_chat", "clear chat"),
    ]

    busy = reactive(False)

    def __init__(self, paths: ForgePaths, cfg: ForgeConfig) -> None:
        super().__init__()
        self.paths = paths
        self.cfg = cfg
        self.engine: ForgeEngine | None = None
        self._session_thread_id = f"t-{uuid.uuid4().hex[:10]}"
        self._reflect_buffer: list[Any] = []

    def compose(self) -> ComposeResult:
        yield Header(name=f"Forge · {self.paths.repo_root}")
        chat = RichLog(highlight=True, markup=True, wrap=True, id="chat-log")
        chat.border_title = "transcript"
        yield chat
        yield Input(placeholder="message Forge (or /help)…", id="user-input")
        yield Static("ready · turns=0", id="status-strip")
        yield Footer()

    async def on_mount(self) -> None:
        self._chat = self.query_one("#chat-log", RichLog)
        self._status = self.query_one("#status-strip", Static)
        self._turns = 0
        self._chat.write("[b]Booting Forge…[/]")
        await self._boot_engine()

    async def _boot_engine(self) -> None:
        approver: ApproverProtocol = self._tool_approver  # type: ignore[assignment]
        self.engine = await ForgeEngine.start(
            paths=self.paths, cfg=self.cfg, approver=approver,
        )
        self.engine.tracer.subscribe(self._on_trace_event)
        n = len(self.engine.tools)
        self._set_status(f"ready · tools={n} · turns={self._turns}")
        self._chat.write(f"[green]ready[/]. tools={n}.")
        self._chat.write("type [b]/help[/] for commands — [b]/done[/] saves memory for this chat session.")
        self._chat.write(
            "[dim]rich UI lives in the Electron app: "
            "cd apps/forge/electron && npm run start[/]"
        )

    async def _tool_approver(
        self, *, tool_name: str, args: dict, agent_name: str, reason: str,
    ) -> bool:
        screen = ToolGateScreen(
            tool_name=tool_name, args=args,
            agent_name=agent_name, reason=reason,
        )
        return bool(await self.push_screen_wait(screen))

    def _set_status(self, text: str) -> None:
        self._status.update(text)

    def _on_trace_event(self, event: dict) -> None:
        """Compact, single-line trace echoes inline in the transcript.

        Side panels are gone in the slim TUI; everything that used to live in
        them now either prints one dim line in the chat (so the operator
        still sees that *something* is happening) or is dropped entirely
        because the Electron app is the right place to inspect it.
        """
        et = event.get("type", "")
        if et == "agent_spawn":
            self._chat.write(
                f"[dim green]+ {event.get('agent_name')} ({event.get('kind')})[/]"
            )
        elif et == "agent_done":
            self._chat.write(
                f"[dim blue]✓ {event.get('agent_name')}[/]"
            )
        elif et == "tool_call":
            self._chat.write(
                f"[dim]· {event.get('agent_name')} → {event.get('tool')}[/]"
            )
        elif et == "permission_request":
            # The broker is calling ToolGateScreen too; this is just an
            # inline breadcrumb in case modals get backed up.
            self._chat.write(
                f"[dim magenta]? {event.get('tool')} (asking)[/]"
            )
        elif et == "compaction_fired":
            self._chat.write(
                f"[dim]compaction fired: {event.get('strategy')}[/]"
            )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = (event.value or "").strip()
        event.input.value = ""
        if not text or self.busy:
            return
        if text.startswith("/"):
            await self._handle_slash(text)
            return
        await self._run_turn(text)

    async def _handle_slash(self, text: str) -> None:
        cmd, _, rest = text.partition(" ")
        cmd = cmd[1:].lower()
        if cmd == "help":
            self._chat.write(_FORGE_HELP)
        elif cmd == "quit":
            await self.action_quit()
        elif cmd == "plan":
            if not rest.strip():
                self._chat.write("usage: /plan <task>")
                return
            await self._run_turn(rest.strip(), plan_mode=True)
        elif cmd == "done":
            if self.engine is None:
                self._chat.write("[red]engine not booted[/]")
                return
            mem = self.cfg.memory
            if mem.enabled and mem.reflect_on_thread_end and self._reflect_buffer:
                self.engine.reflect_conversation(
                    thread_id=self._session_thread_id,
                    messages=list(self._reflect_buffer),
                )
                self._chat.write(
                    "[dim]reflection saved — episodic/procedural stores updated[/]"
                )
            elif (
                mem.enabled
                and mem.reflect_on_thread_end
                and not self._reflect_buffer
            ):
                self._chat.write("[dim]nothing queued to reflect yet[/]")
            self._reflect_buffer.clear()
            self._session_thread_id = f"t-{uuid.uuid4().hex[:10]}"
            self._chat.write("[dim]new session id — keep chatting[/]")
        else:
            self._chat.write(f"unknown command [red]{cmd!r}[/]. /help to list.")

    async def _run_turn(
        self,
        task: str,
        *,
        plan_mode: bool = False,
    ) -> None:
        if self.engine is None:
            self._chat.write("[red]engine not booted[/]")
            return
        self.busy = True
        suffix = " (plan-mode)" if plan_mode else ""
        self._set_status(f"running{suffix} · turns={self._turns}")
        self._chat.write(f"\n[b]you:[/] {task}")
        try:
            result = await self.engine.run_task(
                task,
                plan_mode=plan_mode,
                thread_id=self._session_thread_id,
                reflect=False,
            )
            self._reflect_buffer.append(HumanMessage(content=task))
            ans = result.answer
            self._reflect_buffer.append(
                AIMessage(content=ans if isinstance(ans, str) else str(ans))
            )
            self._chat.write(f"\n[b green]forge:[/] {result.answer}")
            self._turns += 1
        except Exception as exc:  # noqa: BLE001
            self._chat.write(f"\n[red]error:[/] {exc}")
        finally:
            self.busy = False
            self._set_status(f"ready · turns={self._turns}")

    def action_clear_chat(self) -> None:
        self._chat.clear()

    async def on_unmount(self) -> None:
        if self.engine is not None:
            await self.engine.shutdown()


def run_tui(*, paths: ForgePaths, cfg: ForgeConfig) -> None:
    """Entry point invoked by ``forge.cli:cmd_tui``."""
    paths.ensure()
    app = ForgeTUI(paths=paths, cfg=cfg)
    app.run()


__all__ = ["ForgeTUI", "ToolGateScreen", "run_tui"]
