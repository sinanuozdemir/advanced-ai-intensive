"""Append-only JSONL trace writer.

Every event Forge wants to surface (to the TUI side panel, the Electron
WebSocket, and ``.forge/trace.jsonl``) flows through ``Tracer.emit``. The
writer is thread-safe enough for the TUI + WS to share it; each subscriber
is a callable invoked synchronously after the line is written.

Event shape is intentionally permissive: ``{"type": str, "ts": ISO8601,
**fields}``. Known types (P3-P5):

- ``thread_start``       — payload includes ``task_id``, ``task``
- ``policy_decision``    — ``mode``, ``topology``, ``reason``
- ``plan_drafted``       — ``plan_md``
- ``agent_spawn``        — ``agent_name``, ``kind``, ``parent``
- ``agent_done``         — ``agent_name``, ``result``
- ``tool_call``          — ``agent_name``, ``tool``, ``args``
- ``tool_result``        — ``agent_name``, ``tool``, ``ok``, ``preview``
- ``memory_write``       — ``store``, ``id``, ``text``
- ``memory_read``        — ``store``, ``query``, ``hits``
- ``compaction_fired``   — ``strategy``, ``before``, ``after``
- ``thread_end``         — ``task_id``, ``ok``
"""
from __future__ import annotations

import asyncio
import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, Protocol


# Module-level so the engine (which sets it) and the Tracer instance (which
# reads it inside emit()) share one binding. Using a ContextVar means
# concurrent threads each see their own value.
_current_task_id: ContextVar[str | None] = ContextVar(
    "_forge_current_task_id", default=None,
)


def set_current_task_id(task_id: str | None) -> Any:
    """Set the task_id stamp that ``Tracer.emit`` will fold into every
    subsequent event in this async context. Returns a token usable with
    ``reset_current_task_id`` to undo the change."""
    return _current_task_id.set(task_id)


def reset_current_task_id(token: Any) -> None:
    """Undo a previous ``set_current_task_id``."""
    _current_task_id.reset(token)


@contextmanager
def current_task(task_id: str) -> Iterator[None]:
    """Context manager wrapper: ``with current_task(tid): tracer.emit(...)``."""
    token = set_current_task_id(task_id)
    try:
        yield
    finally:
        reset_current_task_id(token)


class Subscriber(Protocol):
    def __call__(self, event: dict) -> Awaitable[None] | None: ...


class Tracer:
    """Append-only JSONL writer with in-process subscriber fan-out."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._subs: list[Subscriber] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def subscribe(self, sub: Subscriber) -> Callable[[], None]:
        """Register a subscriber. Returns an unsubscribe callable."""
        self._subs.append(sub)
        return lambda: self._subs.remove(sub) if sub in self._subs else None

    def emit(self, event_type: str, **fields: Any) -> dict:
        """Write one event line to disk + notify subscribers. Returns the event.

        If no explicit ``task_id`` is in ``fields`` and the current async
        context has one set via :func:`set_current_task_id`, that value is
        stamped onto the event automatically. This is the mechanism that
        lets ``_load_thread_transcript`` filter events by thread without
        every emit call having to remember to pass ``task_id``.
        """
        if "task_id" not in fields:
            cur = _current_task_id.get()
            if cur is not None:
                fields["task_id"] = cur
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            **fields,
        }
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        if self.enabled:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
        for sub in list(self._subs):
            try:
                rv = sub(event)
                if asyncio.iscoroutine(rv):
                    # Best-effort: if there's a running loop, schedule it; else drop.
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(rv)  # type: ignore[arg-type]
                    except RuntimeError:
                        rv.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                # Subscribers must never break the trace path.
                continue
        return event


__all__ = [
    "Tracer",
    "Subscriber",
    "current_task",
    "set_current_task_id",
    "reset_current_task_id",
]
