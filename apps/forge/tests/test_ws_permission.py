"""WS-based permission approval — request, response, timeout."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from forge.paths import ForgePaths
from forge.server import _ServerState


def _make_state(tmp_path: Path) -> _ServerState:
    (tmp_path / ".forge").mkdir(exist_ok=True)
    paths = ForgePaths.for_repo(tmp_path).ensure()
    return _ServerState(paths)


@pytest.mark.asyncio
async def test_request_approval_resolved_by_client(tmp_path: Path) -> None:
    """Happy path: a 'client' replies before the timeout — the gate returns
    that decision, and the request is no longer pending."""
    state = _make_state(tmp_path)
    state.approval_timeout_s = 5.0

    captured: dict[str, Any] = {}

    def watcher(event: dict) -> None:
        if event.get("type") == "permission_request":
            captured["rid"] = event["request_id"]
            captured["args"] = event["args"]

    # Subscribe directly to the broadcast hook so we don't need a real WS.
    state.subscribers  # touch attribute
    # Re-use the same loop, queue path used in production:
    q = state.add_subscriber()

    async def consumer() -> None:
        while "rid" not in captured:
            event = await q.get()
            watcher(event)

    consumer_task = asyncio.create_task(consumer())
    # The "client" responds 50ms after the broadcast lands.
    async def respond() -> None:
        await asyncio.sleep(0.05)
        # consumer should have populated captured by now; if not, wait
        for _ in range(40):
            if "rid" in captured:
                break
            await asyncio.sleep(0.025)
        await state.resolve_approval(captured["rid"], approved=True)

    respond_task = asyncio.create_task(respond())

    approved = await state.request_approval(
        tool_name="fs_write",
        args={"path": "/tmp/x", "content": "x" * 2_000},
        agent_name="main",
        reason="config says ask",
    )
    await consumer_task
    await respond_task

    assert approved is True
    # Captured a real request_id from the broadcast.
    assert isinstance(captured.get("rid"), str)
    # Args were truncated to keep the WS frame small.
    assert isinstance(captured["args"].get("content"), str)
    assert len(captured["args"]["content"]) < 2_000

    # Future is gone now.
    assert state.pending_approvals == {}


@pytest.mark.asyncio
async def test_request_approval_times_out(tmp_path: Path) -> None:
    """No client subscribed: the timeout fallback kicks in."""
    state = _make_state(tmp_path)
    state.approval_timeout_s = 0.2
    state.approval_timeout_decision = True

    approved = await state.request_approval(
        tool_name="fs_edit",
        args={"path": "/tmp/x"},
        agent_name="main",
        reason="config says ask",
    )
    assert approved is True  # default: auto-approve on timeout
    assert state.pending_approvals == {}


@pytest.mark.asyncio
async def test_request_approval_times_out_to_deny(tmp_path: Path) -> None:
    """Flip the fallback to False (safe mode)."""
    state = _make_state(tmp_path)
    state.approval_timeout_s = 0.1
    state.approval_timeout_decision = False

    approved = await state.request_approval(
        tool_name="fs_edit",
        args={"path": "/tmp/x"},
        agent_name="main",
        reason="config says ask",
    )
    assert approved is False


@pytest.mark.asyncio
async def test_resolve_approval_unknown_request_id(tmp_path: Path) -> None:
    """Stale or invalid request_ids are a no-op (the WS inbound loop must
    not crash if the user clicks 'allow' on a modal we've already
    auto-approved)."""
    state = _make_state(tmp_path)
    ok = await state.resolve_approval("not-a-real-id", approved=True)
    assert ok is False
