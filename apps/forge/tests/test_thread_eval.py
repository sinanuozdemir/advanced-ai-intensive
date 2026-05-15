"""Tests for the per-thread eval module.

We stub out the judge LLM with a deterministic ``MagicMock`` so the tests
don't hit any network. The shape of the stub matches what
``with_structured_output(schema, method="function_calling")`` returns:
something whose ``.invoke([messages])`` produces a Pydantic instance of the
target schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from forge.config import ForgeConfig
from forge.eval.thread_eval import (
    OUTCOME_RUBRIC_SYSTEM,
    TRAJECTORY_RUBRIC_SYSTEM,
    OutcomeScore,
    ToolCallRecord,
    TrajectoryScore,
    _format_trajectory_for_prompt,
    evaluate_thread,
    get_thread_eval,
    list_thread_evals,
    load_thread_events,
    reconstruct_trajectory,
    rubric_prompts,
    thread_summary_from_events,
)
from forge.paths import ForgePaths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def paths(tmp_path: Path) -> ForgePaths:
    p = ForgePaths.for_repo(tmp_path)
    p.ensure()
    return p


@pytest.fixture
def cfg() -> ForgeConfig:
    return ForgeConfig()


def _write_trace(paths: ForgePaths, events: list[dict]) -> None:
    """Append rows to the trace file the way ``Tracer.emit`` would."""
    with paths.trace_jsonl.open("a", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


def _sample_thread_events(thread_id: str = "t-test") -> list[dict]:
    """A minimal but realistic thread: user task -> fs_read -> fs_edit -> answer."""
    return [
        {"ts": "2026-05-14T00:00:00Z", "type": "thread_start", "task_id": thread_id,
         "task": "Rename greet() to hello() in greet.py.", "topology": "solo"},
        {"ts": "2026-05-14T00:00:01Z", "type": "tool_call", "task_id": thread_id,
         "agent_name": "main", "tool": "fs_read", "args": {"path": "greet.py"}},
        {"ts": "2026-05-14T00:00:01Z", "type": "tool_result", "task_id": thread_id,
         "agent_name": "main", "tool": "fs_read", "ok": True,
         "preview": "def greet(name):\n    return 'Hello, ' + name\n"},
        {"ts": "2026-05-14T00:00:02Z", "type": "tool_call", "task_id": thread_id,
         "agent_name": "main", "tool": "fs_edit",
         "args": {"path": "greet.py", "old": "def greet", "new": "def hello"}},
        {"ts": "2026-05-14T00:00:02Z", "type": "tool_result", "task_id": thread_id,
         "agent_name": "main", "tool": "fs_edit", "ok": True, "preview": "1 edit applied"},
        {"ts": "2026-05-14T00:00:03Z", "type": "agent_done", "task_id": thread_id,
         "agent_name": "main", "result": "Renamed greet to hello in greet.py."},
        {"ts": "2026-05-14T00:00:03Z", "type": "thread_end", "task_id": thread_id, "ok": True},
    ]


# ---------------------------------------------------------------------------
# Trace I/O
# ---------------------------------------------------------------------------


def test_load_thread_events_filters_by_task_id(paths: ForgePaths) -> None:
    _write_trace(paths, _sample_thread_events("t-a"))
    _write_trace(paths, _sample_thread_events("t-b"))
    events = load_thread_events(paths, "t-a")
    assert events
    assert all(ev.get("task_id") == "t-a" for ev in events)
    # We're not mixing the second thread in.
    assert len(events) == 7


def test_load_thread_events_handles_missing_file(tmp_path: Path) -> None:
    p = ForgePaths.for_repo(tmp_path)
    # No .ensure() — the trace file legitimately does not exist.
    assert load_thread_events(p, "anything") == []


def test_reconstruct_trajectory_pairs_calls_with_results() -> None:
    events = _sample_thread_events("t1")
    traj = reconstruct_trajectory(events)
    assert [c.tool for c in traj] == ["fs_read", "fs_edit"]
    assert all(c.ok is True for c in traj)
    # The fs_read preview ended up on the record:
    assert "Hello," in traj[0].preview
    # fs_edit args came through as a dict:
    assert traj[1].args["new"] == "def hello"


def test_reconstruct_trajectory_handles_orphan_result() -> None:
    # A result with no matching call (e.g. we joined the trace late). The
    # function should still surface it so the eval prompt sees the evidence.
    events = [
        {"type": "tool_result", "tool": "fs_read", "agent_name": "main",
         "ok": False, "preview": "ENOENT"},
    ]
    traj = reconstruct_trajectory(events)
    assert len(traj) == 1
    assert traj[0].tool == "fs_read" and traj[0].ok is False


def test_thread_summary_extracts_user_and_answer() -> None:
    summary = thread_summary_from_events(_sample_thread_events("t1"))
    assert "Rename greet" in summary["user_task"]
    assert "Renamed greet" in summary["final_answer"]
    assert summary["ok"] is True
    assert summary["topology"] == "solo"
    assert len(summary["trajectory"]) == 2


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def test_format_trajectory_truncates_long_args() -> None:
    long_blob = "x" * 1_000
    traj = [
        ToolCallRecord(agent="main", tool="fs_write",
                       args={"path": "a", "content": long_blob}, ok=True,
                       preview="ok"),
    ]
    rendered = _format_trajectory_for_prompt(traj)
    # 200-char cap (with the "..." suffix) means the whole long_blob can't
    # survive. We assert by length and substring containment.
    assert len(rendered) < 800
    assert "fs_write" in rendered and "ok" in rendered


def test_format_trajectory_empty() -> None:
    assert "(no tool calls)" in _format_trajectory_for_prompt([])


# ---------------------------------------------------------------------------
# Rubric prompts (exposed via /api/eval/rubrics)
# ---------------------------------------------------------------------------


def test_rubric_prompts_match_constants() -> None:
    rp = rubric_prompts()
    assert rp["outcome"] is OUTCOME_RUBRIC_SYSTEM
    assert rp["trajectory"] is TRAJECTORY_RUBRIC_SYSTEM
    # The trajectory rubric must teach the model about safety — it's the
    # whole point of having a per-thread eval in a coding agent.
    assert "safety" in rp["trajectory"].lower()


# ---------------------------------------------------------------------------
# evaluate_thread end-to-end with a stub judge LLM
# ---------------------------------------------------------------------------


def _stub_judge_pair(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Patch ``_bind_judge`` so the two rubric calls return deterministic
    Pydantic instances without touching a real LLM. Returns the bound
    objects so tests can assert on the SystemMessage they received."""
    outcome_bound = MagicMock()
    trajectory_bound = MagicMock()

    def _stub_outcome_invoke(messages):  # type: ignore[no-untyped-def]
        # Make sure we got the right system prompt.
        assert messages[0].content == OUTCOME_RUBRIC_SYSTEM
        return OutcomeScore(
            correctness=5, completeness=4, grounding=5,
            overall=4.75,
            rationale="Mock: looks fine.",
        )

    def _stub_trajectory_invoke(messages):  # type: ignore[no-untyped-def]
        assert messages[0].content == TRAJECTORY_RUBRIC_SYSTEM
        return TrajectoryScore(
            tool_choice=5, efficiency=4, safety=5,
            overall=4.75,
            rationale="Mock: tight trajectory.",
        )

    outcome_bound.invoke.side_effect = _stub_outcome_invoke
    trajectory_bound.invoke.side_effect = _stub_trajectory_invoke

    calls = {"n": 0}

    def fake_bind(_judge_llm, schema):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if schema is OutcomeScore:
            return outcome_bound
        if schema is TrajectoryScore:
            return trajectory_bound
        raise AssertionError(f"unexpected schema: {schema!r}")

    monkeypatch.setattr("forge.eval.thread_eval._bind_judge", fake_bind)
    return {"outcome": outcome_bound, "trajectory": trajectory_bound}


def test_evaluate_thread_full_roundtrip(
    paths: ForgePaths, cfg: ForgeConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_trace(paths, _sample_thread_events("t-roundtrip"))
    _stub_judge_pair(monkeypatch)

    rec = evaluate_thread(paths=paths, cfg=cfg, thread_id="t-roundtrip")

    # Shape checks on the persisted record.
    assert rec.thread_id == "t-roundtrip"
    assert "Rename greet" in rec.user_task
    assert "Renamed greet" in rec.final_answer
    assert rec.outcome["overall"] == 4.75
    assert rec.trajectory["overall"] == 4.75
    assert rec.outcome["rationale"].startswith("Mock:")
    assert rec.trajectory["rationale"].startswith("Mock:")
    assert rec.error == ""
    # The judge_models block falls back to ``cfg.models.judge`` because we
    # didn't override either rubric model in EvalConfig.
    assert rec.judge_models["outcome"] == cfg.models.judge
    assert rec.judge_models["trajectory"] == cfg.models.judge

    # JSONL on disk: list view returns this row.
    rows = list_thread_evals(paths)
    assert len(rows) == 1
    assert rows[0]["thread_id"] == "t-roundtrip"

    # get_thread_eval reads it back, too.
    fetched = get_thread_eval(paths, "t-roundtrip")
    assert fetched is not None
    assert fetched["outcome"]["overall"] == 4.75


def test_evaluate_thread_returns_most_recent_run(
    paths: ForgePaths, cfg: ForgeConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_thread_eval`` should return the *last* line for a given thread,
    not the first. (Users can re-run rubrics via the API.)"""
    _write_trace(paths, _sample_thread_events("t-rerun"))
    _stub_judge_pair(monkeypatch)

    evaluate_thread(paths=paths, cfg=cfg, thread_id="t-rerun")
    first_ts = get_thread_eval(paths, "t-rerun")["ts"]  # type: ignore[index]
    evaluate_thread(paths=paths, cfg=cfg, thread_id="t-rerun")
    second_ts = get_thread_eval(paths, "t-rerun")["ts"]  # type: ignore[index]

    assert second_ts >= first_ts
    assert len(list_thread_evals(paths)) == 2


def test_evaluate_thread_missing_raises(
    paths: ForgePaths, cfg: ForgeConfig,
) -> None:
    with pytest.raises(RuntimeError, match="no trace events"):
        evaluate_thread(paths=paths, cfg=cfg, thread_id="t-nonexistent")


def test_evaluate_thread_captures_partial_failure(
    paths: ForgePaths, cfg: ForgeConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If one rubric raises (network blip, judge timeout) the eval should
    still persist with the other rubric's score and a non-empty ``error``
    field — we never lose data because one LLM call flaked."""
    _write_trace(paths, _sample_thread_events("t-partial"))

    outcome_bound = MagicMock()
    outcome_bound.invoke.side_effect = RuntimeError("transient")
    trajectory_bound = MagicMock()
    trajectory_bound.invoke.return_value = TrajectoryScore(
        tool_choice=4, efficiency=4, safety=5, overall=4.25,
        rationale="Stub.",
    )

    def fake_bind(_judge_llm, schema):  # type: ignore[no-untyped-def]
        return outcome_bound if schema is OutcomeScore else trajectory_bound

    monkeypatch.setattr("forge.eval.thread_eval._bind_judge", fake_bind)

    rec = evaluate_thread(paths=paths, cfg=cfg, thread_id="t-partial")
    assert "outcome rubric failed" in rec.error
    assert rec.outcome == {}
    assert rec.trajectory["overall"] == 4.25


def test_list_thread_evals_newest_first(
    paths: ForgePaths, cfg: ForgeConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_trace(paths, _sample_thread_events("t-A"))
    _write_trace(paths, _sample_thread_events("t-B"))
    _stub_judge_pair(monkeypatch)
    evaluate_thread(paths=paths, cfg=cfg, thread_id="t-A")
    evaluate_thread(paths=paths, cfg=cfg, thread_id="t-B")

    rows = list_thread_evals(paths)
    assert [r["thread_id"] for r in rows] == ["t-B", "t-A"]


def test_eval_config_overrides_judge_model(
    paths: ForgePaths, cfg: ForgeConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``cfg.eval.outcome_judge_model`` is set, we pass that model
    slug to ``shared.get_llm`` (mocked here) and record it in
    ``judge_models``."""
    cfg.eval.outcome_judge_model = "google/gemini-2.5-flash"
    cfg.eval.trajectory_judge_model = "anthropic/claude-haiku-4.5"

    _write_trace(paths, _sample_thread_events("t-models"))
    _stub_judge_pair(monkeypatch)

    fake_get_llm = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("shared.get_llm", fake_get_llm)

    rec = evaluate_thread(paths=paths, cfg=cfg, thread_id="t-models")
    assert rec.judge_models["outcome"] == "google/gemini-2.5-flash"
    assert rec.judge_models["trajectory"] == "anthropic/claude-haiku-4.5"
    # Both override slugs were resolved via shared.get_llm.
    assert {c.args[0] for c in fake_get_llm.call_args_list} == {
        "google/gemini-2.5-flash",
        "anthropic/claude-haiku-4.5",
    }
