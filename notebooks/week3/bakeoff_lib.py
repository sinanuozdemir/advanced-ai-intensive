"""Plan/Act bake-off — isolated experiment harness.

This module is INTENTIONALLY decoupled from Forge. The bake-off is an
upstream experiment whose results inform which policy Forge should ship —
Forge is downstream, not under test.

What lives here:

- :class:`Decision`             — what every policy returns
- :class:`BakeoffContext`       — passed to every ``policy.decide(...)`` call
- :class:`PlanActPolicy`        — the protocol
- :class:`GoldTask`             — one row of ``data/plan_act_golds.jsonl``
- :func:`load_golds`            — read the JSONL into ``GoldTask`` objects
- :func:`make_fs_tools`         — LangChain tools scoped to a fixture dir
- :func:`make_coding_workers`   — WorkerSpec list used by build_solo/supervisor
- :func:`run_task_on_fixture`   — execute one (task, policy) trial end-to-end

The five policies live in ``plan_act_alts.py``. Scoring goes through
``notebooks/week1/judges.py:judge_with_rubric`` directly.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from multi_agent.topologies import build_solo, build_supervisor
from multi_agent.workers import WorkerSpec
from shared import get_llm

# Week 1's rubric judge. We use it directly — no Forge wrapper.
import sys
_W1 = Path(__file__).resolve().parents[1] / "week1"
if str(_W1) not in sys.path:
    sys.path.insert(0, str(_W1))
from judges import judge_with_rubric  # noqa: E402


# ---------------------------------------------------------------------------
# Policy protocol
# ---------------------------------------------------------------------------


Mode = Literal["plan", "act"]
TopologyName = Literal["solo", "supervisor"]


@dataclass
class Decision:
    """What every plan/act policy returns."""
    mode: Mode
    topology: TopologyName
    plan_md: str | None = None
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSlugs:
    """Minimal model registry passed to policies. No Forge dependency."""
    planner: str = "openai/gpt-5.4-nano"
    coder: str = "openai/gpt-5.4-nano"
    critic: str = "openai/gpt-5.4-nano"
    trajectory_probe: str = "openai/gpt-5.4-nano"


@dataclass
class BakeoffContext:
    """A small bag of references handed to every policy at decide time."""
    models: ModelSlugs = field(default_factory=ModelSlugs)
    episodic_headlines: list[str] = field(default_factory=list)
    self_critique_threshold: float = 3.5


class PlanActPolicy(Protocol):
    name: str

    async def decide(
        self, task: str, history: list[Any], ctx: BakeoffContext,
    ) -> Decision: ...


# ---------------------------------------------------------------------------
# Gold set
# ---------------------------------------------------------------------------


GOLDS_PATH = Path(__file__).resolve().parent / "data" / "plan_act_golds.jsonl"


@dataclass
class GoldTask:
    id: str
    task: str
    success_criteria: str
    bucket: str = "misc"
    difficulty: int = 1
    blast_radius: str = "low"
    expected_plan_required: bool = False
    files: dict[str, str] | None = None  # inline fixture {rel_path: content}
    oracle: dict[str, Any] | None = None  # post-run filesystem checks; see evaluate_oracle


# ---------------------------------------------------------------------------
# Filesystem oracle — objective post-run checks on the fixture dir
# ---------------------------------------------------------------------------


@dataclass
class OracleResult:
    """Outcome of running a task's filesystem oracle."""
    passed: bool
    passed_count: int
    total: int
    failures: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.total == 0:
            return "no-oracle"
        if self.passed:
            return f"{self.passed_count}/{self.total} ok"
        return f"{self.passed_count}/{self.total} ok; failed: " + "; ".join(self.failures[:3])


def evaluate_oracle(fixture_dir: Path, oracle: dict[str, Any] | None) -> OracleResult:
    """Run the structured filesystem checks in ``oracle`` against ``fixture_dir``.

    Supported keys (all optional):

    * ``files_exist``: list of relative paths that MUST exist as files.
    * ``files_absent``: list of relative paths that MUST NOT exist.
    * ``files_contain``: ``{rel_path: [substring, ...]}`` — every substring must
      appear in the file's text. Use a list to require multiple substrings.
    * ``files_not_contain``: ``{rel_path: [substring, ...]}`` — none of the
      substrings may appear.
    """
    if not oracle:
        return OracleResult(passed=True, passed_count=0, total=0)

    failures: list[str] = []
    passed_count = 0
    total = 0

    fixture_dir = Path(fixture_dir)

    for rel in oracle.get("files_exist", []) or []:
        total += 1
        if (fixture_dir / rel).is_file():
            passed_count += 1
        else:
            failures.append(f"missing file: {rel}")

    for rel in oracle.get("files_absent", []) or []:
        total += 1
        if not (fixture_dir / rel).exists():
            passed_count += 1
        else:
            failures.append(f"unexpected file: {rel}")

    def _as_list(v: Any) -> list[str]:
        if v is None:
            return []
        return [v] if isinstance(v, str) else list(v)

    for rel, needles in (oracle.get("files_contain") or {}).items():
        for needle in _as_list(needles):
            total += 1
            p = fixture_dir / rel
            if not p.is_file():
                failures.append(f"{rel} missing (wanted '{needle[:40]}')")
                continue
            txt = p.read_text(encoding="utf-8", errors="replace")
            if needle in txt:
                passed_count += 1
            else:
                failures.append(f"{rel} missing substring '{needle[:40]}'")

    for rel, needles in (oracle.get("files_not_contain") or {}).items():
        for needle in _as_list(needles):
            total += 1
            p = fixture_dir / rel
            if not p.is_file():
                # Absent file trivially satisfies "must not contain".
                passed_count += 1
                continue
            txt = p.read_text(encoding="utf-8", errors="replace")
            if needle not in txt:
                passed_count += 1
            else:
                failures.append(f"{rel} still contains '{needle[:40]}'")

    return OracleResult(
        passed=(total > 0 and passed_count == total),
        passed_count=passed_count,
        total=total,
        failures=failures,
    )


def load_golds(path: Path | str | None = None) -> list[GoldTask]:
    """Load the bake-off gold set. Each row is one trial input."""
    p = Path(path) if path else GOLDS_PATH
    out: list[GoldTask] = []
    if not p.is_file():
        return out
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            # Drop any keys the dataclass doesn't know about (e.g. legacy `fixture`)
            keep = {k: v for k, v in data.items() if k in GoldTask.__dataclass_fields__}
            out.append(GoldTask(**keep))
    return out


def materialise_fixture(task: GoldTask) -> Path:
    """Write the task's inline fixture files to a fresh temp dir."""
    dst = Path(tempfile.mkdtemp(prefix=f"bakeoff-{task.id}-"))
    for rel, content in (task.files or {}).items():
        p = dst / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return dst


# ---------------------------------------------------------------------------
# Tools: minimal filesystem tools scoped to a fixture dir
# ---------------------------------------------------------------------------


def _safe_join(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root`` and refuse to escape it."""
    p = (root / rel).resolve()
    root_r = root.resolve()
    if root_r != p and root_r not in p.parents:
        raise ValueError(f"path escapes fixture root: {rel}")
    return p


def make_fs_tools(root: Path) -> list[Any]:
    """Build read/list/write/edit LangChain tools scoped to ``root``."""

    root = Path(root)

    @tool
    def fs_read(path: str) -> str:
        """Read a file relative to the project root. Returns its full text.

        Args:
            path: relative path, e.g. 'src/main.py'.
        """
        p = _safe_join(root, path)
        if not p.is_file():
            return f"ERROR: not a file: {path}"
        return p.read_text(encoding="utf-8")[:20_000]

    @tool
    def fs_list(path: str = ".") -> str:
        """List the contents of a directory relative to the project root.

        Args:
            path: relative dir, default '.'.
        """
        p = _safe_join(root, path)
        if not p.is_dir():
            return f"ERROR: not a dir: {path}"
        entries = []
        for child in sorted(p.iterdir()):
            kind = "d" if child.is_dir() else "f"
            entries.append(f"{kind} {child.name}")
        return "\n".join(entries) or "(empty)"

    @tool
    def fs_write(path: str, content: str) -> str:
        """Create or overwrite a file with ``content``.

        Args:
            path: relative path.
            content: the new file contents.
        """
        p = _safe_join(root, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {path} ({len(content)} bytes)"

    @tool
    def fs_edit(path: str, old: str, new: str) -> str:
        """Replace exactly one occurrence of ``old`` with ``new`` in ``path``.

        Args:
            path: relative path.
            old: literal text to find (must be unique in the file).
            new: replacement text.
        """
        p = _safe_join(root, path)
        if not p.is_file():
            return f"ERROR: not a file: {path}"
        text = p.read_text(encoding="utf-8")
        n = text.count(old)
        if n == 0:
            return f"ERROR: 'old' not found in {path}"
        if n > 1:
            return f"ERROR: 'old' is not unique in {path} ({n} matches)"
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"edited {path} (1 replacement)"

    @tool
    def fs_delete(path: str) -> str:
        """Delete a file relative to the project root.

        Args:
            path: relative path.
        """
        p = _safe_join(root, path)
        if not p.is_file():
            return f"ERROR: not a file: {path}"
        p.unlink()
        return f"deleted {path}"

    return [fs_read, fs_list, fs_write, fs_edit, fs_delete]


# ---------------------------------------------------------------------------
# Workers + topology builders
# ---------------------------------------------------------------------------


_CODER_SYS = (
    "You are a careful coding assistant working in a small repo. Use the "
    "filesystem tools to read, list, write, edit, or delete files as needed. "
    "Be terse. Prefer the smallest change that satisfies the task. "
    "If the task is read-only, do NOT modify files."
)

_PLANNER_SYS = (
    "You are a planner. Produce a numbered plan a coder can execute. "
    "3-7 steps. The final step describes how success is verified."
)

_CRITIC_SYS = (
    "You are a critic. Review the work that was done and identify issues "
    "or confirm completion. Be specific and terse."
)


def make_coding_workers(root: Path, model_slug: str) -> list[WorkerSpec]:
    """A planner / coder / critic trio sharing the same fs tools.

    Used by ``build_supervisor`` when a policy picks topology=supervisor.
    For topology=solo, ``build_solo`` merges all three workers' tools — and
    the planner/critic become tool-noops the solo agent rarely calls.
    """
    fs = make_fs_tools(root)
    return [
        WorkerSpec(
            name="coder", description="Read / write / edit / delete files.",
            system_prompt=_CODER_SYS, tools=fs, model_slug=model_slug,
        ),
        WorkerSpec(
            name="planner", description="Draft a step-by-step plan; no tools.",
            system_prompt=_PLANNER_SYS, tools=[], model_slug=model_slug,
        ),
        WorkerSpec(
            name="critic", description="Review work and confirm or critique.",
            system_prompt=_CRITIC_SYS, tools=fs, model_slug=model_slug,
        ),
    ]


# ---------------------------------------------------------------------------
# Per-trial runner
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    task_id: str
    bucket: str
    policy: str
    topology: str
    planned: bool
    answer: str
    score: float
    passed: bool
    rationale: str
    elapsed_s: float
    decision_reason: str
    cost_usd: float = 0.0
    error: str = ""
    oracle_total: int = 0
    oracle_passed_count: int = 0
    oracle_passed: bool = True
    oracle_summary: str = "no-oracle"


PASS_THRESHOLD = 4.0


async def run_task_on_fixture(
    task: GoldTask,
    policy: PlanActPolicy,
    *,
    model_slug: str = "openai/gpt-5.4-nano",
    judge_model: str | None = None,
    ctx: BakeoffContext | None = None,
) -> TrialResult:
    """Run one (task, policy) trial end-to-end.

    Materializes the fixture, asks the policy for a ``Decision``, dispatches
    to the matching topology, judges the answer, cleans up the fixture.
    """
    work_dir = materialise_fixture(task)
    ctx = ctx or BakeoffContext(models=ModelSlugs(
        planner=model_slug, coder=model_slug, critic=model_slug,
        trajectory_probe=model_slug,
    ))
    t0 = time.perf_counter()
    error = ""
    answer = ""
    topology = "solo"
    planned = False
    decision_reason = ""
    cost = 0.0
    oracle_result: OracleResult = OracleResult(passed=True, passed_count=0, total=0)
    try:
        decision = await policy.decide(task=task.task, history=[], ctx=ctx)
        topology = decision.topology
        planned = decision.mode == "plan"
        decision_reason = (decision.reason or "")[:140]

        executable = task.task
        if decision.plan_md:
            executable = f"{task.task}\n\nApproved plan:\n{decision.plan_md}"

        workers = make_coding_workers(work_dir, model_slug)
        if topology == "supervisor":
            topo = build_supervisor(workers, supervisor_model=model_slug, max_steps=6)
        else:
            topo = build_solo(workers, model_slug=model_slug)
        result = await asyncio.to_thread(topo.invoke, {"task": executable})
        answer = result.answer or ""
        cost = float(getattr(result, "cost_usd", 0.0) or 0.0)
        # Run the filesystem oracle BEFORE the temp dir is wiped.
        oracle_result = evaluate_oracle(work_dir, task.oracle)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    elapsed = time.perf_counter() - t0
    if error:
        return TrialResult(
            task_id=task.id, bucket=task.bucket, policy=policy.name,
            topology=topology, planned=planned, answer="", score=0.0,
            passed=False, rationale="", elapsed_s=elapsed,
            decision_reason=decision_reason, cost_usd=cost, error=error,
            oracle_total=oracle_result.total,
            oracle_passed_count=oracle_result.passed_count,
            oracle_passed=oracle_result.passed,
            oracle_summary=oracle_result.summary,
        )
    try:
        judge_llm = get_llm(judge_model) if judge_model else None
        rv = await asyncio.to_thread(
            judge_with_rubric,
            question=task.task,
            answer=answer,
            reference=task.success_criteria,
            evidence=None,
            judge_llm=judge_llm,
        )
        overall = float(getattr(rv, "overall", 0) or 0)
        notes = str(getattr(rv, "notes", "") or "")
    except Exception as exc:  # noqa: BLE001
        return TrialResult(
            task_id=task.id, bucket=task.bucket, policy=policy.name,
            topology=topology, planned=planned, answer=answer, score=0.0,
            passed=False, rationale="", elapsed_s=elapsed,
            decision_reason=decision_reason, cost_usd=cost,
            error=f"judge failed: {exc!r}",
            oracle_total=oracle_result.total,
            oracle_passed_count=oracle_result.passed_count,
            oracle_passed=oracle_result.passed,
            oracle_summary=oracle_result.summary,
        )

    # Combined pass: text judge must clear PASS_THRESHOLD AND, when the task
    # ships a filesystem oracle, every oracle check must pass too. The oracle
    # is the objective veto — a confident wrong answer can't bluff past it.
    judge_pass = overall >= PASS_THRESHOLD
    has_oracle = oracle_result.total > 0
    passed = judge_pass and (oracle_result.passed if has_oracle else True)

    # If the oracle vetoes a high-scoring answer, clamp the displayed score so
    # downstream aggregations reflect reality.
    if has_oracle and not oracle_result.passed and overall >= PASS_THRESHOLD:
        overall = min(overall, 2.5)

    return TrialResult(
        task_id=task.id, bucket=task.bucket, policy=policy.name,
        topology=topology, planned=planned, answer=answer,
        score=overall, passed=passed,
        rationale=notes, elapsed_s=elapsed,
        decision_reason=decision_reason, cost_usd=cost,
        oracle_total=oracle_result.total,
        oracle_passed_count=oracle_result.passed_count,
        oracle_passed=oracle_result.passed,
        oracle_summary=oracle_result.summary,
    )


__all__ = [
    "Decision",
    "BakeoffContext",
    "ModelSlugs",
    "PlanActPolicy",
    "GoldTask",
    "OracleResult",
    "TrialResult",
    "load_golds",
    "materialise_fixture",
    "make_fs_tools",
    "make_coding_workers",
    "evaluate_oracle",
    "run_task_on_fixture",
    "PASS_THRESHOLD",
]
