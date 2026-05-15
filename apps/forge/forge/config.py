"""ForgeConfig — single source of truth for tunable knobs.

Loaded from ``<repo>/.forge/config.toml``; written atomically when the
Electron settings form (or `forge config --set`) saves. Sections mirror the
TOML one-for-one; see ``DEFAULT_TOML`` for the documented schema.

Hot-reload rule (enforced by callers, not by this module): changes apply to
the *next* thread; in-flight threads keep their original config. Fields are
annotated with ``live=True`` (in ``json_schema_extra``) when it is safe to
swap them mid-process.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError

try:
    import tomllib  # type: ignore[import-not-found]  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

from .paths import ForgePaths


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------


def _live(live: bool = False) -> dict[str, Any]:
    """Mark a field as hot-reloadable (or not) so the Electron form can badge it."""
    return {"json_schema_extra": {"live": live}}


class ModelsConfig(BaseModel):
    """Slugs accepted in every field:

    * OpenRouter slugs — ``"<vendor>/<model>"`` (e.g. ``"openai/gpt-5.4-nano"``,
      ``"anthropic/claude-opus-4.7"``). Requires ``OPENROUTER_API_KEY``.
    * Ollama slugs — ``"ollama/<model>"`` (e.g. ``"ollama/llama3.2"``). Routes
      to a local Ollama server via ``OLLAMA_HOST`` (default
      ``http://localhost:11434``). No API key required, cost = $0.

    Hit ``GET /api/models/health?slug=...`` to verify any slug end-to-end.
    """

    # All model fields are hot-reloadable: PUT /api/config now syncs
    # the new slugs into the running engine and calls
    # ``engine.rebuild_main()`` so the next turn picks them up without
    # a ``forge serve`` restart. The ``model_in_use`` trace event
    # echoes the live slug at the start of every turn for verification.
    default_agent: str = Field("openai/gpt-5.4-nano", **_live(True))
    planner: str = Field("openai/gpt-5.4-nano", **_live(True))
    coder: str = Field("openai/gpt-5.4-nano", **_live(True))
    critic: str = Field("openai/gpt-5.4-nano", **_live(True))
    summarizer: str = Field("anthropic/claude-opus-4.7", **_live(True))
    judge: str = Field("anthropic/claude-opus-4.7", **_live(True))
    # Cheap structured-output LLM used to gate procedural-skill injection.
    # It receives the user's current message + each candidate skill's
    # ``when_to_use`` + fragment, then returns a per-skill keep/drop with
    # one-sentence chain-of-thought reasoning. Defaults to the smallest
    # available slug — this fires once per chat turn.
    procedural_judge: str = Field("openai/gpt-5.4-nano", **_live(True))


CompactionStrategy = Literal[
    "refine", "rules_first", "map_reduce", "lc_sliding_window",
    "recursive", "hierarchical", "none",
]
TriggerKind = Literal["tokens", "messages"]


class CompactionConfig(BaseModel):
    strategy: CompactionStrategy = Field("refine", **_live(False))
    trigger_kind: TriggerKind = Field("tokens", **_live(False))
    trigger_threshold: int = Field(32000, ge=100, **_live(False))
    keep_last: int = Field(6, ge=0, **_live(False))
    summary_max_tokens: int = Field(1500, ge=100, **_live(False))


class MemoryConfig(BaseModel):
    enabled: bool = Field(True, **_live(False))
    semantic_k: int = Field(5, ge=0, **_live(True))
    episodic_k: int = Field(3, ge=0, **_live(True))
    # Procedural-skill injection is two-staged per chat turn:
    #   1. cosine search over each skill's ``when_to_use`` cue picks the
    #      top ``procedural_candidate_pool`` candidates (cheap recall).
    #   2. ``models.procedural_judge`` evaluates each candidate with a
    #      structured output {reasoning, keep}; only ``keep=True`` skills
    #      are injected, capped at ``skill_inject_count``.
    # Reusing ``skill_inject_count`` keeps existing config.toml valid.
    skill_inject_count: int = Field(3, ge=0, **_live(True))
    procedural_candidate_pool: int = Field(6, ge=1, le=50, **_live(True))
    reflect_on_thread_end: bool = Field(True, **_live(True))
    semantic_thread_start_k: int = Field(3, ge=0, le=20, **_live(True))


class RepoRagConfig(BaseModel):
    bm25_k: int = Field(20, ge=1, **_live(True))
    dense_k: int = Field(20, ge=1, **_live(True))
    rrf_k: int = Field(60, ge=1, **_live(True))
    rerank_top_k: int = Field(5, ge=1, **_live(True))
    chunk_size: int = Field(800, ge=64, **_live(False))
    chunk_overlap: int = Field(120, ge=0, **_live(False))
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", **_live(False)
    )
    index_excludes: list[str] = Field(
        default_factory=lambda: [
            "node_modules", "dist", ".venv", ".git", ".forge",
            "**/*.lock", "**/*.min.js",
        ],
        **_live(False),
    )


PermissionDecision = Literal["allow", "ask", "deny"]


class PermissionsConfig(BaseModel):
    default: PermissionDecision = Field("ask", **_live(True))
    shell_allowlist: list[str] = Field(
        default_factory=lambda: [
            "ls", "cat", "rg", "git status", "git diff", "pytest",
        ],
        **_live(True),
    )
    tools: dict[str, PermissionDecision] = Field(
        default_factory=lambda: {
            # Reads default to allow.
            "fs_read": "allow",
            "fs_list": "allow",
            "repo_rag_hybrid_retrieve": "allow",
            "git_status": "allow",
            "git_diff": "allow",
            "git_log": "allow",
            "git_branch": "allow",
            "shell_allowlist": "allow",
            # Writes default to ask (caught by the broker's approver if set,
            # else denied in headless mode).
            "fs_write": "ask",
            "fs_edit": "ask",
            "fs_mkdir": "ask",
            "git_add": "ask",
            "git_commit": "ask",
            "git_reset": "ask",
            # Destructive / outbound stays denied.
            "git_push": "deny",
        },
        **_live(True),
    )


class CheckpointConfig(BaseModel):
    db_path: str = Field(".forge/checkpoints.sqlite", **_live(False))
    resume_on_launch: bool = Field(True, **_live(True))


class TraceConfig(BaseModel):
    enabled: bool = Field(True, **_live(True))
    path: str = Field(".forge/trace.jsonl", **_live(False))


UITheme = Literal["dark", "light"]


class UIConfig(BaseModel):
    theme: UITheme = Field("dark", **_live(True))
    server_port: int = Field(6790, ge=1024, le=65535, **_live(False))
    show_audit_panel: bool = Field(True, **_live(True))


class EvalConfig(BaseModel):
    """Per-thread auto-eval (LLM-as-judge over outcomes + trajectories).

    See ``forge.eval.thread_eval`` for the rubric prompts. Costs two LLM
    calls per finished thread when enabled — turn off if you're paying
    per token and don't need the dashboard.
    """

    auto_evaluate_threads: bool = Field(
        True,
        description=(
            "Run the outcome + trajectory rubrics automatically when each "
            "chat thread ends. Off = manual via POST /api/eval/threads/{id}/run."
        ),
        **_live(True),
    )
    outcome_judge_model: str | None = Field(
        None,
        description=(
            "Model slug for the outcome rubric. None = use ``models.judge``."
        ),
        **_live(True),
    )
    trajectory_judge_model: str | None = Field(
        None,
        description=(
            "Model slug for the trajectory rubric. None = use ``models.judge``."
        ),
        **_live(True),
    )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class ForgeConfig(BaseModel):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    repo_rag: RepoRagConfig = Field(default_factory=RepoRagConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_config(paths: ForgePaths) -> ForgeConfig:
    """Load `<repo>/.forge/config.toml`. If missing, return defaults (and do
    NOT create the file — that's the caller's choice)."""
    if not paths.config_toml.is_file():
        return ForgeConfig()
    with paths.config_toml.open("rb") as fh:
        raw = tomllib.load(fh)
    try:
        return ForgeConfig.model_validate(raw)
    except ValidationError as exc:
        # Surface the bad keys but don't crash a TUI just because config drifted.
        raise ForgeConfigError(
            f"Invalid Forge config at {paths.config_toml}:\n{exc}"
        ) from exc


def _strip_none(value: Any) -> Any:
    """Recursively drop ``None`` values from dicts/lists.

    TOML has no representation for null, so ``tomli_w`` raises on it. Any
    ``Optional[...]`` field on ``ForgeConfig`` (e.g. ``eval.outcome_judge_model``)
    will dump as ``None`` and break the write. Stripping is safe because
    pydantic re-applies the default (``None``) on load when the key is
    absent — same semantics, just no on-disk row.
    """
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(v) for v in value if v is not None]
    return value


def write_config(paths: ForgePaths, cfg: ForgeConfig) -> Path:
    """Atomically write the config TOML; returns the final path."""
    paths.ensure()
    target = paths.config_toml
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _strip_none(cfg.model_dump(mode="python"))
    fd, tmp = tempfile.mkstemp(
        prefix=".config.toml.", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            tomli_w.dump(data, fh)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def ensure_config(paths: ForgePaths) -> tuple[ForgeConfig, bool]:
    """Return ``(config, wrote_default)``. Creates a default file if missing."""
    if paths.config_toml.is_file():
        return load_config(paths), False
    cfg = ForgeConfig()
    write_config(paths, cfg)
    return cfg, True


class ForgeConfigError(RuntimeError):
    """Raised when a TOML on disk fails pydantic validation."""


__all__ = [
    "ForgeConfig",
    "ModelsConfig",
    "CompactionConfig",
    "MemoryConfig",
    "RepoRagConfig",
    "PermissionsConfig",
    "CheckpointConfig",
    "TraceConfig",
    "UIConfig",
    "EvalConfig",
    "load_config",
    "write_config",
    "ensure_config",
    "ForgeConfigError",
]
