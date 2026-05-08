"""Per-conversation log of Celery sub-tasks dispatched by the supervisor.

The supervisor's `delegate_*` tools enqueue Celery jobs (one of the existing
per-config agents). We mirror each dispatch into a tiny SQLite table so the
Flask app can answer `GET /api/subtasks/<conversation_id>` and the UI can
render a live panel without having to ask Celery for every task it ever saw.

The Celery result backend is still the source of truth for *current* status —
the Flask route reconciles each row against `get_task_result(task_id)` on
every poll and writes the new state back here.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# In the repo layout this file lives 3 levels under the repo root; in the
# container it lives at /app/. We support both by walking up looking for a
# sibling `src/` and falling back to the file's own parent. Override either
# location with `AGENT_DATA_ROOT`.
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return here.parent


_REPO_ROOT = _find_repo_root()
_DATA_ROOT = Path(os.environ.get("AGENT_DATA_ROOT", _REPO_ROOT / "data" / "sdr_runtime"))
_DATA_ROOT.mkdir(parents=True, exist_ok=True)

_DB_PATH = _DATA_ROOT / "subtasks.sqlite"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS subtasks (
    task_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    agent_slug TEXT NOT NULL,
    agent_config TEXT NOT NULL,
    message TEXT NOT NULL,
    status TEXT NOT NULL,
    result_summary TEXT,
    thread_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_subtasks_conv ON subtasks(conversation_id, created_at);
"""

_COLUMNS = (
    "task_id, conversation_id, agent_slug, agent_config, message, "
    "status, result_summary, thread_id, created_at, updated_at"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SubtaskLog:
    def __init__(self, db_path: Path | str = _DB_PATH):
        self._path = str(db_path)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)
            cols = {row[1] for row in conn.execute("PRAGMA table_info(subtasks)").fetchall()}
            if "thread_id" not in cols:
                conn.execute("ALTER TABLE subtasks ADD COLUMN thread_id TEXT")

    def record(
        self,
        *,
        task_id: str,
        conversation_id: str,
        agent_slug: str,
        agent_config: str,
        message: str,
        status: str = "PENDING",
        thread_id: str | None = None,
    ) -> None:
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO subtasks
                   (task_id, conversation_id, agent_slug, agent_config, message,
                    status, result_summary, thread_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)""",
                (task_id, conversation_id, agent_slug, agent_config,
                 message, status, thread_id, now, now),
            )

    def update_status(
        self,
        task_id: str,
        status: str,
        result_summary: str | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            if result_summary is None:
                conn.execute(
                    "UPDATE subtasks SET status=?, updated_at=? WHERE task_id=?",
                    (status, _now(), task_id),
                )
            else:
                conn.execute(
                    """UPDATE subtasks
                       SET status=?, result_summary=?, updated_at=?
                       WHERE task_id=?""",
                    (status, result_summary, _now(), task_id),
                )

    def search(
        self,
        *,
        query: str | None = None,
        agent_slug: str | None = None,
        status: str | None = None,
        conversation_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search across all conversations.

        - ``query`` matches anywhere in ``message`` or ``result_summary`` (LIKE).
        - ``agent_slug``/``status``/``conversation_id`` are exact filters.
        Newest rows first.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if query:
            clauses.append("(message LIKE ? OR IFNULL(result_summary,'') LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like])
        if agent_slug:
            clauses.append("agent_slug = ?")
            params.append(agent_slug)
        if status:
            clauses.append("status = ?")
            params.append(status.upper())
        if conversation_id:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            f"SELECT {_COLUMNS} "
            f"FROM subtasks {where} ORDER BY created_at DESC LIMIT ?"
        )
        params.append(int(limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM subtasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_for_conversation(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM subtasks "
                "WHERE conversation_id=? ORDER BY created_at ASC",
                (conversation_id,),
            ).fetchall()
        return [dict(r) for r in rows]
