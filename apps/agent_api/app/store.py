"""SQLite-backed artifact store.

Kept intentionally tiny: one row per artifact, with the full ``Artifact`` JSON
(including ``provenance``) blobbed into a ``payload`` column. We never query
by the inner fields, only by primary key and creation time, so a single JSON
column keeps the schema migration story zero.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .schemas import Artifact, ArtifactSummary


_DDL = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id     TEXT PRIMARY KEY,
    topic           TEXT NOT NULL,
    rounds          INTEGER NOT NULL,
    findings_count  INTEGER NOT NULL,
    outcome         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    payload         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_created_at
    ON artifacts(created_at DESC);
"""


class ArtifactStore:
    """Thread-safe wrapper around a single SQLite file.

    The store is constructed once at FastAPI startup and shared across
    requests. ``check_same_thread=False`` is paired with an explicit lock so
    the in-process workers don't race on writes.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        with self._lock:
            self._migrate_if_needed()
            self._conn.executescript(_DDL)

    def _migrate_if_needed(self) -> None:
        """If the table exists but is missing the new ``rounds`` column
        (left over from the v1 ``shape -> judge`` schema), drop it. The
        artifact format changed enough that we don't try to migrate rows."""
        cur = self._conn.cursor()
        try:
            cur.execute("PRAGMA table_info(artifacts)")
            cols = {row[1] for row in cur.fetchall()}
            if cols and "rounds" not in cols:
                cur.execute("DROP TABLE artifacts")
        finally:
            cur.close()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def put(self, artifact: Artifact) -> None:
        payload = artifact.model_dump_json()
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO artifacts
                    (artifact_id, topic, rounds, findings_count, outcome, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id,
                    artifact.topic,
                    int(artifact.rounds),
                    int(artifact.findings_count),
                    artifact.outcome,
                    artifact.created_at.isoformat(),
                    payload,
                ),
            )

    def get(self, artifact_id: str) -> Artifact | None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT payload FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return Artifact.model_validate(json.loads(row[0]))

    def list(self, limit: int = 20, offset: int = 0) -> tuple[list[ArtifactSummary], int]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT artifact_id, topic, rounds, findings_count, outcome, created_at
                FROM artifacts
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (int(limit), int(offset)),
            )
            rows = cur.fetchall()
            cur.execute("SELECT COUNT(*) FROM artifacts")
            total = int(cur.fetchone()[0])
        items = [
            ArtifactSummary(
                artifact_id=r[0],
                topic=r[1],
                rounds=int(r[2]),
                findings_count=int(r[3]),
                outcome=r[4],
                created_at=r[5],
            )
            for r in rows
        ]
        return items, total

    def is_writable(self) -> bool:
        """Used by /readyz. Round-trips a probe row inside a transaction we
        roll back."""
        try:
            with self._cursor() as cur:
                cur.execute("BEGIN")
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS _readyz_probe (id INTEGER PRIMARY KEY)"
                )
                cur.execute("INSERT INTO _readyz_probe DEFAULT VALUES")
                cur.execute("ROLLBACK")
            return True
        except sqlite3.Error:
            return False

    def close(self) -> None:
        with self._lock:
            self._conn.close()
