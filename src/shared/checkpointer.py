"""LangGraph checkpointer factory.

A thin wrapper around LangGraph's `SqliteSaver` / `PostgresSaver` so the same
agent code can pick the right persistence backend based on environment.

Usage from a notebook:

    from shared import make_checkpointer
    cp = make_checkpointer("dev")             # SQLite at data/checkpoints.sqlite
    graph = builder.compile(checkpointer=cp)

Used by:
- `src/multi_agent/topologies.py` (any compiled graph that wants resumability)
- `apps/sdr_multi_agent/flask_app/checkpoint_hooks.py` (the SDR Celery task)
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

DEFAULT_SQLITE_PATH = Path("data/checkpoints.sqlite")


async def make_async_postgres_checkpointer(database_url: str):
    """Return `AsyncPostgresSaver` bound to a pool created on the running loop.

    Use this from async code (the SDR Flask + Celery agents call ``ainvoke``).
    The sync ``PostgresSaver`` will silently misbehave inside an asyncio loop.
    """
    try:
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Async Postgres checkpoints need langgraph-checkpoint-postgres + "
            "psycopg[binary,pool]. "
            "Run: pip install 'langgraph-checkpoint-postgres' 'psycopg[binary,pool]'"
        ) from exc

    pool = AsyncConnectionPool(
        database_url,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=False,
    )
    await pool.open()
    saver = AsyncPostgresSaver(pool)
    try:
        await saver.setup()
    except Exception:  # noqa: BLE001 - idempotent setup; ignore "already exists"
        pass
    return saver


async def make_async_sqlite_checkpointer(path: Path | None = None):
    """Return `AsyncSqliteSaver` for graphs that use ``ainvoke`` / ``aget_state``.

    Must be called from a running asyncio loop (e.g. inside ``async def initialize``).
    Requires `aiosqlite`.
    """
    try:
        import aiosqlite  # noqa: F401
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Async SQLite checkpoints need aiosqlite. Run: pip install aiosqlite"
        ) from exc

    db = Path(path or DEFAULT_SQLITE_PATH)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db))
    return AsyncSqliteSaver(conn)


def make_checkpointer(env: str = "dev", *, path: Path | None = None):
    """Return a LangGraph checkpointer appropriate for `env`.

    Parameters
    ----------
    env : {"dev", "test", "prod"}
        - "dev"/"test" -> SQLite saver at `path` (defaults to data/checkpoints.sqlite)
        - "prod"       -> Postgres saver if `DATABASE_URL` is set, else SQLite

    Returns
    -------
    A LangGraph BaseCheckpointSaver subclass instance, ready to pass to
    `graph.compile(checkpointer=...)`.
    """
    if env == "prod" and os.environ.get("DATABASE_URL"):
        return _make_postgres_checkpointer(os.environ["DATABASE_URL"])
    return _make_sqlite_checkpointer(path or DEFAULT_SQLITE_PATH)


def _make_sqlite_checkpointer(path: Path):
    try:
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:  # pragma: no cover - optional dep guard
        raise RuntimeError(
            "langgraph-checkpoint-sqlite not installed. "
            "Run: pip install langgraph-checkpoint-sqlite"
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # NB: do NOT use SqliteSaver.from_conn_string — it wraps the connection in
    # `closing(...)` and yields the saver, so the connection dies as soon as
    # the generator is GC'd. We own the connection here so the saver outlives
    # any one cell or function call.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def _make_postgres_checkpointer(database_url: str):
    try:
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "langgraph-checkpoint-postgres + psycopg-pool not installed. "
            "Run: pip install 'langgraph-checkpoint-postgres' 'psycopg[binary,pool]'"
        ) from exc

    # Same fix as the sqlite path: own the pool here so its lifetime is the
    # process, not the call site. PostgresSaver(conn) accepts a pool directly.
    pool = ConnectionPool(database_url, kwargs={"autocommit": True, "row_factory": None})
    saver = PostgresSaver(pool)
    if hasattr(saver, "setup"):
        try:
            saver.setup()
        except Exception:
            pass
    return saver


@contextmanager
def checkpointer(env: str = "dev", *, path: Path | None = None):
    """Context-manager variant. Use when you want explicit cleanup."""
    cp = make_checkpointer(env, path=path)
    try:
        yield cp
    finally:
        # SqliteSaver wraps a sqlite3 connection; close it on exit.
        for attr in ("conn", "_conn"):
            obj = getattr(cp, attr, None)
            if obj is not None and hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:
                    pass
                break
