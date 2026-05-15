"""Procedural memory — learned skills (prompt fragments) the agent reuses.

  * A skill is just a name + a prompt fragment + a `when_to_use` hint.
  * Reflection at thread end can propose new skills (or strengthen existing
    ones by bumping `usage_count`).
  * SQLite is the source of truth for the row data (cheap exact-name lookup,
    score/usage ordering, week-2 API stability).
  * A sibling Chroma collection embeds each skill's ``when_to_use`` cue so
    the engine can do a per-turn similarity search and only inject the
    skill if the user's current message strongly matches the cue (week 3
    "just-in-time procedural recall"). The Chroma side is best-effort —
    if the collection drifts from SQLite we lazy-backfill on first read.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ProceduralSkill:
    name: str
    fragment: str            # the actual text injected into the system prompt
    when_to_use: str = ""    # one-line cue for when this skill applies
    usage_count: int = 0
    score: float = 0.0       # success-rate proxy, set by reflection
    created_at: str = ""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    name TEXT PRIMARY KEY,
    fragment TEXT NOT NULL,
    when_to_use TEXT NOT NULL,
    usage_count INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class ProceduralMemory:
    def __init__(
        self,
        path: str | Path = "data/memory/procedural.sqlite",
        *,
        when_chroma_path: str | Path | None = None,
        when_collection: str = "procedural_when",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        # Optional Chroma mirror for when_to_use similarity search. We resolve
        # it lazily so unit tests that just want SQLite semantics don't pay
        # the chromadb import cost. Pass an explicit path or it sits next to
        # the SQLite file as ``<stem>_when_chroma``.
        if when_chroma_path is None:
            when_chroma_path = self.path.parent / f"{self.path.stem}_when_chroma"
        self._when_chroma_path = Path(when_chroma_path)
        self._when_collection_name = when_collection
        self._when_coll: object | None = None  # set lazily

    # -- when_to_use vector mirror -------------------------------------------

    def _when_coll_or_none(self) -> object | None:
        """Return the Chroma collection, or None if chromadb can't be loaded."""
        if self._when_coll is not None:
            return self._when_coll
        try:
            import chromadb  # lazy import; tests without chroma still pass
            from .embedding import default_embedding_function
        except Exception:  # noqa: BLE001
            return None
        try:
            self._when_chroma_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self._when_chroma_path))
            self._when_coll = client.get_or_create_collection(
                name=self._when_collection_name,
                embedding_function=default_embedding_function(),
            )
        except Exception:  # noqa: BLE001
            return None
        return self._when_coll

    def _mirror_to_chroma(self, skill: ProceduralSkill) -> None:
        """Upsert one skill into the when_to_use vector index. No-op if
        the cue is empty or chromadb isn't available."""
        cue = (skill.when_to_use or "").strip()
        if not cue:
            return
        coll = self._when_coll_or_none()
        if coll is None:
            return
        try:
            # upsert pattern: delete then add. ``coll.upsert`` exists on
            # recent chromadb versions but we keep delete+add for breadth.
            try:
                coll.delete(ids=[skill.name])
            except Exception:  # noqa: BLE001
                pass
            coll.add(
                ids=[skill.name],
                documents=[cue],
                metadatas=[{"name": skill.name, "score": float(skill.score)}],
            )
        except Exception:  # noqa: BLE001
            pass

    def _backfill_chroma_if_stale(self) -> None:
        """Ensure every SQLite row with a non-empty when_to_use also lives in
        Chroma. Cheap: runs once per process, only when search_when is first
        called on a possibly-stale store."""
        coll = self._when_coll_or_none()
        if coll is None:
            return
        try:
            sql_names = {
                r[0] for r in self._conn.execute(
                    "SELECT name FROM skills WHERE TRIM(when_to_use) != ''"
                ).fetchall()
            }
            chroma_ids = set(coll.get().get("ids") or [])
            missing = sql_names - chroma_ids
            if not missing:
                return
            for name in missing:
                row = self._conn.execute(
                    "SELECT name, fragment, when_to_use, usage_count, score, created_at "
                    "FROM skills WHERE name=?", (name,),
                ).fetchone()
                if row:
                    self._mirror_to_chroma(ProceduralSkill(*row))
        except Exception:  # noqa: BLE001
            pass

    def search_when(
        self,
        query: str,
        *,
        k: int = 3,
        min_score: float = 0.0,
    ) -> list[tuple[ProceduralSkill, float]]:
        """Return up to ``k`` skills whose ``when_to_use`` is similar to
        ``query``, each paired with its cosine similarity (in [0, 1]).
        Filters out anything below ``min_score``. Backfills the Chroma
        index on first call so old SQLite rows aren't invisible."""
        q = (query or "").strip()
        if not q:
            return []
        self._backfill_chroma_if_stale()
        coll = self._when_coll_or_none()
        if coll is None:
            return []
        try:
            total = coll.count()
            if total == 0:
                return []
            res = coll.query(
                query_texts=[q],
                n_results=min(max(k, 1), total),
            )
        except Exception:  # noqa: BLE001
            return []
        ids_row = (res.get("ids") or [[]])[0]
        dists_row = (res.get("distances") or [[]])[0] or [0.0] * len(ids_row)
        out: list[tuple[ProceduralSkill, float]] = []
        for skill_id, dist in zip(ids_row, dists_row):
            similarity = max(0.0, 1.0 - float(dist))
            if similarity < min_score:
                continue
            row = self._conn.execute(
                "SELECT name, fragment, when_to_use, usage_count, score, created_at "
                "FROM skills WHERE name=?",
                (skill_id,),
            ).fetchone()
            if row is None:
                continue  # chroma is stale; ignore the orphan
            out.append((ProceduralSkill(*row), similarity))
        return out

    # -- core CRUD -----------------------------------------------------------

    def save(self, skill: ProceduralSkill) -> None:
        """Insert or update a skill. If it exists, bumps usage_count."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            existing = self._conn.execute(
                "SELECT usage_count FROM skills WHERE name=?", (skill.name,)
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO skills (name, fragment, when_to_use, usage_count, "
                    "score, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (skill.name, skill.fragment, skill.when_to_use,
                     skill.usage_count or 1, skill.score, skill.created_at or now, now),
                )
            else:
                self._conn.execute(
                    "UPDATE skills SET fragment=?, when_to_use=?, "
                    "usage_count=usage_count+1, score=?, updated_at=? WHERE name=?",
                    (skill.fragment, skill.when_to_use, skill.score, now, skill.name),
                )
        self._mirror_to_chroma(skill)

    def top(self, n: int = 5) -> list[ProceduralSkill]:
        rows = self._conn.execute(
            "SELECT name, fragment, when_to_use, usage_count, score, created_at "
            "FROM skills ORDER BY score DESC, usage_count DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [ProceduralSkill(*r) for r in rows]

    def all(self) -> list[ProceduralSkill]:
        return self.top(n=10_000)

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]

    def render_for_system_prompt(self, n: int = 5) -> str:
        """Render the top-n skills as a section to splice into a system prompt."""
        skills = self.top(n=n)
        if not skills:
            return ""
        lines = ["# Learned skills (procedural memory)"]
        for s in skills:
            lines.append(f"\n## {s.name}")
            if s.when_to_use:
                lines.append(f"_Use when: {s.when_to_use}_")
            lines.append(s.fragment)
        return "\n".join(lines)
