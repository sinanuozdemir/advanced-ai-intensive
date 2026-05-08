"""Procedural memory — learned skills (prompt fragments) the agent reuses.

This is the simplest possible version, deliberately. Week 3 will add the
"agent-edits-its-own-prompt" loop with safety rails. For week 2:

  * A skill is just a name + a prompt fragment + a `when_to_use` hint.
  * Reflection at thread end can propose new skills (or strengthen existing
    ones by bumping `usage_count`).
  * The agent's system prompt is templated to include the top-N skills by
    score whenever it boots.

Stored in SQLite for the same reason semantic.py is: cheap exact-name
lookup is the main access pattern.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
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
    def __init__(self, path: str | Path = "data/memory/procedural.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

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
