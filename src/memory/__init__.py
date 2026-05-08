"""Long-term memory subsystems for week 2.

Three independent stores, each one a teaching surface for a different idea:

  semantic.py    - free-form natural-language memories the agent writes via
                   an explicit tool. Retrieved by similarity (Chroma).
  episodic.py    - past thread summaries, written via reflection at thread
                   end, retrieved with vector similarity.
  procedural.py  - learned prompt fragments / "skills" the agent can paste
                   into its system prompt. Written via reflection.

A `reflect.py` module orchestrates the end-of-thread writes for episodic +
procedural memory. Semantic writes are no longer automatic — the agent must
choose to call the ``semantic_write`` tool.
"""
from __future__ import annotations

from .semantic import SemanticMemory, SemanticMemoryRecord
from .episodic import EpisodicMemory, EpisodicEntry
from .procedural import ProceduralMemory, ProceduralSkill
from .reflect import reflect_on_thread

__all__ = [
    "SemanticMemory",
    "SemanticMemoryRecord",
    "EpisodicMemory", "EpisodicEntry",
    "ProceduralMemory", "ProceduralSkill",
    "reflect_on_thread",
]
