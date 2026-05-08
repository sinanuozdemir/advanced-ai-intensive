"""OpenRouter-backed LLM helpers (notebook compatibility).

The canonical implementation lives in ``shared.openrouter_llm`` under ``src/``.
This file stays on the week-1 path so older notebook cells that run with only
``notebooks/week1`` on ``sys.path`` can still ``import llm``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
_s = str(_SRC.resolve())
if _s not in sys.path:
    sys.path.insert(0, _s)

from shared.openrouter_llm import *  # noqa: F403
from shared.openrouter_llm import _StructuredRetryWrapper  # noqa: F401 — judges._bind_judge
