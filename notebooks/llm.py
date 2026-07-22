"""OpenRouter-backed LLM helpers (notebook compatibility).

The canonical implementation lives in ``shared.openrouter_llm`` under ``src/``.
This file stays under ``notebooks/`` so course cells can ``import llm`` when
that directory is on ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_s = str(_SRC.resolve())
if _s not in sys.path:
    sys.path.insert(0, _s)

from shared.openrouter_llm import *  # noqa: F403
from shared.openrouter_llm import _StructuredRetryWrapper  # noqa: F401 — judges._bind_judge
