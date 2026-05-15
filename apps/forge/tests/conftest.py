"""Pytest config for Forge backend tests.

These tests *do not* boot a real ForgeEngine — they exercise the FastAPI
routes and `_ServerState` directly. The engine boot path requires Chroma
+ MCP subprocesses and is covered by the eval harness, not this suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure both the repo's `src/` (for `shared.*`) and `apps/forge/` (for
# `forge.*`) are importable when pytest is invoked from anywhere.
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
for p in (_REPO_ROOT / "src", _REPO_ROOT / "apps" / "forge"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
