"""Forge — a TUI coding agent built backwards from an eval harness.

Importable surface (lazy where useful):

    from forge.config import ForgeConfig, load_config, write_config
    from forge.paths import ForgePaths

The CLI entry is `forge.cli:main`.
"""
from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.0.1"


def _bootstrap_src_onto_path() -> None:
    """Forge wraps several modules in ``../../src`` (shared, memory, middleware,
    multi_agent). They aren't pip-installed; the repo convention is to insert
    them on sys.path at import time. Same trick the SDR app uses."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        src = candidate / "src"
        if (src / "shared" / "__init__.py").is_file():
            s = str(src)
            if s not in sys.path:
                sys.path.insert(0, s)
            return


_bootstrap_src_onto_path()
