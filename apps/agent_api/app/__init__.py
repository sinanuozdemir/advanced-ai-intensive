"""agent_api — Week 3 Segment 4 deployable agentic workflow.

See `apps/agent_api/README.md` for the high-level pitch. This package
contains:

* `settings`        — env-driven config (`pydantic-settings`)
* `schemas`         — Pydantic request/response + workflow models
* `store`           — SQLite artifact store
* `logging_setup`   — JSON formatter + request_id ContextVar
* `metrics`         — Prometheus counters + histograms
* `nodes`           — the four workflow stages
* `workflow`        — LangGraph compilation
* `main`            — FastAPI app + routes
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_CONTAINER_SRC = Path("/app/src")
try:
    _REPO_SRC: Path | None = _APP_DIR.parents[2] / "src"
except IndexError:
    _REPO_SRC = None

for candidate in (_REPO_SRC, _CONTAINER_SRC):
    if candidate is not None and candidate.is_dir():
        sp = str(candidate)
        if sp not in sys.path:
            sys.path.insert(0, sp)
        break

if "PYTHONPATH" not in os.environ or "src" not in os.environ["PYTHONPATH"]:
    pass
