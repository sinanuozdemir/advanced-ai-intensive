"""Multi-agent topology factories.

Three explicit factories that all return compiled LangGraph apps with the same
public interface: `app.invoke({"task": "..."}) -> {"answer": "...", ...}`.

- `build_solo()`         - flat baseline; one agent with all tools
- `build_supervisor()`   - supervisor that delegates to specialist workers
- `build_hierarchical()` - supervisor of supervisors (2 layers)
- `build_peer()`         - workers vote/aggregate without a supervisor
"""

from .topologies import (
    Topology,
    build_solo,
    build_supervisor,
    build_hierarchical,
    build_peer,
)
from .workers import (
    WorkerSpec,
    default_workers,
    make_researcher,
    make_summarizer,
    make_code_runner,
)

__all__ = [
    "Topology",
    "build_solo",
    "build_supervisor",
    "build_hierarchical",
    "build_peer",
    "WorkerSpec",
    "default_workers",
    "make_researcher",
    "make_summarizer",
    "make_code_runner",
]
