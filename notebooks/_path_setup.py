"""Path setup for course notebooks. Import once at the top of every notebook:

    import _path_setup  # noqa: F401

This puts the repo's `src/` (and this `notebooks/` dir) on sys.path so
`from shared import ...`, `from multi_agent import ...`, and local helpers
like `from ep_eval import ...` all resolve.
"""

import sys
from pathlib import Path

_NOTEBOOKS = Path(__file__).resolve().parent
_REPO = _NOTEBOOKS.parent
_SRC = _REPO / "src"

for _p in (str(_SRC), str(_NOTEBOOKS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
