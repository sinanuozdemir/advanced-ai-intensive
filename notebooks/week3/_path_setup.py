"""Path setup for week3 notebooks. Import once at the top of every notebook:

    import _path_setup  # noqa: F401

Puts the repo's `src/` and `notebooks/week1/` on sys.path so `from shared
import ...`, `from judges import ...`, and `from llm import ...` all resolve.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (_REPO / "src", _REPO / "notebooks" / "week1"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
