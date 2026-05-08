"""Path setup for week2 notebooks. Import once at the top of every notebook:

    import _path_setup  # noqa: F401

This puts the repo's `src/` directory on sys.path so `from shared import ...`,
`from multi_agent import ...`, etc. all resolve.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
