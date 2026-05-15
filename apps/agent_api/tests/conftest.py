"""Shared test fixtures.

Most tests need a temp-dir-backed ``Settings`` and ``ArtifactStore`` so they
don't write to the real ``runtime/`` directory. A few tests also need
``llm_smoke_check`` to return ok without a real key — those monkeypatch
``app.deps.llm_smoke_check`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_APP_DIR = _HERE.parent
_REPO_SRC = _APP_DIR.parents[1] / "src"

for path in (_APP_DIR, _REPO_SRC):
    sp = str(path)
    if sp not in sys.path:
        sys.path.insert(0, sp)


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """Build a Settings rooted in a tmp dir and install it as the singleton."""
    from app.settings import Settings, set_settings

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
    s = Settings(
        OPENROUTER_API_KEY="sk-test-fake",
        ARTIFACTS_DB_PATH=tmp_path / "artifacts.db",
        LOG_FILE_PATH=tmp_path / "agent_api.log",
    )
    set_settings(s)
    yield s
    set_settings(Settings())


@pytest.fixture
def store(tmp_settings):
    from app.store import ArtifactStore
    s = ArtifactStore(tmp_settings.artifacts_db_path)
    yield s
    s.close()
