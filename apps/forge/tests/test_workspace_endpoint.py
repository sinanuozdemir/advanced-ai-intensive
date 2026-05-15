"""GET /api/workspace — repo metadata for the Electron header strip."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from forge.paths import ForgePaths
from forge.server import _workspace_snapshot, build_app


def test_workspace_snapshot_in_real_repo(tmp_path: Path) -> None:
    """In a freshly-initialized git repo, snapshot returns the branch and a
    clean dirty=False. Then dirty a file and re-check."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    # Configure a fake identity so `git commit` works in CI.
    subprocess.run(["git", "config", "user.email", "f@example"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "f"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    snap = _workspace_snapshot(tmp_path)
    assert snap["is_git"] is True
    assert snap["branch"] == "main"
    assert snap["dirty"] is False
    assert snap["head"] and len(snap["head"]) >= 7

    # Now dirty the tree.
    (tmp_path / "README.md").write_text("hello changed\n")
    snap2 = _workspace_snapshot(tmp_path)
    assert snap2["dirty"] is True


def test_workspace_snapshot_non_repo(tmp_path: Path) -> None:
    """A bare directory: ``is_git=False``, branch and head None, dirty False."""
    snap = _workspace_snapshot(tmp_path)
    assert snap == {
        "repo_root": str(tmp_path),
        "branch": None,
        "head": None,
        "dirty": False,
        "is_git": False,
    }


def test_workspace_endpoint_returns_200(tmp_path: Path) -> None:
    """End-to-end through FastAPI: the route is wired and returns JSON."""
    # Initialize a .forge/ so ForgePaths picks tmp_path as the root.
    (tmp_path / ".forge").mkdir()
    paths = ForgePaths.for_repo(tmp_path).ensure()
    app = build_app(paths)
    with TestClient(app) as client:
        r = client.get("/api/workspace")
        assert r.status_code == 200
        data = r.json()
        assert data["repo_root"] == str(tmp_path)
        assert "dirty" in data
        assert "is_git" in data
