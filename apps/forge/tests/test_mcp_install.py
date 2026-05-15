"""validate + install of user-added MCP servers.

The only supported install kind is a JSON descriptor — Forge no longer
accepts ``.py`` drops directly. To keep the tests self-contained we ship a
tiny FastMCP server source string, drop it into ``tmp_path`` as a real .py
file, then point a JSON descriptor at it. That mirrors what a real user
does (their server lives in their own repo; the descriptor is the only
thing Forge owns).

Skipped if the ``mcp`` SDK isn't importable in this Python env (so this
suite doesn't fail on dev machines without the optional dep set).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp.server.fastmcp")  # noqa: F841

from forge import mcp as M
from forge.mcp import (
    install_server,
    is_valid_name,
    list_servers_for_api,
    validate_server,
)
from forge.paths import ForgePaths


_MINIMAL_SERVER = """\
from mcp.server.fastmcp import FastMCP
m = FastMCP("smoke")

@m.tool()
def echo(text: str) -> str:
    \"\"\"Return the input text.\"\"\"
    return text

if __name__ == "__main__":
    m.run()
"""


def _paths(tmp_path: Path) -> ForgePaths:
    (tmp_path / ".forge").mkdir(exist_ok=True)
    return ForgePaths.for_repo(tmp_path).ensure()


def _drop_server_script(tmp_path: Path) -> Path:
    """Write the FastMCP test script to ``tmp_path`` and return its path."""
    script = tmp_path / "smoke_server.py"
    script.write_text(_MINIMAL_SERVER, encoding="utf-8")
    return script


def _descriptor(script: Path, *, env: dict[str, str] | None = None) -> str:
    """Build a JSON descriptor pointing at the on-disk script.

    Using ``sys.executable`` rather than ``"python"`` matches what real
    users do on macOS, where ``python`` often isn't on PATH for non-shell
    subprocess spawns.
    """
    spec: dict = {
        "command": sys.executable,
        "args": [str(script)],
    }
    if env:
        spec["env"] = env
    return json.dumps(spec)


def test_is_valid_name() -> None:
    assert is_valid_name("ok")
    assert is_valid_name("ok_name")
    assert is_valid_name("ok-name")
    assert not is_valid_name("1bad")  # must start with a letter
    assert not is_valid_name("")
    assert not is_valid_name("x" * 50)


@pytest.mark.asyncio
async def test_validate_json_descriptor(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    script = _drop_server_script(tmp_path)
    result = await validate_server(paths=paths, contents=_descriptor(script))
    assert result["ok"] is True
    names = [t["name"] for t in result["tools"]]
    assert "echo" in names


@pytest.mark.asyncio
async def test_validate_rejects_missing_command(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    # No 'command' field => descriptor invalid; we expect a clean error,
    # not a TaskGroup unwrap.
    bad = json.dumps({"args": ["nope"]})
    result = await validate_server(paths=paths, contents=bad)
    assert result["ok"] is False
    assert "command" in (result["error"] or "")


@pytest.mark.asyncio
async def test_validate_surfaces_spawn_failure(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    # Point at a binary that does not exist. The adapter will wrap the
    # FileNotFoundError in an ExceptionGroup; ``_format_exc`` unwraps it so
    # the user sees a sensible message instead of "TaskGroup (1 sub-exc)".
    bad = json.dumps({
        "command": "nonexistent_python_binary_xyz",
        "args": ["does_not_matter.py"],
    })
    result = await validate_server(paths=paths, contents=bad)
    assert result["ok"] is False
    err = result["error"] or ""
    assert "FileNotFoundError" in err or "No such file" in err


@pytest.mark.asyncio
async def test_install_and_manifest_roundtrip(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    script = _drop_server_script(tmp_path)

    out = await install_server(
        paths=paths,
        name="smoketest",
        contents=_descriptor(script, env={"SMOKE_VAR": "1"}),
        description="smoke test server",
    )
    assert out["ok"] is True
    assert out["pending_restart"] is True
    assert out["entry"]["name"] == "smoketest"
    assert out["entry"]["kind"] == "json"
    # Descriptor landed on disk as <name>.json.
    descriptor_path = M.user_servers_dir(paths.forge_dir) / "smoketest.json"
    assert descriptor_path.is_file()
    persisted = json.loads(descriptor_path.read_text(encoding="utf-8"))
    assert persisted["env"] == {"SMOKE_VAR": "1"}

    # Manifest reflects it.
    mf = M.load_manifest(paths.forge_dir)
    assert {s.name for s in mf.servers} == {"smoketest"}
    assert mf.servers[0].kind == "json"

    # And so does the API listing.
    listing = list_servers_for_api(paths)
    rows_by_name = {row["name"]: row for row in listing["servers"]}
    assert "smoketest" in rows_by_name
    assert rows_by_name["smoketest"]["kind"] == "user_json"
    assert "echo" in (rows_by_name["smoketest"].get("tools") or [])
    assert listing["pending_restart"] is True


@pytest.mark.asyncio
async def test_install_refuses_builtin_name_collision(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    script = _drop_server_script(tmp_path)
    with pytest.raises(ValueError, match="collides"):
        await install_server(
            paths=paths, name="fs", contents=_descriptor(script),
        )


@pytest.mark.asyncio
async def test_install_refuses_duplicate_user_name(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    script = _drop_server_script(tmp_path)
    await install_server(
        paths=paths, name="dupe", contents=_descriptor(script),
    )
    with pytest.raises(ValueError, match="already installed"):
        await install_server(
            paths=paths, name="dupe", contents=_descriptor(script),
        )


def test_legacy_python_manifest_entries_are_skipped(tmp_path: Path) -> None:
    """A manifest that still carries an old ``kind: "python"`` entry should
    load without raising; the entry just disappears from the runtime view."""
    paths = _paths(tmp_path)
    M.manifest_path(paths.forge_dir).write_text(
        json.dumps({
            "version": 1,
            "servers": [
                {"name": "legacy_py", "kind": "python", "path": "legacy.py"},
                {"name": "modern", "kind": "json", "path": "modern.json"},
            ],
        }),
        encoding="utf-8",
    )
    mf = M.load_manifest(paths.forge_dir)
    assert [s.name for s in mf.servers] == ["modern"]
