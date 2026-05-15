"""MCP server lifecycle: manifest, loader, validator, installer.

Three concerns live here, all small enough that splitting them into a
package would cost more in cross-imports than it saves in size:

1. **User manifest** (was ``forge/mcp/manifest.py``). Reads/writes
   ``<repo>/.forge/mcp_servers.json``. User entries are JSON descriptors —
   the same shape Claude Desktop and Cursor accept — and that is the only
   install kind Forge supports. (We dropped the legacy ``.py``-drop kind
   because it uploaded executable code into Forge's process tree with no
   place to declare env vars; JSON descriptors fix both issues.)

2. **Tool loader** (was ``forge/mcp/tool_loader.py``). Spawns Forge's
   built-in MCP servers + any user-added ones as stdio subprocesses,
   pulls the LangChain ``BaseTool`` list out of each, prefixes the names
   to ``"<server>_<tool>"``, and wraps every tool in a permission gate.

3. **Validate + install** (was ``forge/mcp_install.py``). Backs
   ``POST /api/mcp/validate`` and ``POST /api/mcp/install`` by spawning
   the candidate, listing its tools, then either reporting back or moving
   the descriptor into ``.forge/mcp_servers/<name>.json``.

The repo-RAG indexer (the *other* MCP-shaped responsibility) lives in
``forge.repo_rag`` because it's a heavy module that pulls Chroma, BM25,
and embeddings — unrelated to MCP-the-protocol.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool

from .agent.permissions import PermissionBroker, PermissionDenied
from .paths import ForgePaths
from .trace import Tracer


# ---------------------------------------------------------------------------
# Manifest (was forge/mcp/manifest.py)
# ---------------------------------------------------------------------------


MANIFEST_NAME = "mcp_servers.json"
USER_SERVERS_DIR = "mcp_servers"  # within .forge/

McpKind = Literal["json"]


@dataclass
class UserMcpServer:
    """One entry in the manifest.

    Args:
        name: stable slug (also the keyword the tool loader uses to identify
            the server). Must match ``[a-zA-Z0-9_-]+``.
        kind: always ``"json"``. Retained as a field so the on-disk schema
            is forward-compatible if we ever add a second kind back.
        path: filename inside ``.forge/mcp_servers/`` holding the JSON
            descriptor.
        description: optional human label for the UI.
    """

    name: str
    kind: McpKind
    path: str
    description: str = ""
    # Optional bag for forward-compat (e.g. "tools": [...]).
    extra: dict = field(default_factory=dict)


@dataclass
class Manifest:
    version: int = 1
    servers: list[UserMcpServer] = field(default_factory=list)

    def by_name(self) -> dict[str, UserMcpServer]:
        return {s.name: s for s in self.servers}


def manifest_path(forge_dir: Path) -> Path:
    return forge_dir / MANIFEST_NAME


def user_servers_dir(forge_dir: Path) -> Path:
    return forge_dir / USER_SERVERS_DIR


def load_manifest(forge_dir: Path) -> Manifest:
    """Read the manifest. Returns an empty one if the file doesn't exist or
    is malformed — we never throw, because the boot path calls this and a
    broken manifest shouldn't break Forge."""
    p = manifest_path(forge_dir)
    if not p.is_file():
        return Manifest()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Manifest()
    servers: list[UserMcpServer] = []
    for raw in data.get("servers", []) or []:
        try:
            kind = raw.get("kind", "json")
            if kind != "json":
                # Legacy entry from when Forge accepted ``.py`` drops. Skip it
                # — keep the entry on disk so the user can see what they have
                # and re-install via a JSON descriptor.
                continue
            servers.append(
                UserMcpServer(
                    name=str(raw["name"]),
                    kind="json",
                    path=str(raw["path"]),
                    description=str(raw.get("description", "")),
                    extra=dict(raw.get("extra") or {}),
                ),
            )
        except (KeyError, TypeError):
            # Skip malformed entries instead of crashing the whole manifest.
            continue
    return Manifest(version=int(data.get("version", 1)), servers=servers)


def save_manifest(forge_dir: Path, manifest: Manifest) -> None:
    """Atomic write so a partial flush can't leave us with an empty manifest."""
    p = manifest_path(forge_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    user_servers_dir(forge_dir).mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    payload = {
        "version": manifest.version,
        "servers": [asdict(s) for s in manifest.servers],
    }
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


# ---------------------------------------------------------------------------
# Tool loader (was forge/mcp/tool_loader.py)
# ---------------------------------------------------------------------------
#
# Returned tools are LangChain BaseTools whose names are ``"<server>_<tool>"``
# (e.g. ``fs_read``). The renaming is deliberate — it matches the keys used in
# ``permissions.tools`` in the config TOML and in the audit log. Servers are
# started as subprocesses over stdio. We pin them to the repo Forge is
# operating on via the ``FORGE_REPO`` env var.


MCP_DIR = Path(__file__).resolve().parent.parent / "mcp_servers"


def _server_spec(name: str, script: str, repo_root: Path) -> dict:
    # IMPORTANT: only the *parent* of the ``forge`` package goes on PYTHONPATH.
    # Adding the package dir itself would put a file named ``mcp`` (now this
    # module) at top level and shadow the real ``mcp`` SDK that the server
    # imports as ``mcp.server.fastmcp``.
    _ = name
    return {
        "command": sys.executable,
        "args": [str(MCP_DIR / script)],
        "transport": "stdio",
        "env": _base_env(repo_root),
    }


def _base_env(repo_root: Path) -> dict[str, str]:
    return {
        "FORGE_REPO": str(repo_root),
        "PYTHONPATH": str(MCP_DIR.parent),  # apps/forge/
        "PATH": _path_env(),
        "HOME": _home_env(),
    }


def _user_json_spec(descriptor_path: Path, repo_root: Path) -> dict | None:
    """Read a JSON descriptor and turn it into a MultiServerMCPClient spec.

    The descriptor format is intentionally minimal:

        {
          "command": "uvx",
          "args": ["some-mcp-server"],
          "env": { "FOO": "bar" }
        }

    Anything else in the file is ignored. We always inject ``FORGE_REPO``
    into the subprocess env so cooperating servers can pick it up.
    """
    try:
        spec = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    command = spec.get("command")
    args = spec.get("args") or []
    if not isinstance(command, str) or not command.strip():
        return None
    env = _base_env(repo_root)
    extra_env = spec.get("env") or {}
    if isinstance(extra_env, dict):
        env.update({str(k): str(v) for k, v in extra_env.items()})
    return {
        "command": command,
        "args": [str(a) for a in args],
        "transport": "stdio",
        "env": env,
    }


def _path_env() -> str:
    return os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")


def _home_env() -> str:
    return os.environ.get("HOME", "/")


SERVERS = {
    "fs": "fs_server.py",
    "shell": "shell_server.py",
    "git": "git_server.py",
    "repo_rag": "repo_rag_server.py",
    "code": "code_server.py",
}


@dataclass
class _RawTool:
    """The original MCP-adapter tool + its decided gated name + originating server."""

    raw: BaseTool
    gated_name: str
    server: str = ""  # e.g. "forge-fs", "forge-user-my-thing", or "" if unknown


@dataclass
class LoadedTools:
    tools: list[BaseTool]
    client: Any  # MultiServerMCPClient; held for lifetime control
    _raws: list[_RawTool] = None  # type: ignore[assignment]
    _broker: Any = None
    _tracer: Any = None

    def names(self) -> list[str]:
        return [t.name for t in self.tools]

    def inventory(self) -> list[dict]:
        """Return one dict per gated tool: ``{name, description, server}``.

        ``server`` is the human-friendly slug ("fs", "shell", "user_my_thing")
        rather than the internal MCP key ("forge-fs"). Unknown originating
        servers come back as the empty string so the UI can bucket them
        under an "other" group instead of dropping them silently.
        """
        raws = self._raws or []
        rows: list[dict] = []
        for gated, raw in zip(self.tools, raws, strict=False):
            rows.append({
                "name": gated.name,
                "description": (gated.description or "").strip(),
                "server": _prefix_for_server(raw.server) if raw.server else "",
            })
        return rows

    def wrap_for_agent(self, agent_name: str) -> list[BaseTool]:
        """Return a fresh list of permission-gated tools whose ``agent_name`` is
        ``agent_name`` (instead of the default ``"main"``). Used to give
        persistent agents their own deny-by-default identity at the broker."""
        if not self._raws:
            return list(self.tools)
        return [
            _wrap_with_gate(
                r.raw, gated_name=r.gated_name, broker=self._broker,
                agent_name=agent_name, tracer=self._tracer,
            )
            for r in self._raws
        ]


async def load_mcp_tools(
    *,
    paths: ForgePaths,
    broker: PermissionBroker,
    agent_name: str = "main",
    tracer: Tracer | None = None,
    enabled_servers: list[str] | None = None,
) -> LoadedTools:
    """Boot the configured MCP servers and return permission-gated tools.

    Built-in servers from ``SERVERS`` and user-added ones from
    ``<repo>/.forge/mcp_servers.json`` are merged before launch. User names
    can't override built-in names — we skip duplicates so a user can't
    accidentally shadow ``fs`` or ``shell`` and inherit those tools'
    permission gates.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    builtin_filter = enabled_servers if enabled_servers is not None else list(SERVERS.keys())
    spec_map: dict[str, dict] = {
        f"forge-{name}": _server_spec(name, script, paths.repo_root)
        for name, script in SERVERS.items()
        if name in builtin_filter
    }

    # Layer user servers on top, refusing to clobber built-ins.
    # ``load_manifest`` already skips legacy ``kind: "python"`` entries; the
    # only kind we see here is ``"json"``.
    user_dir = user_servers_dir(paths.forge_dir)
    for entry in load_manifest(paths.forge_dir).servers:
        if entry.name in SERVERS:
            continue
        if enabled_servers is not None and entry.name not in enabled_servers:
            continue
        key = f"forge-user-{entry.name}"
        spec = _user_json_spec(user_dir / entry.path, paths.repo_root)
        if spec is not None:
            spec_map[key] = spec

    client = MultiServerMCPClient(spec_map)

    # Rename each tool to "<server>_<tool>". The name passed to the LLM
    # must match OpenAI's regex (^[a-zA-Z0-9_-]+$) so we map every server
    # key to an underscore-safe prefix (``forge-fs`` -> ``fs``,
    # ``forge-user-foo`` -> ``user_foo``).
    #
    # Attribution is canonical now: we ask the adapter for each server's
    # tools individually (``server_name=...``) instead of relying on a
    # name-sniffing heuristic. That makes user-installed MCP tools show up
    # under their actual server in the picker AND prevents two user
    # servers exposing identically-named tools from colliding under a
    # nameless ``_2`` suffix.
    gated: list[BaseTool] = []
    raws: list[_RawTool] = []
    seen_names: set[str] = set()
    for server_key in spec_map.keys():
        prefix = _prefix_for_server(server_key)
        try:
            server_tools = await client.get_tools(server_name=server_key)
        except Exception as exc:  # noqa: BLE001
            # One broken server shouldn't take the whole stack down.
            import sys
            print(
                f"forge: failed to load MCP server {server_key}: {exc}",
                file=sys.stderr,
            )
            continue
        for tool in server_tools:
            gated_name = f"{prefix}_{tool.name}" if prefix else tool.name
            if gated_name in seen_names:
                i = 2
                while f"{gated_name}_{i}" in seen_names:
                    i += 1
                gated_name = f"{gated_name}_{i}"
            seen_names.add(gated_name)
            gated.append(_wrap_with_gate(
                tool, gated_name=gated_name, broker=broker,
                agent_name=agent_name, tracer=tracer,
            ))
            raws.append(_RawTool(
                raw=tool, gated_name=gated_name, server=server_key,
            ))
    return LoadedTools(
        tools=gated, client=client, _raws=raws, _broker=broker, _tracer=tracer,
    )


def _prefix_for_server(server: str) -> str:
    """Map a server key (``"forge-fs"`` / ``"forge-user-my-thing"``) to a
    tool-name-safe prefix (``"fs"`` / ``"user_my_thing"``).

    Hyphens are valid for the LLM tool-name regex but conflict with config
    keys (``permissions.tools.fs_write``), so we normalize to underscores.
    """
    s = server
    if s.startswith("forge-user-"):
        return "user_" + s[len("forge-user-"):].replace("-", "_")
    if s.startswith("forge-"):
        return s[len("forge-"):].replace("-", "_")
    return s.replace("-", "_")


def _wrap_with_gate(
    tool: BaseTool,
    *,
    gated_name: str,
    broker: PermissionBroker,
    agent_name: str,
    tracer: Tracer | None,
) -> BaseTool:
    """Return a new tool whose runner calls the broker before the underlying tool."""
    original_arun = tool.coroutine
    original_run = tool.func

    async def _gated_arun(**kwargs: Any) -> Any:
        if tracer is not None:
            tracer.emit("tool_call", agent_name=agent_name, tool=gated_name, args=kwargs)
        try:
            await broker.gate(
                tool_name=gated_name, args=kwargs, agent_name=agent_name,
            )
        except PermissionDenied as exc:
            msg = f"DENIED: {exc}"
            if tracer is not None:
                tracer.emit(
                    "tool_result", agent_name=agent_name, tool=gated_name,
                    ok=False, preview=msg[:200],
                )
            return msg
        try:
            if original_arun is not None:
                result = await original_arun(**kwargs)
            else:
                # The MCP adapter always provides coroutine; this is defensive.
                result = await asyncio.to_thread(original_run, **kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            err = f"ERROR: {type(exc).__name__}: {exc}"
            if tracer is not None:
                tracer.emit(
                    "tool_result", agent_name=agent_name, tool=gated_name,
                    ok=False, preview=err[:200],
                )
            return err
        if tracer is not None:
            preview = _preview(result)
            tracer.emit(
                "tool_result", agent_name=agent_name, tool=gated_name,
                ok=True, preview=preview,
            )
        return result

    # Create a StructuredTool with the gated coroutine; sync .func raises so
    # we always go through the async path.
    def _sync_disallowed(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"Tool {gated_name} can only be invoked asynchronously "
            "(permission gating is async)."
        )

    return StructuredTool(
        name=gated_name,
        description=tool.description or "",
        args_schema=tool.args_schema,
        coroutine=_gated_arun,
        func=_sync_disallowed,
    )


def _preview(value: Any, n: int = 240) -> str:
    s = str(value)
    return s if len(s) <= n else s[: n - 3] + "..."


def list_known_servers(paths: ForgePaths) -> list[dict]:
    """Combined view of built-in + user-added servers (no MCP boot required).

    Returned by the ``/api/mcp`` endpoint so the Electron MCP tab can render
    without paying the server-spawn cost. Each entry has::

        {"name": str, "kind": "builtin" | "user_json",
         "source": str | None, "description": str}
    """
    out: list[dict] = []
    for name, script in SERVERS.items():
        out.append({
            "name": name,
            "kind": "builtin",
            "source": str(MCP_DIR / script),
            "description": "",
        })
    for entry in load_manifest(paths.forge_dir).servers:
        out.append({
            "name": entry.name,
            "kind": "user_json",
            "source": str(user_servers_dir(paths.forge_dir) / entry.path),
            "description": entry.description,
        })
    return out


# ---------------------------------------------------------------------------
# Validate + install (was forge/mcp_install.py)
# ---------------------------------------------------------------------------
#
# Used by the FastAPI server to back /api/mcp/validate and /api/mcp/install.
# Strategy: write the descriptor to a temp file, spawn it with the same env
# we use for built-ins, list its tools through the MCP adapter, then tear
# down. Install does the same validation pass first, then persists.


_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,40}$")
_VALIDATE_TIMEOUT_S = 15.0


def is_valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name))


@asynccontextmanager
async def _temp_server(contents: str):
    """Write ``contents`` to a temp ``.json`` file and yield the path."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False,
    ) as f:
        f.write(contents)
        path = Path(f.name)
    try:
        yield path
    finally:
        try:
            path.unlink()
        except OSError:
            pass


async def validate_server(
    *,
    paths: ForgePaths,
    contents: str,
) -> dict:
    """Spawn the candidate over stdio, list its tools, and tear it down.

    ``contents`` is the raw text of a JSON descriptor:
    ``{"command": ..., "args": [...], "env": {...}}``.

    Returns ``{"ok": bool, "tools": list[{name, description}], "error": str | None}``.
    """
    async with _temp_server(contents) as tmp:
        spec = _user_json_spec(tmp, paths.repo_root)
        if spec is None:
            return {
                "ok": False,
                "tools": [],
                "error": (
                    "json descriptor is missing a 'command' field "
                    "or is not valid JSON"
                ),
            }

        # Probe via MultiServerMCPClient with a wall-clock timeout so a
        # hung server can't pin the request.
        try:
            tools = await asyncio.wait_for(
                _list_tools_via_adapter(spec), timeout=_VALIDATE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "tools": [],
                "error": (
                    f"server did not respond to tools/list within "
                    f"{_VALIDATE_TIMEOUT_S}s"
                ),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "tools": [],
                "error": _format_exc(exc),
            }

    if not tools:
        return {
            "ok": False,
            "tools": [],
            "error": "server reported zero tools",
        }
    return {"ok": True, "tools": tools, "error": None}


def _format_exc(exc: BaseException) -> str:
    """Render an exception as ``Type: message``, drilling into ExceptionGroup.

    ``langchain_mcp_adapters`` uses anyio TaskGroups internally, so any
    subprocess-spawn failure (FileNotFoundError, PermissionError, ...) ends
    up wrapped in a BaseExceptionGroup whose default ``str()`` just says
    ``"unhandled errors in a TaskGroup (1 sub-exception)"``. That message is
    useless to a user staring at the install modal, so we unwrap to the
    first leaf exception and prefix the chain so the user can still see we
    came through a TaskGroup.
    """
    chain: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, BaseExceptionGroup) and current.exceptions:
            chain.append(type(current).__name__)
            current = current.exceptions[0]
            continue
        chain.append(f"{type(current).__name__}: {current}")
        break
    return " <- ".join(chain) if chain else f"{type(exc).__name__}: {exc}"


async def _list_tools_via_adapter(spec: dict) -> list[dict]:
    """Boot exactly one server, list its tools, then close.

    We use ``MultiServerMCPClient`` (the same client the agent uses) so the
    validation path doesn't drift from production.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({"candidate": spec})
    raw_tools = await client.get_tools()
    out: list[dict] = []
    for t in raw_tools:
        out.append({
            "name": t.name,
            "description": (t.description or "").strip(),
        })
    # MultiServerMCPClient does its own subprocess lifecycle. Exiting this
    # coroutine drops the only reference; the subprocess gets reaped by GC
    # plus its own stdio-pipe-closed handler.
    return out


async def install_server(
    *,
    paths: ForgePaths,
    name: str,
    contents: str,
    description: str = "",
) -> dict:
    """Validate, then persist (descriptor + manifest entry). Returns the new
    entry plus ``pending_restart: True`` so the UI can banner.

    ``contents`` is a JSON descriptor (see ``validate_server``).

    Raises ``ValueError`` for refused requests (bad name, name collision with
    built-in or another user server, validation failure).
    """
    if not is_valid_name(name):
        raise ValueError(
            "invalid server name; must match [a-zA-Z][a-zA-Z0-9_-]{0,40}",
        )

    # Refuse to shadow a built-in. (The tool loader also refuses at boot, but
    # failing early gives a much better UX.)
    if name in SERVERS:
        raise ValueError(f"name {name!r} collides with a built-in MCP server")

    mf = load_manifest(paths.forge_dir)
    if name in mf.by_name():
        raise ValueError(f"a user server named {name!r} is already installed")

    val = await validate_server(paths=paths, contents=contents)
    if not val["ok"]:
        raise ValueError(f"validation failed: {val.get('error')}")

    # Persist the descriptor next to other user servers.
    user_dir = user_servers_dir(paths.forge_dir)
    user_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{name}.json"
    target = user_dir / filename
    target.write_text(contents, encoding="utf-8")

    entry = UserMcpServer(
        name=name,
        kind="json",
        path=filename,
        description=description,
        extra={"tools": val["tools"]},
    )
    mf.servers.append(entry)
    save_manifest(paths.forge_dir, mf)

    return {
        "ok": True,
        "entry": asdict(entry),
        "tools": val["tools"],
        "pending_restart": True,
    }


def uninstall_server(*, paths: ForgePaths, name: str) -> dict:
    """Remove a user-installed MCP server.

    Deletes the JSON descriptor from ``.forge/mcp_servers/<name>.json``
    (if present), removes the matching manifest entry, and saves the
    pruned manifest atomically. The running engine still has the old
    tools loaded — callers should follow with ``engine.shutdown()`` and
    reboot (or hit ``POST /api/mcp/reload``) to actually unload them.

    Raises ``ValueError`` when ``name`` matches a built-in (which can't
    be uninstalled — those ship with Forge) or when no user server with
    that name is installed.
    """
    if name in SERVERS:
        raise ValueError(
            f"{name!r} is a built-in MCP server and cannot be uninstalled",
        )
    mf = load_manifest(paths.forge_dir)
    by_name = mf.by_name()
    entry = by_name.get(name)
    if entry is None:
        raise ValueError(f"no user MCP server named {name!r} is installed")

    # Delete the descriptor file. Don't error if it's already gone — we
    # still want the manifest entry removed.
    descriptor = user_servers_dir(paths.forge_dir) / entry.path
    try:
        descriptor.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ValueError(f"failed to delete descriptor: {exc}") from exc

    mf.servers = [s for s in mf.servers if s.name != name]
    save_manifest(paths.forge_dir, mf)
    return {"ok": True, "name": name, "pending_restart": True}


def list_servers_for_api(paths: ForgePaths) -> dict:
    """Built-ins + user entries enriched with whatever the manifest knows
    about tools and a ``pending_restart`` flag (set when there are *any*
    user servers — we can't know what's live in the running engine without
    rebooting it)."""
    rows = list_known_servers(paths)
    mf = load_manifest(paths.forge_dir)
    by_name = mf.by_name()
    # Decorate user rows with the cached tool list from the manifest.
    for row in rows:
        if row["kind"].startswith("user_"):
            entry = by_name.get(row["name"])
            if entry is not None:
                tools = entry.extra.get("tools") or []
                if isinstance(tools, list):
                    row["tools"] = [t.get("name") for t in tools if isinstance(t, dict) and t.get("name")]
    return {"servers": rows, "pending_restart": bool(mf.servers)}


__all__ = [
    # manifest
    "MANIFEST_NAME",
    "USER_SERVERS_DIR",
    "Manifest",
    "UserMcpServer",
    "McpKind",
    "manifest_path",
    "user_servers_dir",
    "load_manifest",
    "save_manifest",
    # loader
    "SERVERS",
    "LoadedTools",
    "load_mcp_tools",
    "list_known_servers",
    # validate + install
    "is_valid_name",
    "validate_server",
    "install_server",
    "uninstall_server",
    "list_servers_for_api",
]
