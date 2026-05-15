"""Forge CLI entry.

Two UI surfaces share the same headless backend:

* **Electron app** (rich UI — primary). Launch with
  ``cd apps/forge/electron && npm run start``. It expects ``forge serve``
  to be running (and will spawn one itself if not).
* **Slim TUI** (minimal — companion). The default sub-command. Just chat,
  input, and a one-line status strip. Useful over SSH or when launching
  Electron is impractical. Permission asks and plan approvals are
  modals; audit / agents / memory / settings UIs live exclusively in
  the Electron app.

Sub-commands:

    forge                # slim chat TUI (default)
    forge init           # create .forge/ and write default config.toml
    forge config         # print resolved config
    forge serve          # FastAPI + WS headless backend (for Electron / TUI)
    forge index          # build/refresh the codebase RAG index
    forge eval           # run the bundled gold set, print pass/fail + success-rate
    forge audit          # tail the JSONL audit log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .config import ensure_config, load_config, write_config
from .paths import ForgePaths, env_overrides_root


def _paths(args: argparse.Namespace) -> ForgePaths:
    start = Path(args.repo).resolve() if getattr(args, "repo", None) else env_overrides_root()
    return ForgePaths.for_repo(start)


def cmd_init(args: argparse.Namespace) -> int:
    paths = _paths(args)
    paths.ensure()
    cfg, wrote = ensure_config(paths)
    print(f"forge: {'created' if wrote else 'reusing'} {paths.config_toml}")
    print(f"forge: repo root = {paths.repo_root}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    paths = _paths(args)
    cfg = load_config(paths)
    if args.format == "json":
        print(json.dumps(cfg.model_dump(mode="json"), indent=2))
    else:  # toml-ish: just dump pydantic repr
        for section, value in cfg.model_dump(mode="python").items():
            print(f"[{section}]")
            for k, v in value.items():
                print(f"  {k} = {v!r}")
            print()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from .server import run_server  # lazy to avoid uvicorn import unless needed
    except ImportError as exc:
        print(f"forge serve requires fastapi+uvicorn: {exc}", file=sys.stderr)
        return 2
    paths = _paths(args)
    cfg = load_config(paths)
    host = args.host or "127.0.0.1"
    if host not in {"127.0.0.1", "localhost", "::1"}:
        # P9 risk callout: refuse non-loopback binds.
        print(
            f"forge serve refuses to bind {host!r}; loopback only "
            "(this is by design — no auth on the API)",
            file=sys.stderr,
        )
        return 2
    port = args.port or cfg.ui.server_port
    run_server(host=host, port=port, paths=paths)
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    try:
        from .repo_rag import build_index
    except ImportError as exc:
        print(f"forge index requires repo_rag deps: {exc}", file=sys.stderr)
        return 2
    paths = _paths(args)
    cfg = load_config(paths)
    n = build_index(paths=paths, cfg=cfg.repo_rag, force=args.force)
    print(f"forge: indexed {n} chunks into {paths.rag_index_dir}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    try:
        from .eval.runner import run_eval_cli
    except ImportError as exc:
        print(f"forge eval requires eval deps: {exc}", file=sys.stderr)
        return 2
    paths = _paths(args)
    cfg = load_config(paths)
    return run_eval_cli(paths=paths, cfg=cfg, limit=args.limit)


def cmd_audit(args: argparse.Namespace) -> int:
    paths = _paths(args)
    if not paths.audit_jsonl.is_file():
        print(f"forge: no audit log yet at {paths.audit_jsonl}")
        return 0
    with paths.audit_jsonl.open() as fh:
        lines = fh.readlines()
    for line in lines[-args.tail:]:
        print(line.rstrip())
    return 0


def cmd_tui(args: argparse.Namespace) -> int:
    try:
        from .tui.app import run_tui
    except ImportError as exc:
        print(f"forge TUI requires textual: {exc}", file=sys.stderr)
        return 2
    paths = _paths(args)
    cfg, _ = ensure_config(paths)
    run_tui(paths=paths, cfg=cfg)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forge", description=__doc__.split("\n", 1)[0])
    p.add_argument("--version", action="version", version=f"forge {__version__}")
    p.add_argument("--repo", help="override repo root detection")
    sub = p.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init", help="create .forge/ and default config.toml")
    p_init.set_defaults(func=cmd_init)

    p_cfg = sub.add_parser("config", help="print resolved config")
    p_cfg.add_argument("--format", choices=["toml", "json"], default="toml")
    p_cfg.set_defaults(func=cmd_config)

    p_serve = sub.add_parser("serve", help="headless FastAPI+WS server")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.set_defaults(func=cmd_serve)

    p_idx = sub.add_parser("index", help="build/refresh codebase RAG index")
    p_idx.add_argument("--force", action="store_true", help="rebuild from scratch")
    p_idx.set_defaults(func=cmd_index)

    p_eval = sub.add_parser("eval", help="run the bundled gold set")
    p_eval.add_argument("--limit", type=int, default=None, help="run only N tasks")
    p_eval.set_defaults(func=cmd_eval)

    p_audit = sub.add_parser("audit", help="tail the permissions audit log")
    p_audit.add_argument("--tail", type=int, default=50)
    p_audit.set_defaults(func=cmd_audit)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None) or cmd_tui
    try:
        return int(func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
