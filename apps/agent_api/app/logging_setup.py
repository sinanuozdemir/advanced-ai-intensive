"""Structured JSON logging.

Two design points:

* Every record carries the current ``request_id`` (set by a FastAPI middleware
  before any workflow code runs). The id propagates into the LangGraph nodes
  via ``contextvars`` without us having to plumb it through state.
* Logs are written to both stderr (so ``docker logs`` shows them) AND to a
  local file at ``settings.log_file_path`` so ``GET /trace`` can replay one
  request's full path. The file format is JSON-per-line.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_request_id: ContextVar[str] = ContextVar("_agent_api_request_id", default="-")


def set_request_id(rid: str) -> None:
    _request_id.set(rid)


def get_request_id() -> str:
    return _request_id.get()


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per line. Hand-rolled so we don't pull
    python-json-logger purely for this single use site."""

    _RESERVED = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": _request_id.get(),
        }
        for k, v in record.__dict__.items():
            if k in self._RESERVED or k.startswith("_"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(log_file_path: Path, level: str = "INFO") -> None:
    """Idempotent root-logger setup. Safe to call from app startup and tests."""

    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = _JsonFormatter()

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    log_file_path = Path(log_file_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Quiet noisy libraries — we still get warnings, just not their
    # per-request INFO chatter.
    for noisy in ("httpx", "httpcore", "uvicorn.access", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str = "agent_api") -> logging.Logger:
    return logging.getLogger(name)
