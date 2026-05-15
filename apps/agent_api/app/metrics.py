"""Prometheus metrics exposed at ``/metrics``.

Three intentional choices:

* Counters and histograms live on a **dedicated registry**, not the default
  one, so test runs that re-import the module don't fight a process-global
  registry on duplicate registration.
* ``observe_node`` is the single entry point the workflow nodes call; the
  middleware uses ``observe_request`` for HTTP-level timing. Adding a new
  named stage is just another label value.
* The metric names follow the Prometheus convention: ``_total`` for
  counters, ``_seconds`` for time histograms, and units in the metric name.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


REGISTRY = CollectorRegistry()

# --- workflow-level counters / histograms ---------------------------------

workflow_runs_total = Counter(
    "agent_workflow_runs_total",
    "Number of full workflow runs by terminal outcome.",
    labelnames=("outcome",),  # "complete" | "cap" | "error"
    registry=REGISTRY,
)

reflect_rounds_total = Counter(
    "agent_reflect_rounds_total",
    "Total plan -> agent -> reflect rounds across all runs.",
    registry=REGISTRY,
)

workflow_latency_seconds = Histogram(
    "agent_workflow_latency_seconds",
    "End-to-end workflow latency in seconds (plan -> artifact).",
    buckets=(1, 2, 5, 10, 20, 40, 80, 160, 320, 640),
    registry=REGISTRY,
)

# --- per-node timing -------------------------------------------------------

node_latency_seconds = Histogram(
    "agent_node_latency_seconds",
    "Per-node execution latency (plan / agent / reflect / artifact).",
    labelnames=("node", "status"),  # status: "ok" | "error"
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 40, 80, 160),
    registry=REGISTRY,
)

# --- HTTP-level metrics ---------------------------------------------------

http_requests_total = Counter(
    "agent_http_requests_total",
    "Total HTTP requests served, by route and status class.",
    labelnames=("method", "path", "status"),
    registry=REGISTRY,
)


def render() -> tuple[bytes, str]:
    """Return ``(body, content_type)`` for the /metrics endpoint."""
    from prometheus_client import CONTENT_TYPE_LATEST
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


@contextmanager
def observe_node(name: str) -> Iterator[None]:
    """Time one workflow node. Status defaults to "ok"; if the wrapped
    block raises, we record "error" and re-raise."""
    t0 = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        node_latency_seconds.labels(node=name, status=status).observe(
            time.perf_counter() - t0
        )
