"""FastAPI surface for the agentic-workflow API.

Routes (full list also documented in `apps/agent_api/README.md`):

* GET  /healthz                      liveness
* GET  /readyz                       readiness (DB + LLM key resolvable)
* POST /research                     synchronous workflow run
* GET  /artifacts                    paginated list
* GET  /artifacts/{id}               one artifact with provenance
* GET  /metrics                      Prometheus exposition
* GET  /trace?request_id=...         tail the JSON log filtered to a request
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from . import deps as deps_mod
from . import metrics as metrics_mod
from .logging_setup import configure_logging, get_logger, set_request_id
from .schemas import (
    Artifact,
    ArtifactListResponse,
    ResearchRequest,
    ResearchResponse,
)
from .settings import Settings, get_settings
from .store import ArtifactStore
from .workflow import build_workflow


load_dotenv()


log = get_logger("agent_api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the store, configure logging, and (if the key is set) compile the
    workflow graph once. ``/readyz`` reports the graph as not-ready when the
    key is missing — we don't fail boot, so the lecturer can show readiness
    flipping live after they paste the key."""

    settings = get_settings()
    configure_logging(settings.log_file_path, level=settings.log_level)
    log.info("startup", extra={"event": "lifespan.boot", "settings": {
        "plan_model": settings.plan_model,
        "research_model": settings.research_model,
        "reflect_model": settings.reflect_model,
        "artifact_model": settings.artifact_model,
        "max_iterations": settings.max_iterations,
        "artifacts_db_path": str(settings.artifacts_db_path),
    }})

    store = ArtifactStore(settings.artifacts_db_path)

    graph = None
    graph_error: str | None = None
    ok, why = deps_mod.llm_smoke_check(settings)
    if ok:
        try:
            graph = build_workflow(
                plan_llm=deps_mod.build_plan_llm(settings),
                research_agent=deps_mod.build_research_agent(settings),
                reflect_llm=deps_mod.build_reflect_llm(settings),
                artifact_llm=deps_mod.build_artifact_llm(settings),
                store=store,
            )
        except Exception as exc:  # noqa: BLE001
            graph_error = f"{type(exc).__name__}: {exc}"
            log.warning("startup.graph_build_failed", extra={"error": graph_error})
    else:
        graph_error = why
        log.warning("startup.llm_unavailable", extra={"error": why})

    app.state.settings = settings
    app.state.store = store
    app.state.graph = graph
    app.state.graph_error = graph_error

    try:
        yield
    finally:
        store.close()
        log.info("shutdown", extra={"event": "lifespan.shutdown"})


app = FastAPI(
    title="agent_api",
    summary="Agentic workflow (shape -> research -> judge -> save) as an API.",
    version="0.1.0",
    lifespan=lifespan,
)

# Permissive CORS so the notebook (which runs in Jupyter on a different port)
# can hit the service directly without proxying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: request ID + HTTP-level metrics
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    set_request_id(rid)
    request.state.request_id = rid
    t0 = time.perf_counter()
    status = "5xx"
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["x-request-id"] = rid
        status = f"{status_code // 100}xx"
        return response
    finally:
        path = _path_label(request)
        metrics_mod.http_requests_total.labels(
            method=request.method, path=path, status=status
        ).inc()
        log.info(
            "http",
            extra={
                "method": request.method,
                "path": path,
                "status_class": status,
                "latency_s": round(time.perf_counter() - t0, 4),
            },
        )


def _path_label(request: Request) -> str:
    """Use the route template (e.g. ``/artifacts/{id}``) instead of the raw
    path so the cardinality of the ``path`` label stays bounded."""
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return request.url.path


# ---------------------------------------------------------------------------
# Health + readiness
# ---------------------------------------------------------------------------


@app.get("/healthz", tags=["health"])
async def healthz() -> dict[str, Any]:
    """Liveness: 200 if the process is up. Never touches the DB or LLM."""
    return {"status": "ok"}


@app.get("/readyz", tags=["health"])
async def readyz(request: Request):
    """Readiness: 200 only if the DB is writable and the LLM key resolves
    a model. Production orchestrators should gate traffic on this, not
    ``/healthz``."""
    settings: Settings = request.app.state.settings
    store: ArtifactStore = request.app.state.store

    checks: dict[str, dict[str, Any]] = {}
    overall_ok = True

    db_ok = store.is_writable()
    checks["sqlite"] = {"ok": db_ok, "path": str(settings.artifacts_db_path)}
    overall_ok = overall_ok and db_ok

    llm_ok, llm_msg = deps_mod.llm_smoke_check(settings)
    checks["llm"] = {"ok": llm_ok, "detail": llm_msg, "model": settings.research_model}
    overall_ok = overall_ok and llm_ok

    search_ok, search_msg = deps_mod.search_tools_check(settings)
    checks["search_tools"] = {
        "ok": search_ok,
        "detail": search_msg,
        "serpapi": bool(settings.serpapi_api_key),
        "firecrawl": bool(settings.firecrawl_api_key),
    }
    overall_ok = overall_ok and search_ok

    graph_ok = request.app.state.graph is not None
    checks["workflow"] = {
        "ok": graph_ok,
        "detail": (
            "compiled" if graph_ok else
            request.app.state.graph_error or "not compiled"
        ),
    }
    overall_ok = overall_ok and graph_ok

    body = {"status": "ok" if overall_ok else "not_ready", "checks": checks}
    return JSONResponse(body, status_code=200 if overall_ok else 503)


# ---------------------------------------------------------------------------
# Workflow run
# ---------------------------------------------------------------------------


@app.post("/research", tags=["workflow"], response_model=ResearchResponse)
async def run_research(req: ResearchRequest, request: Request) -> ResearchResponse:
    settings: Settings = request.app.state.settings
    graph = request.app.state.graph
    store: ArtifactStore = request.app.state.store
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "workflow not available",
                "reason": request.app.state.graph_error,
            },
        )

    rid = request.state.request_id
    max_iters = req.max_iterations if req.max_iterations is not None else settings.max_iterations

    log.info(
        "workflow.start",
        extra={"topic": req.topic, "max_iterations": max_iters},
    )

    t0 = time.perf_counter()
    try:
        # Each round fires four node transitions (plan/agent/reflect/gate),
        # so scale recursion_limit with max_iters with headroom.
        final_state = graph.invoke(
            {
                "topic": req.topic,
                "request_id": rid,
                "max_iterations": int(max_iters),
            },
            config={"recursion_limit": max(25, 6 * max_iters + 6)},
        )
    except Exception as exc:  # noqa: BLE001
        metrics_mod.workflow_runs_total.labels(outcome="error").inc()
        log.exception("workflow.failed", extra={"topic": req.topic})
        raise HTTPException(
            status_code=500,
            detail={"error": f"{type(exc).__name__}: {exc}"},
        ) from exc
    elapsed = time.perf_counter() - t0
    metrics_mod.workflow_latency_seconds.observe(elapsed)

    artifact_id = final_state.get("artifact_id")
    if not artifact_id:
        raise HTTPException(status_code=500, detail={"error": "workflow produced no artifact_id"})
    artifact = store.get(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=500, detail={"error": "artifact not found after save"})

    return ResearchResponse(
        artifact_id=artifact_id,
        request_id=rid,
        outcome=final_state.get("outcome", "error"),
        rounds=int(final_state.get("round", 0)),
        findings_count=len(final_state.get("findings") or []),
        artifact=artifact,
    )


# ---------------------------------------------------------------------------
# Artifact retrieval
# ---------------------------------------------------------------------------


@app.get("/artifacts/{artifact_id}", tags=["artifacts"], response_model=Artifact)
async def get_artifact(artifact_id: str, request: Request) -> Artifact:
    store: ArtifactStore = request.app.state.store
    art = store.get(artifact_id)
    if art is None:
        raise HTTPException(status_code=404, detail={"error": "artifact not found"})
    return art


@app.get("/artifacts", tags=["artifacts"], response_model=ArtifactListResponse)
async def list_artifacts(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> ArtifactListResponse:
    store: ArtifactStore = request.app.state.store
    items, total = store.list(limit=limit, offset=offset)
    return ArtifactListResponse(items=items, total=total, limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


@app.get("/metrics", tags=["observability"])
async def metrics() -> Response:
    body, content_type = metrics_mod.render()
    return Response(content=body, media_type=content_type)


@app.get("/trace", tags=["observability"])
async def trace(
    request: Request,
    request_id: str = Query(..., min_length=4, description="request_id to filter for"),
    limit: int = Query(default=200, ge=1, le=2000),
) -> Response:
    """Tail the JSON log file and return only lines whose ``request_id``
    matches. Returns ``text/plain`` so it pastes well into a notebook cell."""
    settings: Settings = request.app.state.settings
    path = Path(settings.log_file_path)
    if not path.is_file():
        return PlainTextResponse("(log file does not exist yet)", status_code=200)
    matched: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("request_id") == request_id:
                matched.append(obj)
    matched = matched[-limit:]
    if not matched:
        return PlainTextResponse(
            f"(no log lines for request_id={request_id})", status_code=200
        )
    body = "\n".join(json.dumps(o) for o in matched)
    return PlainTextResponse(body)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@app.get("/", tags=["meta"])
async def root() -> dict[str, Any]:
    return {
        "service": "agent_api",
        "version": app.version,
        "docs": "/docs",
        "endpoints": [
            "GET /healthz",
            "GET /readyz",
            "POST /research",
            "GET /artifacts",
            "GET /artifacts/{id}",
            "GET /metrics",
            "GET /trace?request_id=...",
        ],
    }
