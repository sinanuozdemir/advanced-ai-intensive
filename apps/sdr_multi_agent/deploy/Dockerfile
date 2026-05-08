# Dockerfile for the week-2 SDR cloud deployment.
#
# This is a *separate* Dockerfile from `flask_app/Dockerfile` because the
# deployed image needs more than the original local-docker setup did:
#
#   - the `src/` package (shared, memory, mcp_demo) from the parent repo
#   - the `memory_agent.py` adapter
#   - gunicorn for production serving
#   - sentence-transformers + chroma already cached so first request is fast
#
# Build context must be the REPO ROOT, not the apps/sdr_multi_agent dir,
# because we need to COPY both the app and src/. Fly handles this when the
# Dockerfile path is given relative to the toml's location.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY apps/sdr_multi_agent/flask_app/requirements.txt /tmp/req-base.txt
COPY requirements.txt /tmp/req-extra.txt

RUN pip install -r /tmp/req-base.txt \
    && pip install -r /tmp/req-extra.txt \
    && pip install gunicorn gevent

COPY src/ /app/src/
COPY notebooks/week1/ /app/notebooks/week1/
COPY apps/sdr_multi_agent/flask_app/ /app/
COPY apps/sdr_multi_agent/mcp_servers/ /app/mcp_servers/

RUN mkdir -p /data /app/data
VOLUME ["/data"]

EXPOSE 8080

CMD ["gunicorn", "-k", "gevent", "-w", "2", "-b", "0.0.0.0:8080", "app:app"]
