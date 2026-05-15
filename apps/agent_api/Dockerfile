FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY mcp_server/ ./mcp_server/

# The shared course modules (`shared`, `multi_agent`, ...) live at the repo root
# under `src/`. They are mounted read-only into /app/src by docker-compose, so
# `from shared import get_llm` keeps working without copying the whole repo into
# the image. Same trick the SDR app uses.
ENV PYTHONPATH=/app:/app/src
ENV PYTHONUNBUFFERED=1
ENV ARTIFACTS_DB_PATH=/app/runtime/artifacts.db
ENV LOG_FILE_PATH=/app/runtime/agent_api.log

EXPOSE 8090

# Single worker so the in-process SQLite store stays consistent for the demo.
# Production deployments would point at a shared Postgres + run --workers >1.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
