# Docker Deployment Guide

Deploy the Open Deep Research API server using Docker.

---

## Quick Start

```bash
# Build the image
docker build -t open-deep-research .

# Run with API key
docker run -p 8000:8000 \
  -e API_KEY=sk-your-key-here \
  -v odr-data:/data \
  open-deep-research
```

The API is available at `http://localhost:8000`. Metrics at `http://localhost:8000/metrics`.

---

## Dockerfile

Place this `Dockerfile` at the repository root:

```dockerfile
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml README.md ./
RUN uv pip install --system --no-cache -e ".[dev]"

FROM python:3.11-slim

RUN addgroup --system --gid 1001 app && \
    adduser --system --uid 1001 --gid 1001 app

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/

USER app

EXPOSE 8000

ENV PYTHONPATH=/app/src
ENV ENABLE_REST_API=true

CMD ["uvicorn", "open_deep_research.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Docker Compose

Place this `docker-compose.yml` at the repository root:

```yaml
version: "3.9"

services:
  api:
    image: open-deep-research
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENABLE_REST_API=true
      - API_KEY=${API_KEY:-}
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - API_DB_PATH=/data/research.db
      - MAX_CONCURRENT_RUNS=3
    volumes:
      - odr-data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/metrics"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped

volumes:
  odr-data:
```

Start with:

```bash
API_KEY=sk-your-key-here docker compose up -d
```

---

## Environment Variables

All variables map to fields in `ApiConfig` (`src/open_deep_research/api/config.py`):

| Variable | Default | Description |
|---|---|---|
| `ENABLE_REST_API` | `false` | Set to `true` to start the API server |
| `API_KEY` | none | API key for request authentication |
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Listen port |
| `API_DB_PATH` | `research.db` | SQLite database path (use `/data/research.db` for persistent volume) |
| `MAX_CONCURRENT_RUNS` | `3` | Max concurrent research graph executions |

Model configuration is read from `Configuration` in `src/open_deep_research/configuration.py`. Pass model settings via environment variables or a `.env` file mounted into the container:

```bash
docker run -p 8000:8000 \
  -e API_KEY=sk-your-key-here \
  -e API_DB_PATH=/data/research.db \
  -v odr-data:/data \
  -v $(pwd)/.env:/app/.env:ro \
  open-deep-research
```

---

## Volumes

Persist the SQLite database across restarts:

| Mount point | Purpose |
|---|---|
| `/data` | SQLite database (`research.db`) |

Without a volume mount, all data is lost when the container stops.

---

## Health Check

The API exposes a Prometheus-format metrics endpoint at `GET /metrics`. Use it as a health check:

```bash
curl http://localhost:8000/metrics
```

The Docker Compose file above configures a health check against this endpoint with 30s intervals and 3 retries.

---

## Logging

All API logs are emitted as JSON-structured lines to stdout via a custom `JSONFormatter` (`src/open_deep_research/api/main.py:74`):

```json
{"timestamp": "2026-06-22 10:00:00,000", "level": "INFO", "module": "main", "message": "API initialized (db=/data/research.db, host=0.0.0.0, port=8000)"}
```

Each request is tagged with a `request_id` in the log entry. Collect logs with `docker logs` or forward to your log aggregator.

---

## Building for Production

```bash
# Build with no-cache for reproducible builds
docker build --no-cache -t open-deep-research:$(git rev-parse --short HEAD) .

# Tag latest
docker tag open-deep-research:$(git rev-parse --short HEAD) open-deep-research:latest
```

---

## Example End-to-End

```bash
# 1. Build
docker build -t open-deep-research .

# 2. Run with API key and persistent volume
docker run -d --name odr \
  -p 8000:8000 \
  -e API_KEY=sk-prod-abc123 \
  -e API_DB_PATH=/data/research.db \
  -e MAX_CONCURRENT_RUNS=5 \
  -v odr-data:/data \
  open-deep-research

# 3. Verify
curl -H "X-API-Key: sk-prod-abc123" http://localhost:8000/metrics

# 4. Check logs
docker logs odr

# 5. Stop
docker stop odr && docker rm odr
```

---

## Notes

- The API server only starts when `ENABLE_REST_API=true` (gated by `ApiConfig.enable_rest_api`).
- Request body size is limited to **1 MB** (enforced by middleware in `main.py:44-57`).
- The container runs as a non-root `app` user (UID 1001).
- Python 3.11+ required — base image uses `python:3.11-slim`.
