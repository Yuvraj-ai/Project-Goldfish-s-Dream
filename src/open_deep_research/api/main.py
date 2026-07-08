from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from open_deep_research.api.config import ApiConfig
from open_deep_research.api.deps import get_config, init_repo, set_config
from open_deep_research.api.exceptions import APIError
from open_deep_research.api.metrics import metrics
from open_deep_research.api.routes import admin, memory, research
from open_deep_research.logging_config import request_id_var, setup_logging

# Configure package-wide logging once, at API entry. Deployments can select
# structured output with LOG_FORMAT=json (keeps the previous JSON + request_id
# behavior) and verbosity with LOG_LEVEL.
setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = ApiConfig()
    set_config(cfg)
    await init_repo(cfg.api_db_path)
    logger.info("API initialized (db=%s, host=%s, port=%d)", cfg.api_db_path, cfg.api_host, cfg.api_port)
    yield


app = FastAPI(
    title="Open Deep Research API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(research.router)
app.include_router(memory.router)
app.include_router(admin.router)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1_000_000:
        return JSONResponse(
            status_code=413, content={"error": "Request body too large"}
        )
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()
        if len(body) > 1_000_000:
            return JSONResponse(
                status_code=413, content={"error": "Request body too large"}
            )
    return await call_next(request)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = str(uuid.uuid4())
    request_id_var.set(rid)
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    metrics.inc("research_requests_total", {"endpoint": request.url.path, "status": str(response.status_code // 100)})
    metrics.observe("research_latency_seconds", duration, {"endpoint": request.url.path})
    if response.status_code >= 400:
        metrics.inc("research_errors_total", {"endpoint": request.url.path, "error_type": str(response.status_code)})
    return response


@app.get("/metrics")
async def metrics_endpoint():
    return PlainTextResponse(metrics.render(), media_type="text/plain; charset=utf-8")


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": exc.code},
    )


def serve() -> None:
    import uvicorn
    cfg = get_config()
    if not cfg.enable_rest_api:
        logger.warning("API server disabled (enable_rest_api=False)")
        return
    uvicorn.run(
        "open_deep_research.api.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=False,
    )


if __name__ == "__main__":
    serve()
