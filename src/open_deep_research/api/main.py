from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from open_deep_research.api.config import ApiConfig
from open_deep_research.api.deps import get_config, init_repo, set_config
from open_deep_research.api.exceptions import APIError
from open_deep_research.api.routes import admin, memory, research

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
