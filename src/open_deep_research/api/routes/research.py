from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from open_deep_research.api.deps import get_repo, require_scope, verify_api_key
from open_deep_research.api.exceptions import NotFoundError
from open_deep_research.api.models import RunRecord
from open_deep_research.api.repository import ResearchRepository
from open_deep_research.api.runner import ResearchRunner
from open_deep_research.api.streaming import watch_run

router = APIRouter(prefix="/research", tags=["research"])

MAX_QUERY_LENGTH = 2000


class ResearchRequest(BaseModel):
    query: str
    config: dict[str, Any] = {}
    idempotency_key: str | None = None


@router.post("")
async def start_research(
    body: ResearchRequest,
    repo: ResearchRepository = Depends(get_repo),
    _: None = Depends(verify_api_key),
) -> dict:
    if len(body.query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query too long (max {MAX_QUERY_LENGTH} characters)",
        )
    run_id = await ResearchRunner.start(
        repo, body.query, body.config, body.idempotency_key,
    )
    return {"run_id": run_id, "status": "pending"}


@router.get("/{run_id}")
async def get_run(
    run_id: str,
    repo: ResearchRepository = Depends(get_repo),
    _: None = Depends(require_scope("research:read")),
) -> RunRecord:
    run = await repo.get_run(run_id)
    if run is None:
        raise NotFoundError(f"Run {run_id} not found")
    return run


@router.get("/{run_id}/stream")
async def stream_research(
    run_id: str,
    repo: ResearchRepository = Depends(get_repo),
    _: None = Depends(require_scope("research:read")),
) -> StreamingResponse:
    run = await repo.get_run(run_id)
    if run is None:
        raise NotFoundError(f"Run {run_id} not found")

    async def event_generator():
        async for event in watch_run(repo, run_id):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{run_id}/report")
async def get_report(
    run_id: str,
    repo: ResearchRepository = Depends(get_repo),
    _: None = Depends(require_scope("research:read")),
) -> dict:
    run = await repo.get_run(run_id)
    if run is None:
        raise NotFoundError(f"Run {run_id} not found")
    report = await repo.get_report(run_id)
    if report is None:
        raise NotFoundError(f"Report for run {run_id} not found")
    return report


@router.get("/{run_id}/export/{fmt}")
async def export_report(
    run_id: str,
    fmt: str,
    repo: ResearchRepository = Depends(get_repo),
    _: None = Depends(require_scope("research:read")),
) -> dict:
    run = await repo.get_run(run_id)
    if run is None:
        raise NotFoundError(f"Run {run_id} not found")
    report = await repo.get_report(run_id)
    if report is None:
        raise NotFoundError(f"Report for run {run_id} not found")

    supported = {"json", "markdown", "html"}
    if fmt not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}. Supported: {supported}")

    return {"run_id": run_id, "format": fmt, "data": report}


@router.post("/{run_id}/cancel")
async def cancel_research(
    run_id: str,
    repo: ResearchRepository = Depends(get_repo),
    _: None = Depends(require_scope("research:write")),
) -> dict:
    run = await repo.get_run(run_id)
    if run is None:
        raise NotFoundError(f"Run {run_id} not found")
    cancelled = await ResearchRunner.cancel(run_id)
    return {"run_id": run_id, "cancelled": cancelled}
