from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from open_deep_research.api.deps import get_repo, require_scope
from open_deep_research.api.models import WebhookConfig
from open_deep_research.api.repository import ResearchRepository

router = APIRouter(prefix="", tags=["admin"])


class WebhookCreate(BaseModel):
    url: str
    events: list[str] = []
    secret: str | None = None


@router.get("/webhooks")
async def list_webhooks(
    repo: ResearchRepository = Depends(get_repo),
    _: None = Depends(require_scope("admin")),
) -> list[WebhookConfig]:
    return await repo.list_webhooks()


@router.post("/webhooks")
async def create_webhook(
    body: WebhookCreate,
    repo: ResearchRepository = Depends(get_repo),
    _: None = Depends(require_scope("admin")),
) -> WebhookConfig:
    wh = WebhookConfig(url=body.url, events=body.events, secret=body.secret)
    await repo.save_webhook(wh)
    return wh


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: str,
    repo: ResearchRepository = Depends(get_repo),
    _: None = Depends(require_scope("admin")),
) -> dict:
    await repo.delete_webhook(webhook_id)
    return {"status": "deleted"}


@router.get("/plugins")
async def list_plugins() -> dict:
    return {"plugins": []}


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
