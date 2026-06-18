from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class ProgressEvent(BaseModel):
    seq: int
    event_type: str
    phase: str | None = None
    message: str
    metadata: dict | None = None
    timestamp: str = Field(default_factory=_utcnow)


class RunRecord(BaseModel):
    id: str = Field(default_factory=_new_id)
    query: str
    config: dict = Field(default_factory=dict)
    idempotency_key: str | None = None
    status: str = "pending"
    error: str | None = None
    created_at: str = Field(default_factory=_utcnow)
    completed_at: str | None = None


class WebhookConfig(BaseModel):
    id: str = Field(default_factory=_new_id)
    url: str
    events: list[str] = Field(default_factory=list)
    secret: str | None = None
    active: bool = True
    timeout_seconds: int = 5
    retry_max: int = 3
