from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from open_deep_research.api.deps import get_repo, require_scope
from open_deep_research.api.memory import ResearchMemory
from open_deep_research.api.repository import ResearchRepository

router = APIRouter(prefix="/memory", tags=["memory"])


class FeedbackRequest(BaseModel):
    topic: str
    rating: int
    comment: str | None = None


@router.get("")
async def list_topics(
    repo: ResearchRepository = Depends(get_repo),
    _: None = Depends(require_scope("research:read")),
) -> dict:
    memory = ResearchMemory(repo)
    keys = await memory.list_research_topics()
    return {"topics": keys}


@router.get("/{topic_hash}")
async def load_research(
    topic_hash: str,
    repo: ResearchRepository = Depends(get_repo),
    _: None = Depends(require_scope("research:read")),
) -> dict:
    memory = ResearchMemory(repo)
    data = await memory.load_research(topic_hash)
    return {"topic_hash": topic_hash, "data": data}


@router.post("/feedback")
async def save_feedback(
    body: FeedbackRequest,
    repo: ResearchRepository = Depends(get_repo),
) -> dict:
    memory = ResearchMemory(repo)
    await memory.save_preferences(f"feedback:{body.topic}", body.model_dump())
    return {"status": "ok"}
