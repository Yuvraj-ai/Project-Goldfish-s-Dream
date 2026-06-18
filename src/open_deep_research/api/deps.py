from __future__ import annotations

from fastapi import Header

from open_deep_research.api.config import ApiConfig
from open_deep_research.api.exceptions import AuthError
from open_deep_research.api.repository import ResearchRepository
from open_deep_research.api.repository_sqlite import SqliteResearchRepository

_repo: ResearchRepository | None = None
_config: ApiConfig | None = None


async def init_repo(db_path: str) -> ResearchRepository:
    global _repo
    _repo = await SqliteResearchRepository.create(db_path)
    return _repo


async def get_repo() -> ResearchRepository:
    if _repo is None:
        raise RuntimeError("Repository not initialized")
    return _repo


def set_config(config: ApiConfig) -> None:
    global _config
    _config = config


def get_config() -> ApiConfig:
    if _config is None:
        return ApiConfig()
    return _config


async def verify_api_key(
    api_key: str | None = Header(None, alias="X-API-Key"),
) -> None:
    cfg = get_config()
    if cfg.api_key and cfg.api_key != api_key:
        raise AuthError("Invalid API key")

    return None
