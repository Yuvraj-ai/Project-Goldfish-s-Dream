from __future__ import annotations

from fastapi import Header

from open_deep_research.api.config import ApiConfig
from open_deep_research.api.exceptions import AuthError, ForbiddenError
from open_deep_research.api.repository import ResearchRepository
from open_deep_research.api.repository_sqlite import SqliteResearchRepository

_repo: ResearchRepository | None = None
_config: ApiConfig | None = None


class ApiKeyValidator:
    def __init__(self, config: ApiConfig) -> None:
        self._keys = dict(config.api_keys or {})
        if config.api_key:
            self._keys[config.api_key] = {
                "name": "default",
                "scopes": ["research:read", "research:write"],
            }

    def validate(self, api_key: str | None) -> dict:
        if api_key is None:
            raise AuthError("API key required")
        key_data = self._keys.get(api_key)
        if key_data is None:
            raise AuthError("Invalid API key")
        return key_data  # type: ignore


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


def _get_validator() -> ApiKeyValidator | None:
    cfg = get_config()
    if not cfg.api_key and not cfg.api_keys:
        return None
    return ApiKeyValidator(cfg)


async def verify_api_key(
    api_key: str | None = Header(None, alias="X-API-Key"),
) -> None:
    validator = _get_validator()
    if validator is None:
        return
    validator.validate(api_key)


def require_scope(scope: str):
    async def dependency(
        api_key: str | None = Header(None, alias="X-API-Key"),
    ) -> None:
        validator = _get_validator()
        if validator is None:
            return
        key_data = validator.validate(api_key)
        scopes: list = key_data.get("scopes", [])
        if scope not in scopes and "admin" not in scopes:
            raise ForbiddenError(f"Scope '{scope}' required")
    return dependency
