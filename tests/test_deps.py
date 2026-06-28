"""Tests for deps.py auth validation."""
import pytest
from open_deep_research.api.config import ApiConfig
from open_deep_research.api.deps import (
    ApiKeyValidator,
    get_repo,
    require_scope,
    set_config,
    verify_api_key,
)
from open_deep_research.api.exceptions import AuthError, ForbiddenError


def test_valid_api_key():
    config = ApiConfig(
        api_key="sk-test1",
        api_keys={"sk-test2": {"name": "test2", "scopes": ["admin"]}},
    )
    validator = ApiKeyValidator(config)
    result = validator.validate("sk-test1")
    assert result is not None
    assert result["name"] == "default"
    assert "research:read" in result["scopes"]


def test_valid_api_key_from_keys_dict():
    config = ApiConfig(
        api_keys={"sk-custom": {"name": "custom", "scopes": ["admin"]}}
    )
    validator = ApiKeyValidator(config)
    result = validator.validate("sk-custom")
    assert result is not None
    assert result["name"] == "custom"


def test_invalid_api_key_raises():
    config = ApiConfig(api_key="sk-valid")
    validator = ApiKeyValidator(config)
    with pytest.raises(AuthError) as exc:
        validator.validate("sk-invalid")
    assert exc.value.status_code == 401


def test_none_api_key_raises():
    config = ApiConfig(api_key="sk-test")
    validator = ApiKeyValidator(config)
    with pytest.raises(AuthError) as exc:
        validator.validate(None)
    assert exc.value.status_code == 401


def test_no_api_key_configured():
    config = ApiConfig()
    validator = ApiKeyValidator(config)
    with pytest.raises(AuthError):
        validator.validate("anything")


@pytest.mark.asyncio
async def test_verify_api_key_no_auth_configured():
    set_config(ApiConfig())
    result = await verify_api_key(api_key=None)
    assert result is None


@pytest.mark.asyncio
async def test_verify_api_key_valid():
    set_config(ApiConfig(api_key="sk-test"))
    result = await verify_api_key(api_key="sk-test")
    assert result is None


@pytest.mark.asyncio
async def test_verify_api_key_invalid():
    set_config(ApiConfig(api_key="sk-test"))
    with pytest.raises(AuthError):
        await verify_api_key(api_key="wrong")


@pytest.mark.asyncio
async def test_require_scope_valid():
    set_config(ApiConfig(api_key="sk-test"))
    dependency = require_scope("research:read")
    result = await dependency(api_key="sk-test")
    assert result is None


@pytest.mark.asyncio
async def test_require_scope_invalid_key():
    set_config(ApiConfig(api_key="sk-test"))
    dependency = require_scope("research:read")
    with pytest.raises(AuthError):
        await dependency(api_key="wrong-key")


@pytest.mark.asyncio
async def test_require_scope_no_auth():
    set_config(ApiConfig())
    dependency = require_scope("research:read")
    result = await dependency(api_key=None)
    assert result is None


@pytest.mark.asyncio
async def test_get_repo_not_initialized():
    from open_deep_research.api import deps
    deps._repo = None
    with pytest.raises(RuntimeError, match="Repository not initialized"):
        await deps.get_repo()


@pytest.mark.asyncio
async def test_require_scope_insufficient_scope():
    set_config(ApiConfig(api_key="sk-test"))
    dependency = require_scope("admin")
    with pytest.raises(ForbiddenError):
        await dependency(api_key="sk-test")


@pytest.mark.asyncio
async def test_init_repo_creates_sqlite(tmp_path):
    from open_deep_research.api.deps import init_repo
    db_path = str(tmp_path / "init_test.db")
    repo = await init_repo(db_path)
    assert repo is not None
    await repo.close()
    import os
    assert os.path.exists(db_path)
