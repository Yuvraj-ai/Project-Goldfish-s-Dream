from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True)
async def init_repo():
    import os
    import tempfile

    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    repo_global = await SqliteResearchRepository.create(db_path)
    import open_deep_research.api.deps as deps
    deps._repo = repo_global
    deps._config = None
    yield
    await repo_global.close()


@pytest.mark.asyncio
async def test_health_endpoint():
    from open_deep_research.api.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_start_research():
    from open_deep_research.api.main import app
    from open_deep_research.api.runner import ResearchRunner

    transport = ASGITransport(app=app)
    with patch.object(ResearchRunner, "start", return_value="mock_run_001"):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/research", json={
                "query": "test query",
                "config": {"model": "gemini"},
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["run_id"] == "mock_run_001"
            assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_get_run_not_found():
    from open_deep_research.api.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/research/nonexistent")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_run():
    from open_deep_research.api.deps import get_repo
    from open_deep_research.api.main import app

    repo = await get_repo()
    run_id = await repo.create_run("test query", {"model": "gemini"})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/research/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test query"


@pytest.mark.asyncio
async def test_cancel_research():
    from open_deep_research.api.deps import get_repo
    from open_deep_research.api.main import app
    from open_deep_research.api.runner import ResearchRunner

    repo = await get_repo()
    run_id = await repo.create_run("test", {})

    transport = ASGITransport(app=app)
    with patch.object(ResearchRunner, "cancel", return_value=True):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(f"/research/{run_id}/cancel")
            assert resp.status_code == 200


@pytest.mark.asyncio
async def test_memory_endpoints():
    from open_deep_research.api.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/memory")
        assert resp.status_code == 200

        resp = await client.get("/memory/abc123")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_webhook_crud():
    from open_deep_research.api.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/webhooks")
        assert resp.status_code == 200
        assert resp.json() == []

        resp = await client.post("/webhooks", json={
            "url": "https://example.com/hook",
            "events": ["research.completed"],
        })
        assert resp.status_code == 200
        wh_id = resp.json()["id"]

        resp = await client.get("/webhooks")
        assert len(resp.json()) == 1

        resp = await client.delete(f"/webhooks/{wh_id}")
        assert resp.status_code == 200

        resp = await client.get("/webhooks")
        assert resp.json() == []