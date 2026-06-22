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


@pytest.mark.asyncio
async def test_auth_required():
    from open_deep_research.api.main import app
    from open_deep_research.api.deps import set_config
    from open_deep_research.api.config import ApiConfig

    cfg = ApiConfig(api_key="test-key-123")
    set_config(cfg)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200

        resp = await client.get("/research/abc")
        assert resp.status_code == 401

        resp = await client.get("/research/abc", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

        resp = await client.get("/research/abc", headers={"X-API-Key": "test-key-123"})
        assert resp.status_code == 404

    set_config(ApiConfig())


def test_openapi_schema_is_valid():
    from open_deep_research.api.main import app
    schema = app.openapi()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
    assert len(schema["paths"]) >= 13


@pytest.mark.asyncio
async def test_health_response_shape():
    from open_deep_research.api.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)

@pytest.mark.asyncio
async def test_start_research_no_body():
    from open_deep_research.api.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/research", json={})
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_metrics_endpoint():
    from open_deep_research.api.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_start_research_query_too_long():
    from open_deep_research.api.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/research", json={
            "query": "x" * 2001,
        })
        assert resp.status_code == 400
        assert "too long" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_report_not_found():
    from open_deep_research.api.deps import get_repo
    from open_deep_research.api.main import app
    repo = await get_repo()
    run_id = await repo.create_run("test query", {})
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/research/{run_id}/report")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_export_unsupported_format():
    from open_deep_research.api.deps import get_repo
    from open_deep_research.api.main import app
    repo = await get_repo()
    run_id = await repo.create_run("test query", {})
    await repo.save_report(run_id, {"content": "test report"})
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/research/{run_id}/export/pdf")
        assert resp.status_code == 400
        assert "Unsupported format" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_cancel_nonexistent_run():
    from open_deep_research.api.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/research/nonexistent/cancel")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stream_research():
    from open_deep_research.api.deps import get_repo
    from open_deep_research.api.main import app
    from open_deep_research.api.models import ProgressEvent

    repo = await get_repo()
    run_id = await repo.create_run("stream test", {})
    await repo.update_run_status(run_id, "running")
    event = ProgressEvent(seq=0, event_type="phase", phase="research", message="Starting research")
    await repo.append_progress(run_id, event)
    await repo.update_run_status(run_id, "completed")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/research/{run_id}/stream")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "Starting research" in resp.text


@pytest.mark.asyncio
async def test_stream_research_not_found():
    from open_deep_research.api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/research/nonexistent/stream")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_report_success():
    from open_deep_research.api.deps import get_repo
    from open_deep_research.api.main import app

    repo = await get_repo()
    run_id = await repo.create_run("report test", {})
    report = {"content": "# Test Report", "format": "markdown"}
    await repo.save_report(run_id, report)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/research/{run_id}/report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "# Test Report"


@pytest.mark.asyncio
async def test_get_report_run_not_found():
    from open_deep_research.api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/research/nonexistent/report")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_export_report_success():
    from open_deep_research.api.deps import get_repo
    from open_deep_research.api.main import app

    repo = await get_repo()
    run_id = await repo.create_run("export test", {})
    report = {"content": "# Test Report", "format": "markdown"}
    await repo.save_report(run_id, report)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/research/{run_id}/export/json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert data["format"] == "json"
        assert "data" in data


@pytest.mark.asyncio
async def test_export_report_run_not_found():
    from open_deep_research.api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/research/nonexistent/export/json")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_export_report_no_report():
    from open_deep_research.api.deps import get_repo
    from open_deep_research.api.main import app

    repo = await get_repo()
    run_id = await repo.create_run("export no report", {})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/research/{run_id}/export/json")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_plugins():
    from open_deep_research.api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/plugins")
        assert resp.status_code == 200
        assert resp.json() == {"plugins": []}


@pytest.mark.asyncio
async def test_save_feedback():
    from open_deep_research.api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/memory/feedback", json={
            "topic": "test topic",
            "rating": 5,
            "comment": "great",
        })
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_export_markdown_and_html():
    from open_deep_research.api.deps import get_repo
    from open_deep_research.api.main import app

    repo = await get_repo()
    run_id = await repo.create_run("multi format export", {})
    report = {"content": "# Multi Format", "format": "markdown"}
    await repo.save_report(run_id, report)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for fmt in ("markdown", "html"):
            resp = await client.get(f"/research/{run_id}/export/{fmt}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["format"] == fmt