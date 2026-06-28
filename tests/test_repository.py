from __future__ import annotations

import json

import aiosqlite
import pytest

from open_deep_research.api.repository import ResearchRepository


def test_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ResearchRepository()


def test_abc_has_abstract_methods():
    methods = [
        "create_run", "get_run", "update_run_status",
        "append_progress", "progress_after", "next_seq",
        "save_checkpoint", "load_checkpoint",
        "save_report", "get_report",
        "save_memory", "load_memory",
        "list_memory_keys",
        "list_webhooks", "save_webhook", "delete_webhook",
    ]
    for m in methods:
        assert hasattr(ResearchRepository, m)
        assert getattr(ResearchRepository, m).__isabstractmethod__


@pytest.mark.asyncio
async def test_sqlite_create_and_get_run(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("test query", {"model": "gemini"})
        assert run_id is not None
        assert len(run_id) == 12

        run = await repo.get_run(run_id)
        assert run is not None
        assert run.query == "test query"
        assert run.config == {"model": "gemini"}
        assert run.status == "pending"
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_progress_events(tmp_path):
    from open_deep_research.api.models import ProgressEvent
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("q", {})
        seq = await repo.next_seq(run_id)
        assert seq == 0
        await repo.append_progress(run_id, ProgressEvent(
            seq=0, event_type="phase_start", phase="research",
            message="Starting", timestamp="2026-01-01T00:00:00",
        ))
        seq = await repo.next_seq(run_id)
        assert seq == 1
        events = await repo.progress_after(run_id, -1)
        assert len(events) == 1
        assert events[0].event_type == "phase_start"
        events2 = await repo.progress_after(run_id, 1)
        assert len(events2) == 0
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_checkpoint(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("q", {})
        state = json.dumps({"key": "value"}).encode()
        await repo.save_checkpoint(run_id, state)
        loaded = await repo.load_checkpoint(run_id)
        assert loaded == state
        none_val = await repo.load_checkpoint("nonexistent")
        assert none_val is None
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_report(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("q", {})
        report = {"title": "Test", "sections": []}
        await repo.save_report(run_id, report)
        loaded = await repo.get_report(run_id)
        assert loaded == report
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_memory(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        data = {"topic": "AI", "summary": "test"}
        await repo.save_memory("research", "ai_topic", data)
        loaded = await repo.load_memory("research", "ai_topic")
        assert loaded == data
        none_val = await repo.load_memory("research", "nonexistent")
        assert none_val is None
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_webhooks(tmp_path):
    from open_deep_research.api.models import WebhookConfig
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        wh = WebhookConfig(
            url="https://example.com/hook",
            events=["research.completed"],
            secret="s3cr3t",
        )
        await repo.save_webhook(wh)
        hooks = await repo.list_webhooks()
        assert len(hooks) == 1
        assert hooks[0].url == "https://example.com/hook"
        await repo.delete_webhook(wh.id)
        hooks = await repo.list_webhooks()
        assert len(hooks) == 0
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_idempotency(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id1 = await repo.create_run("q", {}, idempotency_key="key-1")
        run_id2 = await repo.create_run("q", {}, idempotency_key="key-1")
        assert run_id1 == run_id2
        run_id3 = await repo.create_run("q", {}, idempotency_key="key-2")
        assert run_id3 != run_id1
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_get_nonexistent_run(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run = await repo.get_run("nonexistent")
        assert run is None
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_update_status_with_completion_time(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("q", {})
        await repo.update_run_status(run_id, "running")
        run = await repo.get_run(run_id)
        assert run.status == "running"
        assert run.completed_at is None
        await repo.update_run_status(run_id, "completed")
        run = await repo.get_run(run_id)
        assert run.status == "completed"
        assert run.completed_at is not None
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_progress_empty_after_high_seq(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("q", {})
        events = await repo.progress_after(run_id, 999)
        assert events == []
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_multiple_runs(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        id1 = await repo.create_run("q1", {})
        id2 = await repo.create_run("q2", {})
        assert id1 != id2
        run1 = await repo.get_run(id1)
        run2 = await repo.get_run(id2)
        assert run1.query == "q1"
        assert run2.query == "q2"
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_memory_overwrite(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        await repo.save_memory("ns", "key", {"v": 1})
        await repo.save_memory("ns", "key", {"v": 2})
        loaded = await repo.load_memory("ns", "key")
        assert loaded["v"] == 2
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_update_run_status(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("q", {})
        await repo.update_run_status(run_id, "running")
        run = await repo.get_run(run_id)
        assert run is not None
        assert run.status == "running"
        await repo.update_run_status(run_id, "failed", error="oops")
        run = await repo.get_run(run_id)
        assert run.status == "failed"
        assert run.error == "oops"
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_create_run_persists(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("test query", {"key": "val"})
        run = await repo.get_run(run_id)
        assert run is not None
        assert run.query == "test query"
        assert run.config == {"key": "val"}
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_get_nonexistent_returns_none(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run = await repo.get_run("nonexistent")
        assert run is None
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_save_progress_event(tmp_path):
    from open_deep_research.api.models import ProgressEvent
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("test", {})
        event = ProgressEvent(
            seq=0, event_type="node_start", phase="research",
            message="Starting research",
        )
        await repo.append_progress(run_id, event)
        events = await repo.progress_after(run_id, -1)
        assert len(events) >= 1
        assert events[0].event_type == "node_start"
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_progress_events_empty(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("test", {})
        events = await repo.progress_after(run_id, -1)
        assert events == []
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_sqlite_handles_invalid_path():
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    with pytest.raises(Exception):
        await SqliteResearchRepository.create("/nonexistent/dir/db.db")


@pytest.mark.asyncio
async def test_sqlite_update_run_status_no_completion(tmp_path):
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("q", {})
        await repo.update_run_status(run_id, "running")
        run = await repo.get_run(run_id)
        assert run.status == "running"
        assert run.completed_at is None
    finally:
        await repo.close()
