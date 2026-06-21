from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_watch_run_polls_repo():
    from open_deep_research.api.streaming import watch_run
    repo = AsyncMock()
    repo.progress_after = AsyncMock()
    repo.progress_after.side_effect = [
        [MagicMock(seq=0, event_type="phase_start", message="go")],
        [MagicMock(seq=1, event_type="run_completed", message="done")],
    ]
    repo.get_run = AsyncMock()
    repo.get_run.return_value = MagicMock(status="completed")

    events = []
    async for event in watch_run(repo, "run_id", poll_interval=0.01):
        events.append(event)
        if event.event_type == "run_completed":
            break

    assert len(events) >= 1
    assert events[-1].event_type == "run_completed"


def test_progress_event_model():
    from open_deep_research.api.models import ProgressEvent
    ev = ProgressEvent(seq=0, event_type="test", message="hello")
    assert ev.seq == 0
    assert ev.event_type == "test"
    assert ev.message == "hello"
    assert ev.timestamp is not None


@pytest.mark.asyncio
async def test_watch_run_polls_on_completed_run(tmp_path):
    import os
    from open_deep_research.api.models import ProgressEvent
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    from open_deep_research.api.streaming import watch_run

    db_path = os.path.join(str(tmp_path), "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("test", {})
        await repo.append_progress(run_id, ProgressEvent(
            seq=0, event_type="phase_start", phase="test",
            message="Starting", timestamp="2026-01-01T00:00:00",
        ))
        await repo.update_run_status(run_id, "completed")

        events = []
        poll = 0.05
        async for event in watch_run(repo, run_id, poll_interval=poll):
            events.append(event)

        assert len(events) >= 1
        assert events[0].event_type == "phase_start"
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_watch_run_respects_last_event_id(tmp_path):
    import os
    from open_deep_research.api.models import ProgressEvent
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    from open_deep_research.api.streaming import watch_run

    db_path = os.path.join(str(tmp_path), "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("test", {})
        await repo.append_progress(run_id, ProgressEvent(
            seq=0, event_type="phase_start", phase="test",
            message="First", timestamp="2026-01-01T00:00:00",
        ))
        await repo.append_progress(run_id, ProgressEvent(
            seq=1, event_type="phase_complete", phase="test",
            message="Second", timestamp="2026-01-01T00:00:01",
        ))
        await repo.update_run_status(run_id, "completed")

        events_after = await repo.progress_after(run_id, 0)
        assert len(events_after) == 1
        assert events_after[0].seq == 1
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_watch_run_nonexistent(tmp_path):
    import os
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    from open_deep_research.api.streaming import watch_run

    db_path = os.path.join(str(tmp_path), "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    events = []
    async for event in watch_run(repo, "nonexistent", poll_interval=0.05):
        events.append(event)
    assert events == []
    await repo.close()


@pytest.mark.asyncio
async def test_watch_run_multiple_events(tmp_path):
    import os
    from open_deep_research.api.models import ProgressEvent
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    from open_deep_research.api.streaming import watch_run

    db_path = os.path.join(str(tmp_path), "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("test", {})
        for i in range(5):
            await repo.append_progress(run_id, ProgressEvent(
                seq=i, event_type="phase_start", phase="test",
                message=f"Event {i}", timestamp=f"2026-01-01T00:00:0{i}",
            ))
        await repo.update_run_status(run_id, "completed")

        events = []
        async for event in watch_run(repo, run_id, poll_interval=0.05):
            events.append(event)

        assert len(events) == 5
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_watch_run_stops_on_completed(tmp_path):
    import os
    from open_deep_research.api.models import ProgressEvent
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    from open_deep_research.api.streaming import watch_run

    db_path = os.path.join(str(tmp_path), "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        run_id = await repo.create_run("test", {})
        await repo.update_run_status(run_id, "completed")

        events = []
        async for event in watch_run(repo, run_id, poll_interval=0.05):
            events.append(event)

        assert events == []
    finally:
        await repo.close()
