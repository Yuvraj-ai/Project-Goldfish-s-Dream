from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_runner_start_creates_run():
    from open_deep_research.api.runner import ResearchRunner
    repo = AsyncMock()
    repo.create_run = AsyncMock(return_value="run_abc123")
    repo.next_seq = AsyncMock(return_value=0)

    with patch.object(ResearchRunner, "_execute", return_value=None):
        run_id = await ResearchRunner.start(repo, "test query", {"model": "gemini"})
    assert run_id == "run_abc123"
    repo.create_run.assert_awaited_once_with("test query", {"model": "gemini"}, None)
    assert run_id in ResearchRunner._tasks


@pytest.mark.asyncio
async def test_runner_cancel():
    from open_deep_research.api.runner import ResearchRunner
    repo = AsyncMock()
    repo.create_run = AsyncMock(return_value="run_abc123")
    repo.next_seq = AsyncMock(return_value=0)

    with patch.object(ResearchRunner, "_execute", return_value=None):
        run_id = await ResearchRunner.start(repo, "q", {})

    cancelled = await ResearchRunner.cancel(run_id)
    assert cancelled is True

    cancelled = await ResearchRunner.cancel("nonexistent")
    assert cancelled is False


@pytest.mark.asyncio
async def test_runner_get_status():
    from open_deep_research.api.runner import ResearchRunner
    repo = AsyncMock()
    repo.create_run = AsyncMock(return_value="run_abc123")
    repo.next_seq = AsyncMock(return_value=0)

    with patch.object(ResearchRunner, "_execute", return_value=None):
        run_id = await ResearchRunner.start(repo, "q", {})
    status = ResearchRunner.get_status(run_id)
    assert status is not None

    assert ResearchRunner.get_status("nonexistent") is None


@pytest.mark.asyncio
async def test_runner_get_status_completed():
    import asyncio
    from open_deep_research.api.runner import ResearchRunner
    repo = AsyncMock()
    repo.create_run = AsyncMock(return_value="run_complete")
    repo.next_seq = AsyncMock(return_value=0)

    async def quick_task():
        return 42

    task = asyncio.create_task(quick_task())
    await asyncio.sleep(0.01)
    ResearchRunner._tasks["run_complete"] = task
    status = ResearchRunner.get_status("run_complete")
    assert status == "completed"


@pytest.mark.asyncio
async def test_runner_get_status_failed():
    import asyncio
    from open_deep_research.api.runner import ResearchRunner

    async def fail_task():
        raise ValueError("boom")

    task = asyncio.create_task(fail_task())
    await asyncio.sleep(0.01)
    ResearchRunner._tasks["run_fail"] = task
    status = ResearchRunner.get_status("run_fail")
    assert status == "failed"


@pytest.mark.asyncio
async def test_runner_get_status_cancelled():
    import asyncio
    from open_deep_research.api.runner import ResearchRunner

    async def slow_task():
        await asyncio.sleep(10)

    task = asyncio.create_task(slow_task())
    task.cancel()
    await asyncio.sleep(0.01)
    ResearchRunner._tasks["run_cancel"] = task
    status = ResearchRunner.get_status("run_cancel")
    assert status == "cancelled"


@pytest.mark.asyncio
async def test_runner_semaphore_limits(tmp_path):
    from open_deep_research.api.runner import ResearchRunner
    from open_deep_research.api.repository_sqlite import SqliteResearchRepository

    db_path = str(tmp_path / "test.db")
    repo = await SqliteResearchRepository.create(db_path)
    try:
        import asyncio
        import open_deep_research.api.runner as runner_mod
        semaphore = asyncio.Semaphore(2)
        runner_mod.ResearchRunner._semaphore = semaphore
        try:
            from unittest.mock import patch
            with patch.object(ResearchRunner, "_execute", return_value=None):
                run_id = await ResearchRunner.start(repo, "test", {}, None)
                assert run_id is not None
        finally:
            runner_mod.ResearchRunner._semaphore = None
    finally:
        await repo.close()


@pytest.mark.asyncio
async def test_runner_list_active():
    from open_deep_research.api.runner import ResearchRunner
    active = ResearchRunner.list_active()
    assert isinstance(active, list)


@pytest.mark.asyncio
async def test_runner_executes_graph_and_saves_report(tmp_path):
    import asyncio
    import types
    from unittest.mock import patch

    from open_deep_research.api.repository_sqlite import SqliteResearchRepository
    from open_deep_research.api.runner import ResearchRunner

    db_path = str(tmp_path / "runner_e2e.db")
    repo = await SqliteResearchRepository.create(db_path)

    class StubGraph:
        async def astream(self, input_state, config, stream_mode="values"):
            yield {"messages": input_state["messages"], "research_brief": "test", "final_report": "# Hello World"}
            return

    stub_module = types.ModuleType("open_deep_research.deep_researcher")
    stub_module.deep_researcher = StubGraph()

    patcher = patch.dict("sys.modules", {"open_deep_research.deep_researcher": stub_module})
    patcher.start()
    try:
        run_id = await ResearchRunner.start(repo, "test query", {})
        task = ResearchRunner._tasks[run_id]
        await asyncio.wait_for(task, timeout=10)
    finally:
        patcher.stop()

    report = await repo.get_report(run_id)
    assert report is not None
    assert report["format"] == "markdown"
    assert "Hello World" in report["markdown"]

    run = await repo.get_run(run_id)
    assert run is not None
    assert run.status == "completed"

    events = await repo.progress_after(run_id, -1)
    event_types = [e.event_type for e in events]
    assert "run_started" in event_types
    assert "run_completed" in event_types

    await repo.close()
