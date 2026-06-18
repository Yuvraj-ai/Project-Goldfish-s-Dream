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
