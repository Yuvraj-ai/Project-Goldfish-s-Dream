from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from open_deep_research.api.models import ProgressEvent
from open_deep_research.api.repository import ResearchRepository

logger = logging.getLogger(__name__)


async def watch_run(
    repo: ResearchRepository,
    run_id: str,
    poll_interval: float = 0.5,
) -> AsyncGenerator[ProgressEvent, None]:
    last_seq = -1
    while True:
        events = await repo.progress_after(run_id, last_seq)
        for event in events:
            last_seq = event.seq
            yield event

        run = await repo.get_run(run_id)
        if run and run.status in ("completed", "failed", "cancelled"):
            remaining = await repo.progress_after(run_id, last_seq)
            for event in remaining:
                yield event
            break

        await asyncio.sleep(poll_interval)
