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
    logger.info("Stream opened for run %s (poll_interval=%.2fs)", run_id, poll_interval)
    last_seq = -1
    emitted = 0
    try:
        while True:
            events = await repo.progress_after(run_id, last_seq)
            for event in events:
                last_seq = event.seq
                emitted += 1
                logger.debug(
                    "Stream run %s emitting event seq=%d type=%s",
                    run_id, event.seq, event.event_type,
                )
                yield event

            run = await repo.get_run(run_id)
            if run is None:
                logger.warning("Stream run %s: run disappeared, closing stream", run_id)
                break
            if run.status in ("completed", "failed", "cancelled"):
                remaining = await repo.progress_after(run_id, last_seq)
                for event in remaining:
                    emitted += 1
                    logger.debug(
                        "Stream run %s draining event seq=%d type=%s",
                        run_id, event.seq, event.event_type,
                    )
                    yield event
                logger.info(
                    "Stream run %s reached terminal status=%s after %d event(s)",
                    run_id, run.status, emitted,
                )
                break

            await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        logger.warning("Stream run %s cancelled/disconnected after %d event(s)", run_id, emitted)
        raise
    finally:
        logger.info("Stream closed for run %s (%d event(s) emitted)", run_id, emitted)
