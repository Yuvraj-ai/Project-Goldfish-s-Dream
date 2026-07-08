from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from open_deep_research.api.models import ProgressEvent
from open_deep_research.api.repository import ResearchRepository

logger = logging.getLogger(__name__)


class ResearchRunner:
    _tasks: dict[str, asyncio.Task] = {}
    _semaphore: asyncio.Semaphore | None = None

    @classmethod
    async def start(
        cls,
        repo: ResearchRepository,
        query: str,
        config: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> str:
        max_concurrent = config.get("max_concurrent_runs", 3)
        if cls._semaphore is None or cls._semaphore._value != max_concurrent:
            cls._semaphore = asyncio.Semaphore(max_concurrent)

        run_id = await repo.create_run(query, config, idempotency_key)
        logger.info(
            "Research run %s created (mode=%s, max_concurrent=%s, query=%.80s)",
            run_id, config.get("mode"), max_concurrent, query,
        )
        task = asyncio.create_task(cls._execute(repo, run_id, query, config))
        cls._tasks[run_id] = task
        return run_id

    @classmethod
    async def _execute(
        cls,
        repo: ResearchRepository,
        run_id: str,
        query: str,
        config: dict[str, Any],
    ) -> None:
        async with cls._semaphore:
            started_at = datetime.now(timezone.utc)
            logger.info(
                "Research run %s started (mode=%s, query=%.80s)",
                run_id, config.get("mode"), query,
            )
            try:
                await repo.update_run_status(run_id, "running")
                seq = await repo.next_seq(run_id)
                await repo.append_progress(run_id, ProgressEvent(
                    seq=seq, event_type="run_started",
                    message=f"Research started: {query}",
                    metadata={"query": query},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))

                from langchain_core.messages import HumanMessage

                from open_deep_research.deep_researcher import deep_researcher

                runtime_config = {
                    "configurable": {
                        "repo": repo,
                        "run_id": run_id,
                        **config,
                    }
                }

                logger.info("Run %s: invoking deep_researcher graph", run_id)
                final_state: dict[str, Any] = {}
                async for step_state in deep_researcher.astream(
                    {"messages": [HumanMessage(content=query)], "document_paths": []},
                    runtime_config,
                    stream_mode="values",
                ):
                    final_state = step_state

                report_content = final_state.get("final_report")
                if report_content:
                    logger.info(
                        "Run %s: graph produced report (%d chars)",
                        run_id, len(report_content),
                    )
                    await repo.save_report(run_id, {
                        "markdown": report_content,
                        "format": "markdown",
                        "run_id": run_id,
                    })
                else:
                    logger.warning("Run %s: graph produced empty/partial result (no final_report)", run_id)

                await repo.update_run_status(run_id, "completed")
                duration = (datetime.now(timezone.utc) - started_at).total_seconds()
                logger.info(
                    "Research run %s finished (status=completed, duration=%.2fs)",
                    run_id, duration,
                )
                seq = await repo.next_seq(run_id)
                await repo.append_progress(run_id, ProgressEvent(
                    seq=seq, event_type="run_completed",
                    message="Research completed",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
            except asyncio.CancelledError:
                logger.warning("Research run %s cancelled", run_id)
                await repo.update_run_status(run_id, "cancelled")
                seq = await repo.next_seq(run_id)
                await repo.append_progress(run_id, ProgressEvent(
                    seq=seq, event_type="run_failed",
                    message="Research cancelled",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
            except Exception as e:
                duration = (datetime.now(timezone.utc) - started_at).total_seconds()
                logger.exception(
                    "Research run %s finished (status=failed, duration=%.2fs)",
                    run_id, duration,
                )
                await repo.update_run_status(run_id, "failed", error=str(e))
                seq = await repo.next_seq(run_id)
                await repo.append_progress(run_id, ProgressEvent(
                    seq=seq, event_type="run_failed",
                    message=f"Research failed: {e}",
                    metadata={"error": str(e)},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
            finally:
                cls._tasks.pop(run_id, None)

    @classmethod
    async def cancel(cls, run_id: str) -> bool:
        task = cls._tasks.get(run_id)
        if task is None:
            logger.warning("Cancel requested for unknown/inactive run %s", run_id)
            return False
        logger.info("Cancelling research run %s", run_id)
        task.cancel()
        return True

    @classmethod
    def list_active(cls) -> list[str]:
        active = list(cls._tasks.keys())
        logger.debug("Listing active runs: %d active", len(active))
        return active

    @classmethod
    def get_status(cls, run_id: str) -> str | None:
        task = cls._tasks.get(run_id)
        if task is None:
            logger.debug("Status requested for unknown/inactive run %s", run_id)
            return None
        if task.done():
            if task.cancelled():
                return "cancelled"
            if task.exception():
                return "failed"
            return "completed"
        return "running"
