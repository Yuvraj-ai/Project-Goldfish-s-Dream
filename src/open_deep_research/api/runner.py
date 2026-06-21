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

                final_state: dict[str, Any] = {}
                async for step_state in deep_researcher.astream(
                    {"messages": [HumanMessage(content=query)], "document_paths": []},
                    runtime_config,
                    stream_mode="values",
                ):
                    final_state = step_state

                report_content = final_state.get("final_report")
                if report_content:
                    await repo.save_report(run_id, {
                        "markdown": report_content,
                        "format": "markdown",
                        "run_id": run_id,
                    })

                await repo.update_run_status(run_id, "completed")
                seq = await repo.next_seq(run_id)
                await repo.append_progress(run_id, ProgressEvent(
                    seq=seq, event_type="run_completed",
                    message="Research completed",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
            except asyncio.CancelledError:
                await repo.update_run_status(run_id, "cancelled")
                seq = await repo.next_seq(run_id)
                await repo.append_progress(run_id, ProgressEvent(
                    seq=seq, event_type="run_failed",
                    message="Research cancelled",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
            except Exception as e:
                logger.exception("Run %s failed", run_id)
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
            return False
        task.cancel()
        return True

    @classmethod
    def list_active(cls) -> list[str]:
        return list(cls._tasks.keys())

    @classmethod
    def get_status(cls, run_id: str) -> str | None:
        task = cls._tasks.get(run_id)
        if task is None:
            return None
        if task.done():
            if task.cancelled():
                return "cancelled"
            if task.exception():
                return "failed"
            return "completed"
        return "running"
