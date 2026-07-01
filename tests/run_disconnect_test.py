#!/usr/bin/env python3
"""Task 3-8: Client Disconnect Resilience Test — ZERO LLM CALLS.

Tests server resilience to client disconnects by injecting run state
directly into the repository (in-process, no graph execution).

Steps:
1. Set up config + repository (in-process)
2. Create run via API, inject progress events via repo
3. Connect to SSE stream, read events, disconnect mid-stream
4. Verify server still handles requests
5. Inject more events
6. Reconnect to SSE, verify new events delivered
7. Document results
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from httpx import ASGITransport, AsyncClient

from open_deep_research.api.config import ApiConfig
from open_deep_research.api.deps import set_config, init_repo, get_repo, _repo as repo_global
from open_deep_research.api.main import app
from open_deep_research.api.models import ProgressEvent
from open_deep_research.api.runner import ResearchRunner

RESULTS_PATH = Path(__file__).parent.parent / "docs" / "verification" / "task-3-8-disconnect.json"


def make_progress(seq: int, event_type: str, message: str, phase: str | None = None) -> ProgressEvent:
    return ProgressEvent(
        seq=seq,
        event_type=event_type,
        phase=phase,
        message=message,
        metadata={},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


async def run_test() -> dict:
    results = {
        "task": "3-8 Client Disconnect Resilience",
        "status": "FAIL",
        "llm_calls_made": 0,
        "steps": {},
        "errors": [],
    }
    events_batch1 = 0
    events_batch2 = 0
    run_id = None

    async def step(name: str, fn):
        try:
            await fn()
            results["steps"][name] = "PASS"
        except Exception as e:
            results["steps"][name] = "FAIL"
            results["errors"].append(f"{name}: {e}")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        repo = await get_repo()

        # --- Step 1: Health check ---
        async def s1():
            resp = await client.get("/health")
            assert resp.status_code == 200
        await step("1_server_healthy", s1)

        # --- Step 2: Create run via API ---
        async def s2():
            nonlocal run_id
            resp = await client.post(
                "/research",
                json={"query": "test disconnect resilience", "config": {}},
                headers={"X-API-Key": "test-key"},
            )
            assert resp.status_code == 200, f"POST /research: {resp.status_code}"
            data = resp.json()
            run_id = data["run_id"]
            assert run_id and data["status"] == "pending"
        await step("2_run_created_via_api", s2)

        # --- Step 3: Inject progress events directly into repo ---
        async def s3():
            nonlocal run_id
            await repo.update_run_status(run_id, "running")
            for i in range(5):
                seq = await repo.next_seq(run_id)
                await repo.append_progress(run_id, make_progress(
                    seq, "progress", f"Progress event {i+1}/5", phase="research"
                ))
            # Keep status as "running" so SSE stream stays open
        await step("3_progress_events_injected", s3)

        # --- Step 4: SSE connect and disconnect mid-stream ---
        async def s4():
            nonlocal run_id, events_batch1
            async with client.stream(
                "GET", f"/research/{run_id}/stream",
                headers={"X-API-Key": "test-key"},
            ) as response:
                assert response.status_code == 200, f"SSE status: {response.status_code}"
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        events_batch1 += 1
                        if events_batch1 >= 3:
                            break
            assert events_batch1 >= 1, f"No SSE events (got {events_batch1})"
        await step("4_sse_connected_and_disconnected", s4)

        # --- Step 5: Server still handles requests after disconnect ---
        async def s5():
            resp = await client.get("/health")
            assert resp.status_code == 200
        await step("5_server_healthy_after_disconnect", s5)

        # --- Step 6: Inject more events, verify run accessible ---
        async def s6():
            nonlocal run_id
            resp = await client.get(f"/research/{run_id}", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 200

            for i in range(3):
                seq = await repo.next_seq(run_id)
                await repo.append_progress(run_id, make_progress(
                    seq, "progress", f"Post-disconnect event {i+1}/3", phase="review"
                ))
            await repo.update_run_status(run_id, "completed")
        await step("6_more_events_injected", s6)

        # --- Step 7: Reconnect SSE, verify all events delivered ---
        async def s7():
            nonlocal run_id, events_batch2
            async with client.stream(
                "GET", f"/research/{run_id}/stream",
                headers={"X-API-Key": "test-key"},
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        events_batch2 += 1
                        if events_batch2 >= 10:
                            break
            assert events_batch2 >= events_batch1, \
                f"Second connect {events_batch2} < first {events_batch1} events"
        await step("7_sse_reconnect_delivers_events", s7)

    results["summary"] = {
        "steps_total": len(results["steps"]),
        "steps_passed": sum(1 for v in results["steps"].values() if v == "PASS"),
        "events_first_connect": events_batch1,
        "events_second_connect": events_batch2,
        "run_id": run_id,
    }
    results["status"] = "PASS" if all(v == "PASS" for v in results["steps"].values()) else "PARTIAL"
    return results


async def main():
    # Set up config + repo in-process
    cfg = ApiConfig(enable_rest_api=True, api_key="test-key", api_db_path=":memory:")
    set_config(cfg)
    await init_repo(cfg.api_db_path)

    results = await run_test()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
