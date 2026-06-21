from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import aiosqlite

from open_deep_research.api.models import ProgressEvent, RunRecord, WebhookConfig
from open_deep_research.api.repository import ResearchRepository


class SqliteResearchRepository(ResearchRepository):
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    @classmethod
    async def create(cls, db_path: str) -> SqliteResearchRepository:
        db = await aiosqlite.connect(db_path)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.executescript(_SCHEMA)
        await db.commit()
        return cls(db)

    async def close(self) -> None:
        await self.db.close()

    async def create_run(
        self, query: str, config: dict,
        idempotency_key: str | None = None,
    ) -> str:
        if idempotency_key:
            row = await self.db.execute(
                "SELECT id FROM runs WHERE idempotency_key = ?",
                (idempotency_key,),
            )
            existing = await row.fetchone()
            if existing:
                return existing["id"]

        run_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "INSERT INTO runs (id, query, config, idempotency_key, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, query, json.dumps(config), idempotency_key, "pending", now),
        )
        await self.db.commit()
        return run_id

    async def get_run(self, run_id: str) -> RunRecord | None:
        row = await self.db.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,),
        )
        row_data = await row.fetchone()
        if row_data is None:
            return None
        return RunRecord(
            id=row_data["id"],
            query=row_data["query"],
            config=json.loads(row_data["config"]),
            idempotency_key=row_data["idempotency_key"],
            status=row_data["status"],
            error=row_data["error"],
            created_at=row_data["created_at"],
            completed_at=row_data["completed_at"],
        )

    async def update_run_status(
        self, run_id: str, status: str, error: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if status in ("completed", "failed", "cancelled"):
            await self.db.execute(
                "UPDATE runs SET status = ?, error = ?, completed_at = ? WHERE id = ?",
                (status, error, now, run_id),
            )
        else:
            await self.db.execute(
                "UPDATE runs SET status = ?, error = ? WHERE id = ?",
                (status, error, run_id),
            )
        await self.db.commit()

    async def append_progress(
        self, run_id: str, event: ProgressEvent,
    ) -> None:
        await self.db.execute(
            "INSERT INTO progress_events (run_id, seq, event_type, phase, message, metadata, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, event.seq, event.event_type, event.phase,
             event.message, json.dumps(event.metadata) if event.metadata else None,
             event.timestamp),
        )
        await self.db.commit()

    async def progress_after(
        self, run_id: str, seq: int,
    ) -> list[ProgressEvent]:
        rows = await self.db.execute(
            "SELECT * FROM progress_events WHERE run_id = ? AND seq > ? ORDER BY seq",
            (run_id, seq),
        )
        results = []
        async for row in rows:
            results.append(ProgressEvent(
                seq=row["seq"],
                event_type=row["event_type"],
                phase=row["phase"],
                message=row["message"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                timestamp=row["timestamp"],
            ))
        return results

    async def next_seq(self, run_id: str) -> int:
        row = await self.db.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS nxt FROM progress_events WHERE run_id = ?",
            (run_id,),
        )
        result = await row.fetchone()
        return result["nxt"]

    async def save_checkpoint(self, run_id: str, state: bytes) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "INSERT OR REPLACE INTO checkpoints (run_id, state, updated_at) VALUES (?, ?, ?)",
            (run_id, state, now),
        )
        await self.db.commit()

    async def load_checkpoint(self, run_id: str) -> bytes | None:
        row = await self.db.execute(
            "SELECT state FROM checkpoints WHERE run_id = ?", (run_id,),
        )
        result = await row.fetchone()
        return bytes(result["state"]) if result else None

    async def save_report(self, run_id: str, report: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "INSERT OR REPLACE INTO reports (run_id, data, created_at) VALUES (?, ?, ?)",
            (run_id, json.dumps(report), now),
        )
        await self.db.commit()

    async def get_report(self, run_id: str) -> dict | None:
        row = await self.db.execute(
            "SELECT data FROM reports WHERE run_id = ?", (run_id,),
        )
        result = await row.fetchone()
        return json.loads(result["data"]) if result else None

    async def save_memory(
        self, namespace: str, key: str, data: dict,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "INSERT OR REPLACE INTO memory (namespace, key, data, updated_at) VALUES (?, ?, ?, ?)",
            (namespace, key, json.dumps(data), now),
        )
        await self.db.commit()

    async def load_memory(
        self, namespace: str, key: str,
    ) -> dict | None:
        row = await self.db.execute(
            "SELECT data FROM memory WHERE namespace = ? AND key = ?",
            (namespace, key),
        )
        result = await row.fetchone()
        return json.loads(result["data"]) if result else None

    async def list_memory_keys(self, namespace: str) -> list[str]:
        async with self.db.execute(
            "SELECT key FROM memory WHERE namespace = ?", (namespace,)
        ) as cursor:
            rows = await cursor.fetchall()
        return [row["key"] for row in rows]

    async def list_webhooks(self) -> list[WebhookConfig]:
        rows = await self.db.execute("SELECT * FROM webhook_configs")
        results = []
        async for row in rows:
            results.append(WebhookConfig(
                id=row["id"],
                url=row["url"],
                events=json.loads(row["events"]),
                secret=row["secret"],
                active=bool(row["active"]),
                timeout_seconds=row["timeout_seconds"],
                retry_max=row["retry_max"],
            ))
        return results

    async def save_webhook(self, config: WebhookConfig) -> None:
        await self.db.execute(
            "INSERT OR REPLACE INTO webhook_configs "
            "(id, url, events, secret, active, timeout_seconds, retry_max) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (config.id, config.url, json.dumps(config.events),
             config.secret, int(config.active),
             config.timeout_seconds, config.retry_max),
        )
        await self.db.commit()

    async def delete_webhook(self, webhook_id: str) -> None:
        await self.db.execute(
            "DELETE FROM webhook_configs WHERE id = ?", (webhook_id,),
        )
        await self.db.commit()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    config TEXT NOT NULL,
    idempotency_key TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS progress_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    phase TEXT,
    message TEXT NOT NULL,
    metadata TEXT,
    timestamp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id TEXT PRIMARY KEY REFERENCES runs(id),
    state BLOB NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    run_id TEXT PRIMARY KEY REFERENCES runs(id),
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE TABLE IF NOT EXISTS webhook_configs (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    events TEXT NOT NULL,
    secret TEXT,
    active INTEGER DEFAULT 1,
    timeout_seconds INTEGER DEFAULT 5,
    retry_max INTEGER DEFAULT 3
);
CREATE INDEX IF NOT EXISTS idx_progress_run_id ON progress_events(run_id);
CREATE INDEX IF NOT EXISTS idx_progress_seq ON progress_events(run_id, seq);
CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_idempotency ON runs(idempotency_key) WHERE idempotency_key IS NOT NULL;
"""
