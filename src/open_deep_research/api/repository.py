from __future__ import annotations

from abc import ABC, abstractmethod

from open_deep_research.api.models import ProgressEvent, RunRecord, WebhookConfig


class ResearchRepository(ABC):
    @abstractmethod
    async def create_run(
        self, query: str, config: dict,
        idempotency_key: str | None = None,
    ) -> str: ...

    @abstractmethod
    async def get_run(self, run_id: str) -> RunRecord | None: ...

    @abstractmethod
    async def update_run_status(
        self, run_id: str, status: str, error: str | None = None,
    ) -> None: ...

    @abstractmethod
    async def append_progress(
        self, run_id: str, event: ProgressEvent,
    ) -> None: ...

    @abstractmethod
    async def progress_after(
        self, run_id: str, seq: int,
    ) -> list[ProgressEvent]: ...

    @abstractmethod
    async def next_seq(self, run_id: str) -> int: ...

    @abstractmethod
    async def save_checkpoint(
        self, run_id: str, state: bytes,
    ) -> None: ...

    @abstractmethod
    async def load_checkpoint(self, run_id: str) -> bytes | None: ...

    @abstractmethod
    async def save_report(self, run_id: str, report: dict) -> None: ...

    @abstractmethod
    async def get_report(self, run_id: str) -> dict | None: ...

    @abstractmethod
    async def save_memory(
        self, namespace: str, key: str, data: dict,
    ) -> None: ...

    @abstractmethod
    async def load_memory(
        self, namespace: str, key: str,
    ) -> dict | None: ...

    @abstractmethod
    async def list_memory_keys(self, namespace: str) -> list[str]: ...

    @abstractmethod
    async def list_webhooks(self) -> list[WebhookConfig]: ...

    @abstractmethod
    async def save_webhook(self, config: WebhookConfig) -> None: ...

    @abstractmethod
    async def delete_webhook(self, webhook_id: str) -> None: ...
