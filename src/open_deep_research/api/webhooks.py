from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Any

import httpx

from open_deep_research.api.models import WebhookConfig
from open_deep_research.api.repository import ResearchRepository

logger = logging.getLogger(__name__)


def generate_signature(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class WebhookNotifier:
    def __init__(self, repo: ResearchRepository) -> None:
        self._repo = repo
        self._client: httpx.AsyncClient | None = None
        self._delivery_tasks: set[asyncio.Task] = set()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def notify(self, event_type: str, payload: dict[str, Any]) -> None:
        webhooks = await self._repo.list_webhooks()
        matching = [wh for wh in webhooks if wh.active and event_type in wh.events]
        if not matching:
            return

        body = json.dumps(payload).encode()
        client = await self._get_client()
        for wh in matching:
            task = asyncio.create_task(
                self._deliver(client, wh, event_type, body)
            )
            self._delivery_tasks.add(task)
            task.add_done_callback(self._delivery_tasks.discard)

    async def _deliver(
        self,
        client: httpx.AsyncClient,
        wh: WebhookConfig,
        event_type: str,
        body: bytes,
    ) -> None:
        headers = {"Content-Type": "application/json"}
        if wh.secret:
            headers["X-Webhook-Signature"] = generate_signature(wh.secret, body)

        for attempt in range(wh.retry_max):
            try:
                response = await client.post(
                    wh.url,
                    content=body,
                    headers=headers,
                    timeout=wh.timeout_seconds,
                )
                if response.status_code < 500:
                    return
            except Exception:
                logger.exception("Webhook delivery attempt %d failed", attempt + 1)

            if attempt < wh.retry_max - 1:
                await asyncio.sleep(2 ** attempt)

    async def close(self) -> None:
        if self._delivery_tasks:
            await asyncio.gather(*self._delivery_tasks, return_exceptions=True)
        if self._client:
            await self._client.aclose()
            self._client = None
