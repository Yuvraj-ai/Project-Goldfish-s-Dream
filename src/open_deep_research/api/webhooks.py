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
            logger.debug("No matching webhooks for event %s", event_type)
            return

        body = json.dumps(payload).encode()
        logger.info(
            "Dispatching webhook event %s to %d endpoint(s) (payload=%d bytes)",
            event_type, len(matching), len(body),
        )
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

        logger.debug(
            "Delivering webhook %s event=%s to %s (%d bytes, retry_max=%d)",
            wh.id, event_type, wh.url, len(body), wh.retry_max,
        )
        for attempt in range(wh.retry_max):
            try:
                response = await client.post(
                    wh.url,
                    content=body,
                    headers=headers,
                    timeout=wh.timeout_seconds,
                )
                if response.status_code < 500:
                    logger.info(
                        "Webhook %s delivered event=%s (status=%d, attempt=%d)",
                        wh.id, event_type, response.status_code, attempt + 1,
                    )
                    return
                logger.warning(
                    "Webhook %s delivery failed event=%s (status=%d, attempt=%d, will retry=%s)",
                    wh.id, event_type, response.status_code, attempt + 1,
                    attempt < wh.retry_max - 1,
                )
            except Exception:
                logger.exception(
                    "Webhook %s delivery attempt %d failed (event=%s, will retry=%s)",
                    wh.id, attempt + 1, event_type, attempt < wh.retry_max - 1,
                )

            if attempt < wh.retry_max - 1:
                await asyncio.sleep(2 ** attempt)

        logger.error(
            "Webhook %s failed to deliver event=%s after %d attempt(s)",
            wh.id, event_type, wh.retry_max,
        )

    @staticmethod
    def _matches_event(wh: WebhookConfig, event_type: str) -> bool:
        if not wh.events:
            return True
        return event_type in wh.events

    def _sign(self, payload: dict, secret: str) -> str:
        return generate_signature(secret, json.dumps(payload).encode())

    async def close(self) -> None:
        if self._delivery_tasks:
            logger.debug("Awaiting %d in-flight webhook delivery task(s)", len(self._delivery_tasks))
            await asyncio.gather(*self._delivery_tasks, return_exceptions=True)
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("WebhookNotifier closed")
