from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_notify_matching_webhooks():
    import asyncio

    import httpx

    from open_deep_research.api.models import WebhookConfig
    from open_deep_research.api.webhooks import WebhookNotifier

    repo = AsyncMock()
    wh = WebhookConfig(
        id="wh1", url="https://example.com/hook",
        events=["research.completed"],
    )
    repo.list_webhooks = AsyncMock(return_value=[wh])

    notifier = WebhookNotifier(repo)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post = AsyncMock(return_value=mock_response)

    with patch.object(httpx.AsyncClient, "post", mock_post):
        await notifier.notify("research.completed", {"run_id": "r1"})
        if notifier._delivery_tasks:
            await asyncio.gather(*notifier._delivery_tasks, return_exceptions=True)
        mock_post.assert_awaited_once()


@pytest.mark.asyncio
async def test_notify_no_matching_webhooks():
    from open_deep_research.api.models import WebhookConfig
    from open_deep_research.api.webhooks import WebhookNotifier

    repo = AsyncMock()
    wh = WebhookConfig(
        id="wh1", url="https://example.com/hook",
        events=["research.started"],
    )
    repo.list_webhooks = AsyncMock(return_value=[wh])

    notifier = WebhookNotifier(repo)
    await notifier.notify("research.completed", {"run_id": "r1"})


def test_generate_signature():
    from open_deep_research.api.webhooks import generate_signature
    sig = generate_signature("s3cr3t", b'{"key":"val"}')
    assert isinstance(sig, str)
    assert len(sig) > 0


def test_webhook_config_model():
    from open_deep_research.api.models import WebhookConfig
    wh = WebhookConfig(
        url="https://example.com/hook",
        events=["research.completed"],
    )
    assert wh.active is True
    assert wh.timeout_seconds == 5
    assert wh.retry_max == 3
