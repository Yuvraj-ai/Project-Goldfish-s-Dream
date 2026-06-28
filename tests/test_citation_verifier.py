"""Tests for citation verification system."""

import pytest
import respx
from httpx import Response, TimeoutException

from open_deep_research.citation_verifier import CitationVerifier


@pytest.mark.asyncio
async def test_verify_url_alive():
    """Test that alive URL returns correct status."""
    async with respx.mock:
        respx.get("https://example.com/article").mock(return_value=Response(200, text="Content"))

        verifier = CitationVerifier()
        result = await verifier.verify_url("https://example.com/article")

        assert result["status"] == "alive"
        assert result["status_code"] == 200


@pytest.mark.asyncio
async def test_verify_url_dead():
    """Test that dead URL returns correct status."""
    async with respx.mock:
        respx.get("https://example.com/dead").mock(return_value=Response(404))

        verifier = CitationVerifier()
        result = await verifier.verify_url("https://example.com/dead")

        assert result["status"] == "dead"
        assert result["status_code"] == 404


@pytest.mark.asyncio
async def test_verify_url_blocked():
    """Test that bot-blocked URL returns unverified."""
    async with respx.mock:
        respx.get("https://example.com/blocked").mock(return_value=Response(403))

        verifier = CitationVerifier()
        result = await verifier.verify_url("https://example.com/blocked")

        assert result["status"] == "unverified"
        assert result["status_code"] == 403


@pytest.mark.asyncio
async def test_verify_url_timeout():
    """Test that timeout returns unverified."""
    async with respx.mock:
        respx.get("https://example.com/slow").mock(side_effect=Exception("Timeout"))

        verifier = CitationVerifier()
        result = await verifier.verify_url("https://example.com/slow")

        assert result["status"] == "unverified"


@pytest.mark.asyncio
async def test_verify_batch_parallel():
    """Test batch verification runs in parallel."""
    async with respx.mock:
        respx.get("https://example.com/1").mock(return_value=Response(200, text="OK"))
        respx.get("https://example.com/2").mock(return_value=Response(404))
        respx.get("https://example.com/3").mock(return_value=Response(200, text="OK"))

        verifier = CitationVerifier()
        results = await verifier.verify_batch([
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ])

        assert results["https://example.com/1"]["status"] == "alive"
        assert results["https://example.com/2"]["status"] == "dead"
        assert results["https://example.com/3"]["status"] == "alive"


@pytest.mark.asyncio
async def test_verify_url_server_error():
    """Server error (500) returns alive status."""
    async with respx.mock:
        respx.get("https://example.com/error").mock(return_value=Response(500))
        verifier = CitationVerifier()
        result = await verifier.verify_url("https://example.com/error")
        assert result["status"] == "alive"
        assert result["status_code"] == 500


@pytest.mark.asyncio
async def test_verify_url_timeout_with_head_fallback():
    """TimeoutException triggers HEAD fallback."""
    async with respx.mock:
        respx.get("https://example.com/timeout").mock(side_effect=TimeoutException("timed out"))
        respx.head("https://example.com/timeout").mock(return_value=Response(200))
        verifier = CitationVerifier()
        result = await verifier.verify_url("https://example.com/timeout")
        assert result["status"] == "alive"


@pytest.mark.asyncio
async def test_verify_url_timeout_head_fails():
    """When both GET and HEAD timeout, return unverified."""
    async with respx.mock:
        respx.get("https://example.com/bad").mock(side_effect=TimeoutException("timed out"))
        respx.head("https://example.com/bad").mock(side_effect=Exception("head also failed"))
        verifier = CitationVerifier()
        result = await verifier.verify_url("https://example.com/bad")
        assert result["status"] == "unverified"


@pytest.mark.asyncio
async def test_verifier_close():
    """Close method cleans up shared client."""
    async with respx.mock:
        respx.get("https://example.com/close").mock(return_value=Response(200, text="OK"))
        verifier = CitationVerifier()
        await verifier.verify_url("https://example.com/close")
        await verifier.close()
        # close again is a no-op
        await verifier.close()
