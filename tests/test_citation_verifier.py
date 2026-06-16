"""Tests for citation verification system."""

import pytest
import respx
from httpx import Response

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
