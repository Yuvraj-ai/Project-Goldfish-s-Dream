"""Tests for multi-format export pipeline."""

import json

import pytest

from open_deep_research.exporters import (
    EXPORTER_MAP,
    BibTeXExporter,
    HTMLExporter,
    JSONExporter,
    MarkdownExporter,
    get_exporter,
)

SAMPLE_MARKDOWN = "# Test Report\n\nThis is a test report.\n\n## Section 1\n\nContent here."
SAMPLE_METADATA = {
    "title": "Test Report",
    "generated_at": "2024-01-15",
    "profile": "executive_brief",
    "sources": [
        {"url": "https://example.com", "title": "Example", "publisher": "Example Press", "date": "2024-01-15"},
    ],
}


@pytest.mark.asyncio
async def test_markdown_export(tmp_path):
    """Test markdown export."""
    exporter = MarkdownExporter()
    result = await exporter.export(SAMPLE_MARKDOWN, SAMPLE_METADATA, tmp_path / "report")
    assert result.exists()
    assert result.suffix == ".md"
    assert "Test Report" in result.read_text()


@pytest.mark.asyncio
async def test_html_export(tmp_path):
    """Test HTML export."""
    exporter = HTMLExporter()
    result = await exporter.export(SAMPLE_MARKDOWN, SAMPLE_METADATA, tmp_path / "report")
    assert result.exists()
    assert result.suffix == ".html"
    content = result.read_text()
    assert "<html" in content
    assert "Test Report" in content


@pytest.mark.asyncio
async def test_json_export(tmp_path):
    """Test JSON export."""
    exporter = JSONExporter()
    result = await exporter.export(SAMPLE_MARKDOWN, SAMPLE_METADATA, tmp_path / "report")
    assert result.exists()
    assert result.suffix == ".json"
    data = json.loads(result.read_text())
    assert data["report"]["title"] == "Test Report"
    assert data["report"]["word_count"] > 0


@pytest.mark.asyncio
async def test_bibtex_export(tmp_path):
    """Test BibTeX export."""
    exporter = BibTeXExporter()
    result = await exporter.export(SAMPLE_MARKDOWN, SAMPLE_METADATA, tmp_path / "report")
    assert result.exists()
    assert result.suffix == ".bib"
    content = result.read_text()
    assert "@misc{source1," in content


def test_get_exporter_markdown():
    """Test get_exporter returns correct exporter."""
    exporter = get_exporter("markdown")
    assert isinstance(exporter, MarkdownExporter)


def test_get_exporter_html():
    """Test get_exporter returns HTML exporter."""
    exporter = get_exporter("html")
    assert isinstance(exporter, HTMLExporter)


def test_get_exporter_unknown():
    """Test get_exporter raises ValueError for unknown format."""
    with pytest.raises(ValueError, match="Unknown export format"):
        get_exporter("unknown_format")


def test_exporter_map_has_all_formats():
    """Test that EXPORTER_MAP contains all expected formats."""
    assert "markdown" in EXPORTER_MAP
    assert "html" in EXPORTER_MAP
    assert "pdf" in EXPORTER_MAP
    assert "docx" in EXPORTER_MAP
    assert "json" in EXPORTER_MAP
    assert "bibtex" in EXPORTER_MAP
