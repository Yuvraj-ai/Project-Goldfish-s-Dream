"""Tests for multi-format export pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from open_deep_research.exporters import (
    EXPORTER_MAP,
    BaseExporter,
    BibTeXExporter,
    DOCXExporter,
    HTMLExporter,
    JSONExporter,
    MarkdownExporter,
    PDFExporter,
    get_exporter,
)

SAMPLE_MARKDOWN = "# Test Report\n\nThis is a test report.\n\n## Section 1\n\nContent here."
SAMPLE_MARKDOWN_RICH = "# Main Title\n\nIntro paragraph.\n\n## Level 2\n\nSome text.\n\n### Level 3\n\nDeeper text.\n\n- item one\n- item two\n\nPlain paragraph."
SAMPLE_METADATA = {
    "title": "Test Report",
    "generated_at": "2024-01-15",
    "profile": "executive_brief",
    "sources": [
        {"url": "https://example.com", "title": "Example", "publisher": "Example Press", "date": "2024-01-15"},
    ],
}


class _ConcreteChild(BaseExporter):
    async def export(self, markdown: str, metadata: dict, output_path: Path) -> Path:
        return await super().export(markdown, metadata, output_path)


@pytest.mark.asyncio
async def test_base_exporter_abstract_method(tmp_path):
    """Test that the abstract base method can be called via super()."""
    exporter = _ConcreteChild()
    result = await exporter.export("hello", {}, tmp_path / "report")
    assert result is None


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
    assert "font-family" in content
    assert "max-width: 800px" in content
    assert "body" in content


@pytest.mark.asyncio
async def test_pdf_export(tmp_path):
    """Test PDF export with mocked weasyprint."""
    pdf_path = tmp_path / "report.pdf"

    def _write_pdf(target_path):
        pdf_path.write_text("mock pdf content")

    mock_html_instance = MagicMock()
    mock_html_instance.write_pdf.side_effect = _write_pdf
    mock_html_cls = MagicMock(return_value=mock_html_instance)
    mock_weasyprint = MagicMock()
    mock_weasyprint.HTML = mock_html_cls

    with patch.dict("sys.modules", {"weasyprint": mock_weasyprint}):
        exporter = PDFExporter()
        result = await exporter.export(SAMPLE_MARKDOWN, SAMPLE_METADATA, tmp_path / "report")

    assert result == pdf_path
    assert result.exists()
    assert result.suffix == ".pdf"
    assert result.read_text() == "mock pdf content"
    mock_html_cls.assert_called_once()
    mock_html_instance.write_pdf.assert_called_once_with(str(result))


@pytest.mark.asyncio
async def test_docx_export(tmp_path):
    """Test DOCX export with mocked python-docx."""
    docx_path = tmp_path / "report.docx"

    def _save(target_path):
        docx_path.write_text("mock docx content")

    mock_document = MagicMock()
    mock_document.save.side_effect = _save
    mock_document.add_heading.return_value = MagicMock()
    mock_document.add_paragraph.return_value = MagicMock()

    mock_docx = MagicMock()
    mock_docx.Document = MagicMock(return_value=mock_document)
    mock_docx_enum = MagicMock()
    mock_docx_enum_text = MagicMock()
    mock_docx_enum_text.WD_ALIGN_PARAGRAPH = MagicMock()
    mock_docx_enum_text.WD_ALIGN_PARAGRAPH.CENTER = "CENTER"

    with patch.dict(
        "sys.modules",
        {
            "docx": mock_docx,
            "docx.enum": mock_docx_enum,
            "docx.enum.text": mock_docx_enum_text,
        },
    ):
        exporter = DOCXExporter()
        result = await exporter.export(SAMPLE_MARKDOWN_RICH, SAMPLE_METADATA, tmp_path / "report")

    assert result == docx_path
    assert result.exists()
    assert result.suffix == ".docx"
    assert result.read_text() == "mock docx content"
    mock_document.save.assert_called_once_with(str(result))
    mock_document.add_heading.assert_any_call("Test Report", 0)


@pytest.mark.asyncio
async def test_json_export(tmp_path):
    """Test JSON export."""
    exporter = JSONExporter()
    result = await exporter.export(SAMPLE_MARKDOWN, SAMPLE_METADATA, tmp_path / "report")
    assert result.exists()
    assert result.suffix == ".json"
    data = json.loads(result.read_text())
    assert data["report"]["markdown"] == SAMPLE_MARKDOWN
    assert data["report"]["title"] == "Test Report"
    assert data["report"]["generated_at"] == "2024-01-15"
    assert data["report"]["profile"] == "executive_brief"
    assert data["report"]["word_count"] > 0
    assert data["metadata"] == SAMPLE_METADATA


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


def test_get_exporter_pdf():
    """Test get_exporter returns PDF exporter."""
    exporter = get_exporter("pdf")
    assert isinstance(exporter, PDFExporter)


def test_get_exporter_docx():
    """Test get_exporter returns DOCX exporter."""
    exporter = get_exporter("docx")
    assert isinstance(exporter, DOCXExporter)


def test_get_exporter_json():
    """Test get_exporter returns JSON exporter."""
    exporter = get_exporter("json")
    assert isinstance(exporter, JSONExporter)


def test_get_exporter_bibtex():
    """Test get_exporter returns BibTeX exporter."""
    exporter = get_exporter("bibtex")
    assert isinstance(exporter, BibTeXExporter)


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
    assert len(EXPORTER_MAP) == 6
