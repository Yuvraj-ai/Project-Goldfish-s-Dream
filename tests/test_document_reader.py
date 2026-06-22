"""Tests for document reader."""
from unittest.mock import MagicMock, patch

import pytest

from open_deep_research.document_reader import (
    DocumentArtifact,
    DocumentMetadata,
    DocumentReader,
    DocumentSection,
)


class TestDocumentArtifact:
    def test_artifact_creation(self):
        artifact = DocumentArtifact()
        assert artifact.sections == []
        assert artifact.parse_confidence == 1.0

    def test_artifact_with_sections(self):
        sections = [DocumentSection(id="s1", heading="Intro", level=1, content="Hello")]
        artifact = DocumentArtifact(sections=sections)
        assert len(artifact.sections) == 1

    def test_artifact_defaults(self):
        artifact = DocumentArtifact()
        assert artifact.file_path == ""
        assert artifact.tables == []
        assert artifact.images == []


class TestDocumentReader:
    def test_reader_creation(self):
        reader = DocumentReader()
        assert reader.use_grobid is False

    def test_reader_creation_with_grobid(self):
        reader = DocumentReader(use_grobid=True)
        assert reader.use_grobid is True

    @pytest.mark.asyncio
    async def test_parse_nonexistent_returns_empty(self):
        reader = DocumentReader()
        artifact = await reader.parse_pdf("nonexistent.pdf")
        assert isinstance(artifact, DocumentArtifact)
        assert artifact.parse_confidence == 0.0

    @pytest.mark.asyncio
    async def test_parse_pdf_importerror(self):
        reader = DocumentReader()
        with patch.dict("sys.modules", {"pymupdf": None}), patch("os.path.exists", return_value=True):
            result = await reader.parse_pdf("test.pdf")
        assert result.parse_confidence == 0.0
        assert result.file_path == "test.pdf"

    @pytest.mark.asyncio
    async def test_parse_pdf_generic_exception(self):
        reader = DocumentReader()
        with patch("os.path.exists", return_value=True):
            result = await reader.parse_pdf("test.pdf")
        assert result.parse_confidence == 0.0
        assert result.file_path == "test.pdf"

    @pytest.mark.asyncio
    async def test_parse_pdf_success(self):
        mock_page = MagicMock()
        mock_page.get_text.side_effect = lambda *args, **kw: (
            "Hello World\n\nThis is a test document." if not args else
            {"blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {"text": "Introduction", "size": 20, "flags": 16}
                            ]
                        }
                    ]
                }
            ]}
        )

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page

        mock_pymupdf = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=12345),
            patch.dict("sys.modules", {"pymupdf": mock_pymupdf}),
        ):
            reader = DocumentReader()
            result = await reader.parse_pdf("/path/to/test.pdf")

        assert result.parse_confidence == 1.0
        assert result.file_path == "/path/to/test.pdf"
        assert result.metadata.page_count == 1
        assert result.metadata.title == "test.pdf"
        assert result.metadata.file_size == 12345
        assert len(result.sections) == 1
        assert result.sections[0].heading == "Introduction"
        assert result.sections[0].page_start == 1
        assert "Hello World" in result.raw_text

    @pytest.mark.asyncio
    async def test_extract_evidence_from_empty(self):
        reader = DocumentReader()
        artifact = DocumentArtifact(raw_text="")
        cards = await reader.extract_evidence_cards(artifact)
        assert cards == []

    @pytest.mark.asyncio
    async def test_extract_evidence_low_confidence(self):
        reader = DocumentReader()
        artifact = DocumentArtifact(parse_confidence=0.1, raw_text="Some text")
        cards = await reader.extract_evidence_cards(artifact)
        assert cards == []

    @pytest.mark.asyncio
    async def test_extract_evidence_short_paragraphs_skipped(self):
        reader = DocumentReader()
        text = "Short.\n\n" + "A" * 60 + "\n\n" + "Tiny.\n\n" + "B" * 70
        artifact = DocumentArtifact(raw_text=text, parse_confidence=1.0)
        cards = await reader.extract_evidence_cards(artifact)
        assert len(cards) == 2
        for card in cards:
            assert len(card["claim"]) > 50

    @pytest.mark.asyncio
    async def test_extract_evidence_max_20_paragraphs(self):
        reader = DocumentReader()
        long_para = "A" * 100
        paragraphs = "\n\n".join([long_para] * 30)
        artifact = DocumentArtifact(raw_text=paragraphs, parse_confidence=1.0)
        cards = await reader.extract_evidence_cards(artifact)
        assert len(cards) == 20

    @pytest.mark.asyncio
    async def test_extract_evidence_from_text(self):
        reader = DocumentReader()
        text = "This is a long paragraph about a very important topic that should be extracted as evidence.\n\n" * 5
        artifact = DocumentArtifact(raw_text=text, parse_confidence=1.0)
        cards = await reader.extract_evidence_cards(artifact)
        assert len(cards) > 0
        assert "claim" in cards[0]
        assert "confidence" in cards[0]
        header_title = "test_" in cards[0]["id"]
        assert header_title or "doc__" in cards[0]["id"]


class TestDocumentMetadata:
    def test_metadata_defaults(self):
        meta = DocumentMetadata()
        assert meta.title == ""
        assert meta.page_count == 0

    def test_metadata_with_values(self):
        meta = DocumentMetadata(title="Test", page_count=10)
        assert meta.title == "Test"
        assert meta.page_count == 10
