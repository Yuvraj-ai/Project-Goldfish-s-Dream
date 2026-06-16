"""Tests for document reader."""
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

    @pytest.mark.asyncio
    async def test_parse_nonexistent_returns_empty(self):
        reader = DocumentReader()
        artifact = await reader.parse_pdf("nonexistent.pdf")
        assert isinstance(artifact, DocumentArtifact)
        assert artifact.parse_confidence == 0.0

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
    async def test_extract_evidence_from_text(self):
        reader = DocumentReader()
        text = "This is a long paragraph about a very important topic that should be extracted as evidence.\n\n" * 5
        artifact = DocumentArtifact(raw_text=text, parse_confidence=1.0)
        cards = await reader.extract_evidence_cards(artifact)
        assert len(cards) > 0
        assert "claim" in cards[0]
        assert "confidence" in cards[0]


class TestDocumentMetadata:
    def test_metadata_defaults(self):
        meta = DocumentMetadata()
        assert meta.title == ""
        assert meta.page_count == 0

    def test_metadata_with_values(self):
        meta = DocumentMetadata(title="Test", page_count=10)
        assert meta.title == "Test"
        assert meta.page_count == 10
