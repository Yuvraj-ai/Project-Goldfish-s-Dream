"""Tests for citation formatting engine."""


from open_deep_research.citation import CitationEntry, CitationFormatter

SAMPLE_SOURCE = {
    "url": "https://example.com/article",
    "title": "Test Article",
    "publisher": "Example Press",
    "date": "2024-01-15",
    "source_type": "web",
}


def test_vanilla_inline_citation():
    """Test vanilla inline citation format."""
    formatter = CitationFormatter("vanilla")
    result = formatter.format_inline(SAMPLE_SOURCE)
    assert "Example Press" in result
    assert "2024" in result


def test_apa_inline_citation():
    """Test APA inline citation format."""
    formatter = CitationFormatter("apa")
    result = formatter.format_inline(SAMPLE_SOURCE)
    assert "2024" in result


def test_mla_inline_citation():
    """Test MLA inline citation format."""
    formatter = CitationFormatter("mla")
    result = formatter.format_inline(SAMPLE_SOURCE)
    assert "2024" in result


def test_chicago_inline_citation():
    """Test Chicago inline citation format."""
    formatter = CitationFormatter("chicago")
    result = formatter.format_inline(SAMPLE_SOURCE)
    assert "2024" in result
    assert "example.com" in result


def test_harvard_inline_citation():
    """Test Harvard inline citation format."""
    formatter = CitationFormatter("harvard")
    result = formatter.format_inline(SAMPLE_SOURCE)
    assert "2024" in result


def test_ieee_inline_citation():
    """Test IEEE inline citation format."""
    formatter = CitationFormatter("ieee")
    source_with_index = {**SAMPLE_SOURCE, "_index": 5}
    result = formatter.format_inline(source_with_index)
    assert "[5]" in result


def test_vanilla_bibliography():
    """Test vanilla bibliography format."""
    formatter = CitationFormatter("vanilla")
    result = formatter.format_bibliography([SAMPLE_SOURCE])
    assert "Test Article" in result
    assert "Example Press" in result


def test_apa_bibliography():
    """Test APA bibliography format."""
    formatter = CitationFormatter("apa")
    result = formatter.format_bibliography([SAMPLE_SOURCE])
    assert "Test Article" in result


def test_ieee_bibliography():
    """Test IEEE bibliography format."""
    formatter = CitationFormatter("ieee")
    result = formatter.format_bibliography([SAMPLE_SOURCE])
    assert "[1]" in result
    assert "Test Article" in result


def test_bibtex_export():
    """Test BibTeX export format."""
    formatter = CitationFormatter("vanilla")
    result = formatter.export_bibtex([SAMPLE_SOURCE])
    assert "@misc{source1," in result
    assert "title = {Test Article}" in result
    assert "url = {https://example.com/article}" in result


def test_citation_entry_structure():
    """Test CitationEntry structure."""
    entry = CitationEntry(
        authors=["Author One"],
        title="Test",
        year=2024,
        url="https://example.com",
        doi="10.1234/test",
        source_type="academic",
        publisher="Test Publisher",
    )
    assert entry.authors == ["Author One"]
    assert entry.year == 2024
    assert entry.doi == "10.1234/test"


def test_inline_citation_missing_date():
    """Test inline citation with missing date."""
    formatter = CitationFormatter("vanilla")
    source_no_date = {**SAMPLE_SOURCE, "date": ""}
    result = formatter.format_inline(source_no_date)
    assert "n.d." in result
