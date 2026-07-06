"""Tests for citation index building and rendering."""

import pytest

from open_deep_research.deep_researcher import (
    _build_citation_index,
    _format_evidence_cards,
    _replace_raw_ids_in_text,
    _format_bibliography_from_index,
    compile_report,
)


SAMPLE_CARDS = [
    {"id": "researcher_1_0", "claim": "Python 3.12 has improvements", "supporting_source_ids": ["https://example.com/python"], "confidence": 0.8},
    {"id": "researcher_1_1", "claim": "GIL removal is complex", "supporting_source_ids": ["https://example.com/gil"], "confidence": 0.7},
    {"id": "researcher_2_0", "claim": "Perf improvements in 3.12", "supporting_source_ids": ["https://example.com/perf"], "confidence": 0.9},
]

SAMPLE_SOURCES = [
    {"url": "https://example.com/python", "title": "Python 3.12 Release", "publisher": "PSF", "date": "2024-10-01"},
    {"url": "https://example.com/gil", "title": "GIL Removal Status", "publisher": "Real Python", "date": "2024-09-15"},
    {"url": "https://example.com/perf", "title": "Performance Benchmarks", "publisher": "PyPerf", "date": "2024-11-01"},
]


def test_build_citation_index_creates_sequential_numbers():
    """Test that citation_index assigns sequential numbers to allocated cards."""
    allocated = {
        "subq_1": ["researcher_1_0", "researcher_1_1"],
        "subq_2": ["researcher_2_0"],
    }
    index = _build_citation_index(allocated, SAMPLE_CARDS, SAMPLE_SOURCES)
    assert len(index) == 3
    assert index["researcher_1_0"]["number"] == 1
    assert index["researcher_1_1"]["number"] == 2
    assert index["researcher_2_0"]["number"] == 3
    assert index["researcher_1_0"]["title"] == "Python 3.12 Release"
    assert index["researcher_1_0"]["url"] == "https://example.com/python"


def test_build_citation_index_no_duplicates():
    """Test that same card ID appearing in multiple buckets is only indexed once."""
    allocated = {
        "subq_1": ["researcher_1_0"],
        "main": ["researcher_1_0"],
    }
    index = _build_citation_index(allocated, SAMPLE_CARDS, SAMPLE_SOURCES)
    assert len(index) == 1


def test_build_citation_index_empty_allocation():
    """Test empty allocation returns empty index."""
    index = _build_citation_index({}, [], [])
    assert index == {}


def test_format_evidence_cards_uses_index():
    """Test that _format_evidence_cards uses citation_index numbers."""
    citation_index = {
        "researcher_1_0": {"number": 1, "title": "Python 3.12 Release", "url": "https://example.com/python"},
    }
    result = _format_evidence_cards(SAMPLE_CARDS[:1], citation_index)
    assert "[1]" in result
    assert "researcher_1_0" not in result  # raw ID must not appear


def test_format_evidence_cards_fallback_without_index():
    """Test that _format_evidence_cards works without citation_index."""
    result = _format_evidence_cards(SAMPLE_CARDS[:1])
    assert "researcher_1_0" in result  # fallback: raw ID


@pytest.mark.asyncio
async def test_compile_report_strips_et_al_fabrications():
    """Integration test: compile_report strips (Author et al., YYYY) fabrications.

    Exercises the et al. strip, safety-net regex, bibliography assembly,
    and the import re fix in a single call.
    """
    state = {
        "report_outline": {
            "profile": "deep_research_report",
            "sections": [],
            "citation_index": {
                "researcher_0_0": {"number": 1, "title": "CPython Source", "url": "https://example.com"},
            },
            "subquestions": [],
        },
        "written_sections": [
            {
                "section_title": "Findings",
                "content": (
                    "Python 3.12 improves performance significantly. "
                    "(Smith et al., 2024) reported a 15% speedup. "
                    "However concurrency remains challenging [1]. "
                    "Another view [researcher_0_0] supports this."
                ),
            }
        ],
        "sources": [],
    }
    from langchain_core.runnables import RunnableConfig
    result = await compile_report(state, RunnableConfig())
    final = result["final_report"]
    assert "(Smith et al., 2024)" not in final
    assert "[1]" in final
    assert "[researcher_0_0]" not in final


def test_compile_report_safety_net_replaces_raw_ids():
    """Test that safety-net replaces raw card IDs with [N]."""
    citation_index = {
        "researcher_1_0": {"number": 1, "title": "Src", "url": ""},
        "researcher_1_1": {"number": 2, "title": "Src 2", "url": ""},
    }
    text = "Evidence shows improvement [researcher_1_0] and also [researcher_1_1]."
    result = _replace_raw_ids_in_text(text, citation_index)
    assert "[1]" in result
    assert "[2]" in result
    assert "[researcher_1_0]" not in result
    assert "[researcher_1_1]" not in result


def test_replace_raw_ids_ignores_already_correct_refs():
    """Test that already-correct [N] refs are not touched."""
    citation_index = {
        "researcher_1_0": {"number": 1, "title": "Src", "url": ""},
    }
    text = "Good [1] and bad [researcher_1_0]."
    result = _replace_raw_ids_in_text(text, citation_index)
    assert result == "Good [1] and bad [1]."


def test_bibliography_from_citation_index():
    """Test that bibliography is built from citation_index ordering."""
    citation_index = {
        "id_1": {"number": 1, "title": "First Source", "url": "https://example.com/1"},
        "id_2": {"number": 2, "title": "Second Source", "url": "https://example.com/2"},
    }
    bib = _format_bibliography_from_index(citation_index)
    assert "[1]" in bib
    assert "First Source" in bib
    assert "[2]" in bib
    assert bib.index("[1]") < bib.index("[2]")


def test_bibliography_from_index_skips_missing_number():
    """Test that entries without number are skipped."""
    citation_index = {
        "id_1": {"number": 1, "title": "Has Number", "url": "https://example.com/1"},
        "id_2": {"title": "No Number", "url": "https://example.com/2"},
    }
    bib = _format_bibliography_from_index(citation_index)
    assert "Has Number" in bib
    assert "No Number" not in bib
