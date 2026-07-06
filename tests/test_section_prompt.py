"""Tests for section prompt building with premise cross-check."""

from open_deep_research.deep_researcher import _build_section_prompt
from open_deep_research.report_profiles import get_profile


def test_section_prompt_includes_research_topic_for_fact_check():
    """Test that fact_check prompt includes the actual research topic."""
    profile = get_profile("fact_check")
    section = {"title": "Claim", "description": "The assertion being evaluated"}
    topic = "Python 3.12 removed the GIL"
    prompt = _build_section_prompt(section, profile, [], topic)
    assert topic in prompt
    assert "Python 3.12" in prompt


def test_section_prompt_includes_premise_instruction_for_fact_check():
    """Test that fact_check prompt includes premise cross-check with topic."""
    profile = get_profile("fact_check")
    section = {"title": "Claim", "description": "The assertion being evaluated"}
    topic = "Python 3.12 removed the GIL"
    prompt = _build_section_prompt(section, profile, [], topic)
    assert "determine if this claim's premise is true or false" in prompt
    assert "Never validate a false premise" in prompt


def test_section_prompt_no_premise_instruction_for_other_profiles():
    """Test that non-fact_check profiles don't include premise instruction."""
    profile = get_profile("deep_research_report")
    section = {"title": "Findings", "description": "Detailed analysis"}
    prompt = _build_section_prompt(section, profile, [], "any topic")
    assert "determine if this claim's premise is true or false" not in prompt
    assert "Never validate a false premise" not in prompt


def test_build_section_prompt_returns_string():
    """Test that prompt building returns a non-empty string."""
    profile = get_profile("executive_brief")
    section = {"title": "Summary", "description": "Overview"}
    prompt = _build_section_prompt(section, profile, [], "topic")
    assert isinstance(prompt, str)
    assert len(prompt) > 50
    assert "Summary" in prompt
    assert "inline citations" in prompt
