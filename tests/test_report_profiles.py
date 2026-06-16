"""Tests for report profile registry."""


from open_deep_research.report_profiles import (
    MODE_TO_PROFILE,
    SectionDefinition,
    get_profile,
    list_profiles,
)


def test_all_builtin_profiles_registered():
    """Test that all 8 profiles are registered."""
    profiles = list_profiles()
    assert len(profiles) == 8
    assert "executive_brief" in profiles
    assert "deep_research_report" in profiles
    assert "academic_literature_review" in profiles
    assert "investment_memo" in profiles
    assert "competitive_landscape" in profiles
    assert "technical_design_research" in profiles
    assert "policy_memo" in profiles
    assert "news_brief" in profiles


def test_get_profile_returns_correct_profile():
    """Test that get_profile returns the correct profile."""
    profile = get_profile("executive_brief")
    assert profile is not None
    assert profile.name == "executive_brief"
    assert profile.tone == "formal"
    assert profile.max_length == 3000


def test_get_profile_returns_none_for_unknown():
    """Test that get_profile returns None for unknown profile."""
    profile = get_profile("nonexistent")
    assert profile is None


def test_profile_has_required_sections():
    """Test that profiles have required sections."""
    profile = get_profile("deep_research_report")
    assert len(profile.required_sections) >= 5
    assert all(s.required for s in profile.required_sections)


def test_profile_section_definitions():
    """Test section definition structure."""
    section = SectionDefinition(name="Test", description="Test section")
    assert section.name == "Test"
    assert section.required is True
    assert section.evidence_allocation == "auto"


def test_mode_to_profile_mapping():
    """Test that all research modes have profile mappings."""
    assert "comparison" in MODE_TO_PROFILE
    assert "market_landscape" in MODE_TO_PROFILE
    assert "academic_literature_review" in MODE_TO_PROFILE
    assert "company_due_diligence" in MODE_TO_PROFILE
    assert "technical_implementation" in MODE_TO_PROFILE
    assert "news_or_current_events" in MODE_TO_PROFILE
    assert "policy_legal_regulatory" in MODE_TO_PROFILE
    assert "validation_or_fact_check" in MODE_TO_PROFILE
    assert "custom" in MODE_TO_PROFILE


def test_executive_brief_no_toc():
    """Test that executive brief disables TOC."""
    profile = get_profile("executive_brief")
    assert profile.include_table_of_contents is False


def test_deep_research_report_has_appendix():
    """Test that deep research report enables appendix."""
    profile = get_profile("deep_research_report")
    assert profile.include_appendix is True
    assert profile.include_source_quality_notes is True
