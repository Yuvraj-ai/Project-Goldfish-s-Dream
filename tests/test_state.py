# tests/test_state.py
"""Tests for structured state models and reducers."""
import pytest
from pydantic import ValidationError

from open_deep_research.state import (
    AgentState,
    CitationCheck,
    ConflictFlag,
    EvidenceCard,
    ResearcherState,
    ResearchPlan,
    ReviewResult,
    Source,
    SupervisorState,
    append_evidence,
    merge_sources,
    override_reducer,
)

# ──────────────────────────────────────────────
# Source Model
# ──────────────────────────────────────────────

class TestSourceModel:
    """Test Source model validation."""

    def test_source_creation_minimal(self):
        src = Source(url="https://example.com")
        assert src.url == "https://example.com"
        assert src.credibility_score == 0.5
        assert src.source_type == "web"
        assert src.raw_excerpts == []

    def test_source_creation_full(self):
        src = Source(
            url="https://example.com",
            title="Test Article",
            publisher="Test Publisher",
            date="2026-01-01",
            credibility_score=0.8,
            source_type="academic",
            raw_excerpts=["excerpt 1", "excerpt 2"],
            provider="tavily",
            accessed_at="2026-06-13",
        )
        assert src.title == "Test Article"
        assert src.publisher == "Test Publisher"
        assert src.source_type == "academic"
        assert len(src.raw_excerpts) == 2

    def test_source_credibility_upper_bound(self):
        with pytest.raises(ValidationError):
            Source(url="https://example.com", credibility_score=1.5)

    def test_source_credibility_lower_bound(self):
        with pytest.raises(ValidationError):
            Source(url="https://example.com", credibility_score=-0.1)

    def test_source_optional_fields_default(self):
        src = Source(url="https://example.com")
        assert src.date is None
        assert src.publisher == ""
        assert src.accessed_at is None


# ──────────────────────────────────────────────
# EvidenceCard Model
# ──────────────────────────────────────────────

class TestEvidenceCardModel:
    """Test EvidenceCard model validation."""

    def test_evidence_card_creation_minimal(self):
        card = EvidenceCard(claim="Python is a programming language")
        assert card.claim == "Python is a programming language"
        assert card.confidence == 0.0
        assert card.supporting_source_ids == []
        assert card.conflicting_source_ids == []

    def test_evidence_card_creation_full(self):
        card = EvidenceCard(
            id="ec_001",
            claim="Python is a programming language",
            confidence=0.95,
            supporting_source_ids=["src_1", "src_2"],
            conflicting_source_ids=["src_3"],
            exact_excerpts=["Python is a high-level programming language"],
            subquestion_id="sq_001",
            researcher_id="r_001",
            deduplicated_from=["ec_000"],
        )
        assert card.id == "ec_001"
        assert card.confidence == 0.95
        assert len(card.supporting_source_ids) == 2
        assert card.subquestion_id == "sq_001"

    def test_evidence_card_confidence_upper_bound(self):
        with pytest.raises(ValidationError):
            EvidenceCard(claim="test", confidence=1.5)

    def test_evidence_card_confidence_lower_bound(self):
        with pytest.raises(ValidationError):
            EvidenceCard(claim="test", confidence=-0.1)


# ──────────────────────────────────────────────
# ConflictFlag Model
# ──────────────────────────────────────────────

class TestConflictFlagModel:
    """Test ConflictFlag model validation."""

    def test_conflict_flag_creation(self):
        flag = ConflictFlag(
            card_a_id="ec_001",
            card_b_id="ec_002",
            conflict_description="Contradictory claims about Python version",
        )
        assert flag.card_a_id == "ec_001"
        assert flag.severity == "medium"

    def test_conflict_flag_severity_levels(self):
        for severity in ["low", "medium", "high"]:
            flag = ConflictFlag(
                card_a_id="a",
                card_b_id="b",
                conflict_description="test",
                severity=severity,
            )
            assert flag.severity == severity


# ──────────────────────────────────────────────
# ResearchPlan Model
# ──────────────────────────────────────────────

class TestResearchPlanModel:
    """Test ResearchPlan model validation."""

    def test_research_plan_creation(self):
        plan = ResearchPlan(
            objective="Research Python frameworks",
            subquestions=["Which is fastest?", "Which has best ecosystem?"],
            strategy="comparison",
        )
        assert plan.objective == "Research Python frameworks"
        assert len(plan.subquestions) == 2
        assert plan.strategy == "comparison"

    def test_research_plan_defaults(self):
        plan = ResearchPlan(objective="test")
        assert plan.subquestions == []
        assert plan.expected_source_types == []
        assert plan.risks == []


# ──────────────────────────────────────────────
# CitationCheck Model
# ──────────────────────────────────────────────

class TestCitationCheckModel:
    """Test CitationCheck model validation."""

    def test_citation_check_creation(self):
        check = CitationCheck(
            claim="Python is popular",
            url="https://example.com",
            supports_claim=True,
            status="verified",
        )
        assert check.supports_claim is True
        assert check.status == "verified"

    def test_citation_check_status_values(self):
        for status in ["verified", "unverified", "dead", "stale"]:
            check = CitationCheck(claim="test", url="https://a.com", status=status)
            assert check.status == status


# ──────────────────────────────────────────────
# ReviewResult Model
# ──────────────────────────────────────────────

class TestReviewResultModel:
    """Test ReviewResult model validation."""

    def test_review_result_creation(self):
        result = ReviewResult(
            coverage_score=0.8,
            evidence_score=0.9,
            style_score=0.7,
            contradiction_flags=["ec_001 vs ec_002"],
            iteration_count=2,
            feedback="Good coverage",
        )
        assert result.coverage_score == 0.8
        assert result.iteration_count == 2

    def test_review_result_bounds(self):
        with pytest.raises(ValidationError):
            ReviewResult(coverage_score=1.5)
        with pytest.raises(ValidationError):
            ReviewResult(evidence_score=-0.1)


# ──────────────────────────────────────────────
# merge_sources Reducer
# ──────────────────────────────────────────────

class TestMergeSourcesReducer:
    """Test merge_sources reducer deduplication."""

    def test_merge_empty_with_new(self):
        result = merge_sources([], [{"url": "https://a.com", "title": "A"}])
        assert len(result) == 1
        assert result[0]["url"] == "https://a.com"

    def test_merge_dedup_by_url_keeps_higher_credibility(self):
        existing = [{"url": "https://a.com", "title": "A v1", "credibility_score": 0.5}]
        new = [{"url": "https://a.com", "title": "A v2", "credibility_score": 0.8}]
        result = merge_sources(existing, new)
        assert len(result) == 1
        assert result[0]["credibility_score"] == 0.8
        assert result[0]["title"] == "A v2"

    def test_merge_dedup_by_url_keeps_existing_if_higher(self):
        existing = [{"url": "https://a.com", "title": "A v1", "credibility_score": 0.9}]
        new = [{"url": "https://a.com", "title": "A v2", "credibility_score": 0.3}]
        result = merge_sources(existing, new)
        assert len(result) == 1
        assert result[0]["credibility_score"] == 0.9
        assert result[0]["title"] == "A v1"

    def test_merge_keeps_different_urls(self):
        existing = [{"url": "https://a.com"}]
        new = [{"url": "https://b.com"}]
        result = merge_sources(existing, new)
        assert len(result) == 2

    def test_merge_handles_no_url_on_existing(self):
        existing = [{"title": "no url"}]
        new = [{"url": "https://a.com"}]
        result = merge_sources(existing, new)
        assert len(result) == 2

    def test_merge_handles_no_url_on_new(self):
        existing = [{"url": "https://a.com"}]
        new = [{"title": "no url"}]
        result = merge_sources(existing, new)
        assert len(result) == 2

    def test_merge_all_no_url(self):
        existing = [{"title": "a"}]
        new = [{"title": "b"}]
        result = merge_sources(existing, new)
        assert len(result) == 2

    def test_merge_multiple_new_with_dupes(self):
        existing = [{"url": "https://a.com", "credibility_score": 0.5}]
        new = [
            {"url": "https://a.com", "credibility_score": 0.3},
            {"url": "https://b.com", "credibility_score": 0.7},
            {"url": "https://a.com", "credibility_score": 0.9},
        ]
        result = merge_sources(existing, new)
        assert len(result) == 2
        a_src = next(s for s in result if s["url"] == "https://a.com")
        assert a_src["credibility_score"] == 0.9


# ──────────────────────────────────────────────
# append_evidence Reducer
# ──────────────────────────────────────────────

class TestAppendEvidenceReducer:
    """Test append_evidence reducer."""

    def test_append_to_empty(self):
        result = append_evidence([], [{"claim": "test"}])
        assert len(result) == 1
        assert result[0]["claim"] == "test"

    def test_append_to_existing(self):
        existing = [{"claim": "a"}]
        new = [{"claim": "b"}]
        result = append_evidence(existing, new)
        assert len(result) == 2
        assert result[0]["claim"] == "a"
        assert result[1]["claim"] == "b"

    def test_append_preserves_order(self):
        existing = [{"claim": "first"}]
        new = [{"claim": "second"}, {"claim": "third"}]
        result = append_evidence(existing, new)
        assert [r["claim"] for r in result] == ["first", "second", "third"]


# ──────────────────────────────────────────────
# override_reducer
# ──────────────────────────────────────────────

class TestOverrideReducer:
    """Test override_reducer for state overrides."""

    def test_override_with_dict_type(self):
        result = override_reducer("old", {"type": "override", "value": "new"})
        assert result == "new"

    def test_override_with_non_override_dict(self):
        result = override_reducer(["a"], ["b"])
        assert result == ["a", "b"]

    def test_override_with_list(self):
        result = override_reducer([1, 2], [3, 4])
        assert result == [1, 2, 3, 4]


# ──────────────────────────────────────────────
# Backward Compatibility
# ──────────────────────────────────────────────

class TestBackwardCompatibility:
    """Test that existing state patterns still work."""

    def test_agent_state_has_original_fields(self):
        state = AgentState(
            supervisor_messages=[],
            research_brief="test",
            final_report="report",
        )
        assert state["research_brief"] == "test"
        assert state["final_report"] == "report"
        assert state.get("messages", []) == []

    def test_agent_state_has_new_fields_with_defaults(self):
        state = AgentState(
            supervisor_messages=[],
            research_brief="test",
            final_report="report",
        )
        assert state.get("sources", []) == []
        assert state.get("evidence_cards", []) == []
        assert state.get("conflicts", []) == []
        assert state.get("total_tokens", 0) == 0

    def test_supervisor_state_has_original_fields(self):
        state = SupervisorState(
            supervisor_messages=[],
            research_brief="test",
        )
        assert state["research_brief"] == "test"
        assert state.get("research_iterations", 0) == 0

    def test_supervisor_state_has_new_fields(self):
        state = SupervisorState(
            supervisor_messages=[],
            research_brief="test",
        )
        assert state.get("sources", []) == []
        assert state.get("evidence_cards", []) == []

    def test_researcher_state_has_original_fields(self):
        state = ResearcherState(
            researcher_messages=[],
            research_topic="test",
            compressed_research="compressed",
        )
        assert state["research_topic"] == "test"
        assert state["compressed_research"] == "compressed"

    def test_researcher_state_has_new_fields(self):
        state = ResearcherState(
            researcher_messages=[],
            research_topic="test",
            compressed_research="compressed",
        )
        assert state.get("sources", []) == []
        assert state.get("evidence_cards", []) == []
        assert state.get("total_tokens", 0) == 0
