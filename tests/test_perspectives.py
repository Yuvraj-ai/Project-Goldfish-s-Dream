"""Tests for STORM-style perspectives."""
from open_deep_research.perspectives import (
    CoverageMatrix,
    PerspectiveGenerator,
    PerspectiveLens,
)


class TestPerspectiveGenerator:
    def test_market_perspectives(self):
        perspectives = PerspectiveGenerator.generate_perspectives("market_landscape")
        assert len(perspectives) == 5
        assert any(p.name == "customer" for p in perspectives)

    def test_policy_perspectives(self):
        perspectives = PerspectiveGenerator.generate_perspectives("policy_legal_regulatory")
        assert len(perspectives) == 4

    def test_comparison_perspectives(self):
        perspectives = PerspectiveGenerator.generate_perspectives("comparison")
        assert len(perspectives) == 4

    def test_custom_gets_defaults(self):
        perspectives = PerspectiveGenerator.generate_perspectives("custom")
        assert len(perspectives) == 2

    def test_max_perspectives_respected(self):
        perspectives = PerspectiveGenerator.generate_perspectives("market_landscape", max_perspectives=3)
        assert len(perspectives) == 3

    def test_unknown_mode_gets_defaults(self):
        perspectives = PerspectiveGenerator.generate_perspectives("unknown_mode")
        assert len(perspectives) == 2

    def test_perspectives_have_required_fields(self):
        perspectives = PerspectiveGenerator.generate_perspectives("market_landscape")
        for p in perspectives:
            assert p.name
            assert p.description
            assert len(p.focus_areas) > 0


class TestCoverageMatrix:
    def test_find_gaps(self):
        matrix = CoverageMatrix(subquestions=["Q1", "Q2"], perspectives=["P1", "P2"])
        matrix.add_evidence("Q1", "P1", "e1")
        gaps = matrix.find_gaps()
        assert len(gaps) == 3

    def test_no_gaps_when_full(self):
        matrix = CoverageMatrix(subquestions=["Q1"], perspectives=["P1"])
        matrix.add_evidence("Q1", "P1", "e1")
        gaps = matrix.find_gaps()
        assert len(gaps) == 0

    def test_coverage_percentage(self):
        matrix = CoverageMatrix(subquestions=["Q1", "Q2"], perspectives=["P1", "P2"])
        matrix.add_evidence("Q1", "P1", "e1")
        assert matrix.coverage_percentage() == 25.0

    def test_empty_matrix(self):
        matrix = CoverageMatrix()
        gaps = matrix.find_gaps()
        assert len(gaps) == 0
        assert matrix.coverage_percentage() == 100.0

    def test_build_coverage_matrix(self):
        perspectives = [
            PerspectiveLens(name="P1", description="Test"),
            PerspectiveLens(name="P2", description="Test"),
        ]
        matrix = PerspectiveGenerator.build_coverage_matrix(["Q1", "Q2"], perspectives)
        assert matrix.subquestions == ["Q1", "Q2"]
        assert matrix.perspectives == ["P1", "P2"]


class TestPerspectiveLens:
    def test_lens_creation(self):
        lens = PerspectiveLens(name="test", description="A test lens")
        assert lens.name == "test"
        assert lens.focus_areas == []

    def test_lens_with_focus_areas(self):
        lens = PerspectiveLens(name="test", description="Desc", focus_areas=["area1", "area2"])
        assert len(lens.focus_areas) == 2
