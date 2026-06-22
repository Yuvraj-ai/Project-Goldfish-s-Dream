"""Tests for the evaluation harness."""
import pytest

from open_deep_research.eval_harness import EvalHarness, EvalRubricScore


@pytest.fixture
def harness():
    return EvalHarness("tests/golden_set/seed_queries.json")


class TestEvalHarness:
    """Test evaluation harness functionality."""

    def test_load_golden_set(self, harness):
        assert len(harness.golden_set) == 10

    def test_golden_set_has_required_fields(self, harness):
        for query in harness.golden_set:
            assert "id" in query
            assert "mode" in query
            assert "query" in query
            assert "reference_answer" in query

    def test_score_calculation(self, harness):
        scores = EvalRubricScore(
            completeness=8.0,
            citation_accuracy=7.0,
            source_diversity=6.0,
            structure=9.0,
            temporal_relevance=7.5
        )
        weighted = harness.score(scores)
        # (8*0.30) + (7*0.25) + (6*0.15) + (9*0.15) + (7.5*0.15)
        # = 2.4 + 1.75 + 0.9 + 1.35 + 1.125 = 7.525
        assert abs(weighted - 7.53) < 0.01

    def test_evaluate_single(self, harness):
        scores = EvalRubricScore(
            completeness=8.0, citation_accuracy=7.0,
            source_diversity=6.0, structure=9.0,
            temporal_relevance=7.5
        )
        result = harness.evaluate_single(
            query_id="comp_001",
            generated_answer="Test answer",
            scores=scores
        )
        assert result.query_id == "comp_001"
        assert "Compare React" in result.query
        assert result.weighted_average > 0

    def test_regression_check_pass(self, harness):
        current = {"avg_score": 7.4, "min_score": 6.0, "max_score": 9.0, "num_queries": 10, "results": []}
        result = harness.check_regression(current, baseline_avg=7.5, threshold=0.3, is_blocking=True)
        assert result["passed"] is True
        assert result["mode"] == "blocking"

    def test_regression_check_fail(self, harness):
        current = {"avg_score": 7.0, "min_score": 5.0, "max_score": 8.5, "num_queries": 10, "results": []}
        result = harness.check_regression(current, baseline_avg=7.5, threshold=0.3, is_blocking=True)
        assert result["passed"] is False

    def test_regression_check_advisory(self, harness):
        current = {"avg_score": 7.0, "min_score": 5.0, "max_score": 8.5, "num_queries": 10, "results": []}
        result = harness.check_regression(current, baseline_avg=7.5, threshold=0.3, is_blocking=False)
        assert result["mode"] == "advisory"

    def test_evaluate_batch(self, harness):
        batch = [
            {
                "query_id": "comp_001",
                "generated_answer": "Answer 1",
                "scores": {"completeness": 8.0, "citation_accuracy": 7.0, "source_diversity": 6.0, "structure": 9.0, "temporal_relevance": 7.5}
            },
            {
                "query_id": "comp_002",
                "generated_answer": "Answer 2",
                "scores": {"completeness": 7.0, "citation_accuracy": 8.0, "source_diversity": 7.0, "structure": 8.0, "temporal_relevance": 8.0}
            }
        ]
        summary = harness.evaluate_batch(batch)
        assert summary["num_queries"] == 2
        assert summary["avg_score"] > 0
        assert len(summary["results"]) == 2

    def test_evaluate_batch_empty(self, harness):
        summary = harness.evaluate_batch([])
        assert summary["avg_score"] == 0.0
        assert summary["min_score"] == 0.0
        assert summary["max_score"] == 0.0
        assert summary["results"] == []

    def test_load_golden_set_missing_file(self):
        with pytest.raises(FileNotFoundError):
            EvalHarness("tests/golden_set/nonexistent.json")
