"""Golden-set evaluation pipeline for regression testing."""
import json
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field


class EvalRubricScore(BaseModel):
    """Score for a single rubric dimension."""
    completeness: float = Field(ge=0.0, le=10.0)
    citation_accuracy: float = Field(ge=0.0, le=10.0)
    source_diversity: float = Field(ge=0.0, le=10.0)
    structure: float = Field(ge=0.0, le=10.0)
    temporal_relevance: float = Field(ge=0.0, le=10.0)


class EvalResult(BaseModel):
    """Result of evaluating a single query."""
    query_id: str
    query: str
    generated_answer: str
    scores: EvalRubricScore
    weighted_average: float = 0.0
    notes: str = ""


class EvalHarness:
    """Evaluation harness for golden-set regression testing."""

    WEIGHTS = {
        "completeness": 0.30,
        "citation_accuracy": 0.25,
        "source_diversity": 0.15,
        "structure": 0.15,
        "temporal_relevance": 0.15,
    }

    def __init__(self, golden_set_path: str = "tests/golden_set/seed_queries.json"):
        self.golden_set_path = golden_set_path
        self.golden_set = self._load_golden_set()

    def _load_golden_set(self) -> List[Dict]:
        """Load golden set queries from JSON file."""
        path = Path(self.golden_set_path)
        if not path.exists():
            raise FileNotFoundError(f"Golden set not found: {path}")
        with open(path) as f:
            return json.load(f)

    def score(self, scores: EvalRubricScore) -> float:
        """Calculate weighted average from rubric scores."""
        total = 0.0
        for dim, weight in self.WEIGHTS.items():
            total += getattr(scores, dim) * weight
        return round(total, 2)

    def evaluate_single(
        self,
        query_id: str,
        generated_answer: str,
        scores: EvalRubricScore,
        notes: str = ""
    ) -> EvalResult:
        """Evaluate a single query result."""
        query = next(
            (q["query"] for q in self.golden_set if q["id"] == query_id),
            "Unknown query"
        )
        weighted = self.score(scores)
        return EvalResult(
            query_id=query_id,
            query=query,
            generated_answer=generated_answer,
            scores=scores,
            weighted_average=weighted,
            notes=notes
        )

    def evaluate_batch(
        self,
        results: List[Dict]  # [{query_id, generated_answer, scores}]
    ) -> Dict:
        """Evaluate a batch of results and return summary."""
        eval_results = []
        for r in results:
            result = self.evaluate_single(
                query_id=r["query_id"],
                generated_answer=r["generated_answer"],
                scores=EvalRubricScore(**r["scores"]),
                notes=r.get("notes", "")
            )
            eval_results.append(result)

        if not eval_results:
            return {"avg_score": 0.0, "min_score": 0.0, "max_score": 0.0, "results": []}

        scores = [r.weighted_average for r in eval_results]
        return {
            "avg_score": round(sum(scores) / len(scores), 2),
            "min_score": min(scores),
            "max_score": max(scores),
            "num_queries": len(eval_results),
            "results": [r.model_dump() for r in eval_results]
        }

    def check_regression(
        self,
        current_results: Dict,
        baseline_avg: float,
        threshold: float = 0.3,
        is_blocking: bool = False
    ) -> Dict:
        """Check if current results regress from baseline.

        Advisory mode (default): logs warning but doesn't block.
        Blocking mode: returns passed=False for CI gate.
        """
        current_avg = current_results["avg_score"]
        regression = baseline_avg - current_avg
        passed = regression <= threshold
        mode = "blocking" if is_blocking else "advisory"
        return {
            "passed": passed,
            "mode": mode,
            "baseline_avg": baseline_avg,
            "current_avg": current_avg,
            "regression": round(regression, 2),
            "threshold": threshold,
            "message": f"[{mode.upper()}] {'PASS' if passed else f'FAIL: regression of {regression} exceeds threshold {threshold}'}"
        }
