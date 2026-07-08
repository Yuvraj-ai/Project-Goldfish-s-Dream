"""STORM-style perspective generation and coverage matrix."""
import logging
from typing import Dict, List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PerspectiveLens(BaseModel):
    """A single perspective lens for multi-perspective research."""
    name: str
    description: str
    focus_areas: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)


class CoverageMatrix(BaseModel):
    """Tracks which subquestions are covered by which perspectives."""
    subquestions: List[str] = Field(default_factory=list)
    perspectives: List[str] = Field(default_factory=list)
    coverage: Dict[str, List[str]] = Field(default_factory=dict)

    def add_evidence(self, subquestion: str, perspective: str, evidence_id: str):
        """Record that a perspective has evidence for a subquestion."""
        key = f"{subquestion}::{perspective}"
        if key not in self.coverage:
            self.coverage[key] = []
        self.coverage[key].append(evidence_id)
        logger.debug(
            "add_evidence: key=%.80r evidence_id=%s (%d for key)",
            key, evidence_id, len(self.coverage[key]),
        )

    def find_gaps(self) -> List[dict]:
        """Find subquestion/perspective pairs with no evidence."""
        gaps = []
        for sq in self.subquestions:
            for pers in self.perspectives:
                key = f"{sq}::{pers}"
                if key not in self.coverage or not self.coverage[key]:
                    gaps.append({"subquestion": sq, "perspective": pers})
        logger.info(
            "find_gaps: %d gaps across %d subquestions x %d perspectives",
            len(gaps), len(self.subquestions), len(self.perspectives),
        )
        return gaps

    def coverage_percentage(self) -> float:
        """Calculate what percentage of the matrix is covered."""
        total = len(self.subquestions) * len(self.perspectives)
        if total == 0:
            logger.debug("coverage_percentage: empty matrix, returning 100.0")
            return 100.0
        covered = sum(1 for key in self.coverage if self.coverage[key])
        pct = round((covered / total) * 100, 1)
        logger.debug("coverage_percentage: %d/%d covered = %.1f%%", covered, total, pct)
        return pct


# Pre-defined perspective sets by research mode
_MARKET_PERSPECTIVES = [
    PerspectiveLens(
        name="customer",
        description="Customer needs, pain points, and adoption patterns",
        focus_areas=["customer needs", "adoption", "satisfaction", "pain points"],
    ),
    PerspectiveLens(
        name="competitor",
        description="Competitive landscape and market positioning",
        focus_areas=["competitors", "market share", "positioning", "differentiation"],
    ),
    PerspectiveLens(
        name="investor",
        description="Investment trends, valuations, and financial outlook",
        focus_areas=["funding", "valuation", "ROI", "financial metrics"],
    ),
    PerspectiveLens(
        name="regulator",
        description="Regulatory environment and compliance requirements",
        focus_areas=["regulation", "compliance", "policy", "legal"],
    ),
    PerspectiveLens(
        name="technologist",
        description="Technical capabilities, architecture, and innovation",
        focus_areas=["technology", "architecture", "innovation", "R&D"],
    ),
]

_POLICY_PERSPECTIVES = [
    PerspectiveLens(
        name="proponent",
        description="Advocates and supporters of the policy",
        focus_areas=["benefits", "advantages", "support arguments"],
    ),
    PerspectiveLens(
        name="opponent",
        description="Critics and opponents of the policy",
        focus_areas=["criticisms", "risks", "opposition arguments"],
    ),
    PerspectiveLens(
        name="affected_party",
        description="Stakeholders directly affected by the policy",
        focus_areas=["impact", "compliance burden", "adaptation"],
    ),
    PerspectiveLens(
        name="expert",
        description="Subject matter experts and analysts",
        focus_areas=["analysis", "evidence", "recommendations"],
    ),
]

_COMPARISON_PERSPECTIVES = [
    PerspectiveLens(
        name="proponent_a",
        description="Arguments in favor of option A",
        focus_areas=["strengths", "advantages", "use cases for A"],
    ),
    PerspectiveLens(
        name="proponent_b",
        description="Arguments in favor of option B",
        focus_areas=["strengths", "advantages", "use cases for B"],
    ),
    PerspectiveLens(
        name="neutral_analyst",
        description="Objective comparison and trade-off analysis",
        focus_areas=["trade-offs", "benchmarks", "objective metrics"],
    ),
    PerspectiveLens(
        name="practitioner",
        description="Real-world implementation experience",
        focus_areas=["implementation", "practical considerations", "gotchas"],
    ),
]

_DEFAULT_PERSPECTIVES = [
    PerspectiveLens(
        name="general",
        description="General overview and key facts",
        focus_areas=["overview", "key facts", "background"],
    ),
    PerspectiveLens(
        name="critical",
        description="Critical analysis and limitations",
        focus_areas=["limitations", "risks", "criticisms"],
    ),
]


class PerspectiveGenerator:
    """Generates perspective lenses based on research mode."""

    @staticmethod
    def generate_perspectives(
        research_mode: str,
        max_perspectives: int = 7,
    ) -> List[PerspectiveLens]:
        """Generate perspective lenses for the given research mode."""
        logger.info(
            "generate_perspectives: research_mode=%s max_perspectives=%d",
            research_mode, max_perspectives,
        )
        mode_map = {
            "market_landscape": _MARKET_PERSPECTIVES,
            "policy_legal_regulatory": _POLICY_PERSPECTIVES,
            "comparison": _COMPARISON_PERSPECTIVES,
        }
        if research_mode not in mode_map:
            logger.warning(
                "generate_perspectives: unknown research_mode %s, using default perspectives",
                research_mode,
            )
        perspectives = mode_map.get(research_mode, _DEFAULT_PERSPECTIVES)
        selected = perspectives[:max_perspectives]
        logger.info(
            "generate_perspectives: selected %d/%d perspectives for mode %s",
            len(selected), len(perspectives), research_mode,
        )
        return selected

    @staticmethod
    def build_coverage_matrix(
        subquestions: List[str],
        perspectives: List[PerspectiveLens],
    ) -> CoverageMatrix:
        """Build a coverage matrix for subquestions and perspectives."""
        logger.info(
            "build_coverage_matrix: %d subquestions x %d perspectives",
            len(subquestions), len(perspectives),
        )
        return CoverageMatrix(
            subquestions=subquestions,
            perspectives=[p.name for p in perspectives],
        )
