"""Reviewer agents for quality assurance — coverage, evidence, contradiction, style."""

import re
from typing import List

from pydantic import BaseModel, Field


class ReviewFeedback(BaseModel):
    """Structured feedback from a reviewer agent."""

    reviewer_name: str
    score: float = Field(ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    rewrite_instructions: List[str] = Field(default_factory=list)


class CoverageReviewer:
    """Check if report covers all research subquestions."""

    async def review(self, report: str, plan: dict) -> ReviewFeedback:
        """Review report coverage of research subquestions."""
        subquestions = plan.get("subquestions", [])
        if not subquestions:
            return ReviewFeedback(reviewer_name="CoverageReviewer", score=1.0)

        covered = 0
        issues = []
        for sq in subquestions:
            key_terms = [w.lower() for w in sq.split() if len(w) > 4]
            report_lower = report.lower()
            if any(term in report_lower for term in key_terms):
                covered += 1
            else:
                issues.append(f"Subquestion not addressed: {sq[:80]}")

        score = covered / len(subquestions)

        return ReviewFeedback(
            reviewer_name="CoverageReviewer",
            score=score,
            issues=issues,
            rewrite_instructions=[f"Address missing subquestion: {sq[:80]}" for sq in issues],
        )


class EvidenceReviewer:
    """Check if claims are grounded in evidence."""

    _CITATION_PATTERN = re.compile(
        r"\[\d+\]"
        r"|\([\w\s]+,\s*\d{4}\)"
        r"|\([\w\s]+,\s*n\.d\.\)"
    )

    async def review(self, report: str, evidence_cards: list, citation_checks: list) -> ReviewFeedback:
        """Review report for uncited claims."""
        sentences = re.split(r"[.!?]+", report)
        uncited_claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 50 and not self._CITATION_PATTERN.search(sentence):
                uncited_claims.append(sentence[:80])

        score = max(0.0, 1.0 - (len(uncited_claims) * 0.1))

        return ReviewFeedback(
            reviewer_name="EvidenceReviewer",
            score=min(score, 1.0),
            issues=[f"Uncited claim: {c}" for c in uncited_claims[:5]],
            rewrite_instructions=["Add citations to unsupported claims"],
        )


class ContradictionReviewer:
    """Check for unacknowledged contradictions."""

    async def review(self, report: str) -> ReviewFeedback:
        """Review report for unacknowledged contradictions."""
        contradictions = []
        lines = report.split("\n")

        for i, line in enumerate(lines):
            lower = line.lower()
            if any(phrase in lower for phrase in ["however", "but", "on the other hand", "contradicts"]):
                context = lines[max(0, i - 2) : i + 3]
                if not any(
                    ack in " ".join(context).lower()
                    for ack in ["debate", "disagreement", "tension", "conflict", "divergent"]
                ):
                    contradictions.append(line[:80])

        score = max(0.0, 1.0 - (len(contradictions) * 0.15))

        return ReviewFeedback(
            reviewer_name="ContradictionReviewer",
            score=min(score, 1.0),
            issues=[f"Unacknowledged contradiction: {c}" for c in contradictions[:5]],
            rewrite_instructions=["Explicitly acknowledge source disagreements"],
        )


class StyleReviewer:
    """Check if report matches profile style requirements."""

    async def review(self, report: str, profile) -> ReviewFeedback:
        """Review report style against profile requirements."""
        issues = []
        word_count = len(report.split())

        if word_count > profile.max_length * 1.2:
            issues.append(f"Report too long: {word_count} words (max: {profile.max_length})")
        elif word_count < profile.max_length * 0.3:
            issues.append(f"Report too short: {word_count} words (min: ~{int(profile.max_length * 0.3)})")

        if profile.tone == "formal":
            informal_words = ["gonna", "wanna", "yeah", "cool", "awesome"]
            report_lower = report.lower()
            found_informal = [w for w in informal_words if w in report_lower]
            if found_informal:
                issues.append(f"Informal language detected: {found_informal}")

        score = max(0.0, 1.0 - (len(issues) * 0.2))

        return ReviewFeedback(
            reviewer_name="StyleReviewer",
            score=min(score, 1.0),
            issues=issues,
            rewrite_instructions=["Adjust tone and length to match profile"],
        )
