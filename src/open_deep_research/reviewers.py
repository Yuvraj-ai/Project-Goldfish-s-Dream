"""Reviewer agents for quality assurance — coverage, evidence, contradiction, style."""

import logging
import re
from typing import List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
        logger.info(
            "CoverageReviewer.review start: report_len=%d subquestions=%d",
            len(report),
            len(subquestions),
        )
        if not subquestions:
            logger.warning("CoverageReviewer: no subquestions in plan — defaulting score to 1.0")
            return ReviewFeedback(reviewer_name="CoverageReviewer", score=1.0)

        covered = 0
        issues = []
        for sq in subquestions:
            key_terms = [w.lower() for w in sq.split() if len(w) > 4]
            report_lower = report.lower()
            if any(term in report_lower for term in key_terms):
                covered += 1
                logger.debug("CoverageReviewer: subquestion covered: %.80s", sq)
            else:
                logger.debug("CoverageReviewer: subquestion NOT covered: %.80s", sq)
                issues.append(f"Subquestion not addressed: {sq[:80]}")

        score = covered / len(subquestions)
        logger.info(
            "CoverageReviewer.review done: score=%.2f covered=%d/%d issues=%d",
            score,
            covered,
            len(subquestions),
            len(issues),
        )

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
        logger.info(
            "EvidenceReviewer.review start: report_len=%d evidence_cards=%d citation_checks=%d",
            len(report),
            len(evidence_cards),
            len(citation_checks),
        )
        sentences = re.split(r"[.!?]+", report)
        uncited_claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 50 and not self._CITATION_PATTERN.search(sentence):
                uncited_claims.append(sentence[:80])

        score = max(0.0, 1.0 - (len(uncited_claims) * 0.1))
        logger.debug(
            "EvidenceReviewer: sentences=%d uncited_claims=%d raw_score=%.2f",
            len(sentences),
            len(uncited_claims),
            score,
        )
        if uncited_claims:
            logger.warning("EvidenceReviewer: %d uncited claim(s) detected", len(uncited_claims))
        logger.info("EvidenceReviewer.review done: score=%.2f", min(score, 1.0))

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
        logger.info("ContradictionReviewer.review start: report_len=%d", len(report))
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
                    logger.debug("ContradictionReviewer: unacknowledged contradiction at line %d: %.80s", i, line)
                    contradictions.append(line[:80])

        score = max(0.0, 1.0 - (len(contradictions) * 0.15))
        if contradictions:
            logger.warning(
                "ContradictionReviewer: %d unacknowledged contradiction(s) flagged", len(contradictions)
            )
        logger.info(
            "ContradictionReviewer.review done: score=%.2f contradictions=%d",
            min(score, 1.0),
            len(contradictions),
        )

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
        word_count = len(report.split())
        logger.info(
            "StyleReviewer.review start: word_count=%d profile=%s tone=%s max_length=%d",
            word_count,
            getattr(profile, "name", "?"),
            getattr(profile, "tone", "?"),
            getattr(profile, "max_length", -1),
        )
        issues = []

        if word_count > profile.max_length * 1.2:
            logger.debug("StyleReviewer: report exceeds max length (%d > %d)", word_count, profile.max_length)
            issues.append(f"Report too long: {word_count} words (max: {profile.max_length})")
        elif word_count < profile.max_length * 0.3:
            logger.debug("StyleReviewer: report under min length (%d < %d)", word_count, int(profile.max_length * 0.3))
            issues.append(f"Report too short: {word_count} words (min: ~{int(profile.max_length * 0.3)})")

        if profile.tone == "formal":
            informal_words = ["gonna", "wanna", "yeah", "cool", "awesome"]
            report_lower = report.lower()
            found_informal = [w for w in informal_words if w in report_lower]
            if found_informal:
                logger.debug("StyleReviewer: informal language count=%d", len(found_informal))
                issues.append(f"Informal language detected: {found_informal}")

        score = max(0.0, 1.0 - (len(issues) * 0.2))
        if issues:
            logger.warning("StyleReviewer: %d style issue(s) detected", len(issues))
        logger.info("StyleReviewer.review done: score=%.2f issues=%d", min(score, 1.0), len(issues))

        return ReviewFeedback(
            reviewer_name="StyleReviewer",
            score=min(score, 1.0),
            issues=issues,
            rewrite_instructions=["Adjust tone and length to match profile"],
        )


class CompletenessReviewer:
    """Check report for placeholder text and incomplete sections."""

    PLACEHOLDER_PATTERNS = re.compile(
        r"\(source required\)|\[citation needed\]|\[insert source\]|"
        r"\(TODO\)|\[TODO\]|<TODO>|\(insert .+?\)",
        re.IGNORECASE,
    )

    async def review(self, report: str) -> ReviewFeedback:
        logger.info("CompletenessReviewer.review start: report_len=%d", len(report))
        matches = []
        for match in self.PLACEHOLDER_PATTERNS.finditer(report):
            logger.debug("CompletenessReviewer: placeholder match: %.80s", match.group())
            matches.append(match.group())
        score = max(0.0, 1.0 - (len(matches) * 0.3))
        issues = [f"Placeholder found: {m}" for m in matches[:10]]
        rewrite_instructions = (
            ["Replace all placeholder text with actual content before finalizing"]
            if matches
            else []
        )
        if matches:
            logger.warning("CompletenessReviewer: %d placeholder(s) found in report", len(matches))
        logger.info("CompletenessReviewer.review done: score=%.2f placeholders=%d", score, len(matches))
        return ReviewFeedback(
            reviewer_name="CompletenessReviewer",
            score=score,
            issues=issues,
            rewrite_instructions=rewrite_instructions,
        )
