"""Evidence extraction, deduplication, and compression engine."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Tuple

from open_deep_research.state import ConflictFlag, EvidenceCard, Source


def compute_source_credibility(source: Source) -> float:
    """Compute source credibility score (0-1)."""
    score = 0.5  # baseline

    # Boost for academic sources
    if source.source_type == "academic":
        score += 0.2

    # Boost for known authoritative publishers
    authoritative_domains = [
        "arxiv.org", "nature.com", "science.org", "ieee.org",
        "acm.org", "gov", ".edu", "who.int", "nih.gov"
    ]
    for domain in authoritative_domains:
        if domain in source.url:
            score += 0.15
            break

    # Penalize for unknown/untrusted sources
    if source.credibility_score < 0.3:
        score -= 0.2

    return max(0.0, min(1.0, score))


def compute_corroboration_strength(
    claim: str,
    all_cards: List[EvidenceCard],
    similarity_threshold: float = 0.7
) -> float:
    """Compute corroboration strength based on supporting sources."""
    supporting_count = len([
        c for c in all_cards
        if c.claim != claim and _claims_similar(c.claim, claim, similarity_threshold)
    ])
    # Normalize: 0 sources = 0, 3+ sources = 1.0
    return min(1.0, supporting_count / 3.0)


def compute_recency_score(
    source_date: str | None,
    mode: str = "default"
) -> float:
    """Compute recency score based on source date and research mode."""
    if not source_date:
        return 0.5  # Unknown date gets neutral score

    try:
        pub_date = datetime.strptime(source_date, "%Y-%m-%d")
    except ValueError:
        return 0.5

    days_ago = (datetime.now() - pub_date).days

    # Mode-specific windows
    windows = {
        "news_or_current_events": 7,    # 1 week
        "technical_implementation": 90,  # 3 months
        "academic_literature_review": 365 * 3,  # 3 years
        "default": 180,  # 6 months
    }

    window = windows.get(mode, windows["default"])

    if days_ago <= window:
        return 1.0
    else:
        # Linear decay after window
        decay_days = days_ago - window
        return max(0.0, 1.0 - (decay_days / (window * 2)))


def _claims_similar(claim_a: str, claim_b: str, threshold: float = 0.7) -> bool:
    """Check if two claims are semantically similar using Jaccard similarity."""
    words_a = set(re.findall(r'\w+', claim_a.lower()))
    words_b = set(re.findall(r'\w+', claim_b.lower()))

    if not words_a or not words_b:
        return False

    intersection = words_a & words_b
    union = words_a | words_b
    jaccard = len(intersection) / len(union)

    return jaccard >= threshold


def extract_evidence(
    raw_results: List[Dict],
    subquestion_id: str = "",
    researcher_id: str = "",
    mode: str = "default"
) -> Tuple[List[EvidenceCard], List[Source]]:
    """Extract evidence cards from raw search results."""
    cards = []
    sources = []

    for i, result in enumerate(raw_results):
        # Create source
        source = Source(
            url=result.get("url", ""),
            title=result.get("title", ""),
            publisher=result.get("publisher", ""),
            date=result.get("date"),
            source_type=result.get("source_type", "web"),
            provider=result.get("provider", ""),
            raw_excerpts=[result.get("content", "")]
        )
        source.credibility_score = compute_source_credibility(source)
        sources.append(source)

        # Create evidence card from snippet
        content = result.get("content", result.get("snippet", ""))
        if content:
            card = EvidenceCard(
                id=f"{researcher_id}_{subquestion_id}_{i}",
                claim=content[:200],  # First 200 chars as claim
                confidence=0.0,  # Will be computed after dedup
                supporting_source_ids=[source.url],
                exact_excerpts=[content[:500]],
                subquestion_id=subquestion_id,
                researcher_id=researcher_id
            )
            cards.append(card)

    # Compute confidence scores
    for card in cards:
        source_cred = 0.5
        for src in sources:
            if src.url in card.supporting_source_ids:
                source_cred = src.credibility_score
                break

        corroboration = compute_corroboration_strength(card.claim, cards)

        # Use first source date for recency
        recency = 0.5
        for src in sources:
            if src.url in card.supporting_source_ids:
                recency = compute_recency_score(src.date, mode)
                break

        card.confidence = round(
            source_cred * 0.4 +
            corroboration * 0.35 +
            recency * 0.25,
            4
        )

    return cards, sources


def deduplicate_claims(
    cards: List[EvidenceCard],
    similarity_threshold: float = 0.7
) -> Tuple[List[EvidenceCard], List[str]]:
    """Deduplicate claims using Jaccard similarity (LLM judge in production)."""
    if not cards:
        return [], []

    unique_cards = [cards[0]]
    duplicate_ids = []

    for card in cards[1:]:
        is_duplicate = False
        for unique in unique_cards:
            if _claims_similar(card.claim, unique.claim, similarity_threshold):
                # Merge: keep the one with higher confidence
                if card.confidence > unique.confidence:
                    duplicate_ids.append(unique.id)
                    unique_cards.remove(unique)
                    unique_cards.append(card)
                else:
                    duplicate_ids.append(card.id)
                is_duplicate = True
                break

        if not is_duplicate:
            unique_cards.append(card)

    return unique_cards, duplicate_ids


def detect_conflicts(
    cards: List[EvidenceCard]
) -> List[ConflictFlag]:
    """Detect conflicting claims between evidence cards."""
    conflicts = []

    for i, card_a in enumerate(cards):
        for card_b in cards[i+1:]:
            if _claims_conflict(card_a.claim, card_b.claim):
                conflicts.append(ConflictFlag(
                    card_a_id=card_a.id,
                    card_b_id=card_b.id,
                    conflict_description=f"Claims may conflict: '{card_a.claim[:50]}...' vs '{card_b.claim[:50]}...'",
                    severity="medium"
                ))

    return conflicts


def _claims_conflict(claim_a: str, claim_b: str) -> bool:
    """Check if two claims contradict each other."""
    # Simple negation detection
    negation_patterns = [
        (r'\bis\b', r'\bis not\b'),
        (r'\bcan\b', r'\bcannot\b'),
        (r'\bwill\b', r'\bwill not\b'),
        (r'\bshould\b', r'\bshould not\b'),
        (r'\bincreases\b', r'\bdecreases\b'),
        (r'\bbetter\b', r'\bworse\b'),
        (r'\bfaster\b', r'\bslower\b'),
    ]

    claim_a_lower = claim_a.lower()
    claim_b_lower = claim_b.lower()

    for pos, neg in negation_patterns:
        if (re.search(pos, claim_a_lower) and re.search(neg, claim_b_lower)):
            # Check if subjects are similar
            words_a = set(re.findall(r'\w+', claim_a_lower))
            words_b = set(re.findall(r'\w+', claim_b_lower))
            if len(words_a & words_b) / max(len(words_a | words_b), 1) > 0.3:
                return True

    return False


def compress_evidence(
    cards: List[EvidenceCard]
) -> List[EvidenceCard]:
    """Compress evidence by merging cards with similar claims."""
    if not cards:
        return []

    # Group by similar claims
    groups: Dict[str, List[EvidenceCard]] = {}
    for card in cards:
        # Find existing group or create new one
        matched = False
        for key in groups:
            if _claims_similar(card.claim, key):
                groups[key].append(card)
                matched = True
                break

        if not matched:
            groups[card.claim] = [card]

    # Merge each group into a single card
    compressed = []
    for claim_key, group in groups.items():
        # Merge all supporting sources
        all_source_ids = []
        all_excerpts = []
        for card in group:
            all_source_ids.extend(card.supporting_source_ids)
            all_excerpts.extend(card.exact_excerpts)

        # Deduplicate sources
        unique_source_ids = list(dict.fromkeys(all_source_ids))
        unique_excerpts = list(dict.fromkeys(all_excerpts))[:5]  # Top 5 excerpts

        # Keep highest confidence card as base
        best_card = max(group, key=lambda c: c.confidence)

        compressed_card = EvidenceCard(
            id=best_card.id,
            claim=best_card.claim,
            confidence=best_card.confidence,
            supporting_source_ids=unique_source_ids,
            conflicting_source_ids=best_card.conflicting_source_ids,
            exact_excerpts=unique_excerpts,
            subquestion_id=best_card.subquestion_id,
            researcher_id=best_card.researcher_id,
            deduplicated_from=[c.id for c in group if c.id != best_card.id]
        )
        compressed.append(compressed_card)

    return compressed
