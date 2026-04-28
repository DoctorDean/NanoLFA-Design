"""MD-derived scoring metrics for integration with the composite scorer.

Extracts binding validation metrics from MD trajectories and provides
them in a format compatible with the existing CompositeScorer. Can
optionally update the composite score with MD-informed adjustments.

MD scoring sits alongside the static AF-based scoring — it doesn't
replace it, but provides an independent validation signal that catches
false positives from static structure prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from nanolfa.models.md_validation import MDTrajectoryMetrics

logger = logging.getLogger(__name__)


@dataclass
class MDScoreAdjustment:
    """Adjustment to composite score based on MD validation.

    Applied as a multiplicative modifier to the static composite score:
        adjusted_score = composite_score * md_modifier

    md_modifier ranges from 0.0 (MD completely invalidates) to 1.2
    (MD confirms and boosts confidence).
    """

    candidate_id: str
    md_validation_score: float    # raw MD score [0, 1]
    md_modifier: float            # multiplicative adjustment
    static_score: float           # original composite score
    adjusted_score: float         # static_score * md_modifier
    confidence_level: str         # "high", "medium", "low"
    recommendation: str           # "promote", "keep", "demote", "reject"


def compute_md_adjustment(
    md_metrics: MDTrajectoryMetrics,
    static_composite_score: float,
) -> MDScoreAdjustment:
    """Compute the scoring adjustment from MD validation results.

    Strategy:
    - MD score > 0.7 → boost (modifier 1.0–1.2): MD confirms binding
    - MD score 0.5–0.7 → neutral (modifier 0.9–1.0): inconclusive
    - MD score 0.3–0.5 → penalize (modifier 0.6–0.9): concerning
    - MD score < 0.3 → reject (modifier 0.0–0.6): binding unstable
    - Ligand escaped → reject (modifier 0.0)

    Args:
        md_metrics: Results from MD validation.
        static_composite_score: Original composite score from Phase 3.

    Returns:
        MDScoreAdjustment with modified score and recommendation.
    """
    md_score = md_metrics.md_validation_score

    if md_metrics.ligand_escaped:
        modifier = 0.0
        confidence = "high"
        recommendation = "reject"
    elif md_score >= 0.7:
        # Strong MD confirmation — boost proportionally
        modifier = 1.0 + 0.2 * (md_score - 0.7) / 0.3
        confidence = "high"
        recommendation = "promote"
    elif md_score >= 0.5:
        # Neutral — slight adjustment based on direction
        modifier = 0.9 + 0.1 * (md_score - 0.5) / 0.2
        confidence = "medium"
        recommendation = "keep"
    elif md_score >= 0.3:
        # Concerning — penalize
        modifier = 0.6 + 0.3 * (md_score - 0.3) / 0.2
        confidence = "medium"
        recommendation = "demote"
    else:
        # Poor MD score — heavy penalty
        modifier = max(0.0, md_score * 2.0)
        confidence = "high"
        recommendation = "reject"

    # Additional penalty for very high CDR3 flexibility
    if md_metrics.cdr3_rmsf > 3.0:
        modifier *= 0.9
        logger.debug(
            "%s: CDR3 RMSF=%.1f Å — applying flexibility penalty",
            md_metrics.candidate_id, md_metrics.cdr3_rmsf,
        )

    # Bonus for exceptionally stable contacts
    if md_metrics.contact_persistence > 0.95:
        modifier *= 1.05

    modifier = float(np.clip(modifier, 0.0, 1.2))
    adjusted = static_composite_score * modifier

    return MDScoreAdjustment(
        candidate_id=md_metrics.candidate_id,
        md_validation_score=md_score,
        md_modifier=modifier,
        static_score=static_composite_score,
        adjusted_score=adjusted,
        confidence_level=confidence,
        recommendation=recommendation,
    )


def apply_md_adjustments(
    candidates_scores: dict[str, float],
    md_results: list[MDTrajectoryMetrics],
) -> list[MDScoreAdjustment]:
    """Apply MD adjustments to a batch of candidates.

    Args:
        candidates_scores: {candidate_id: static_composite_score}.
        md_results: MD validation results for each candidate.

    Returns:
        List of MDScoreAdjustment, sorted by adjusted score descending.
    """
    adjustments: list[MDScoreAdjustment] = []

    for md in md_results:
        static_score = candidates_scores.get(md.candidate_id, 0.0)
        adj = compute_md_adjustment(md, static_score)
        adjustments.append(adj)

    adjustments.sort(key=lambda a: a.adjusted_score, reverse=True)

    # Summary stats
    promoted = sum(1 for a in adjustments if a.recommendation == "promote")
    rejected = sum(1 for a in adjustments if a.recommendation == "reject")
    demoted = sum(1 for a in adjustments if a.recommendation == "demote")

    logger.info(
        "MD score adjustments: %d promoted, %d kept, %d demoted, %d rejected",
        promoted,
        len(adjustments) - promoted - rejected - demoted,
        demoted,
        rejected,
    )

    return adjustments
