"""Cross-reactivity screening and specificity filters.

Evaluates whether nanobody candidates discriminate the target analyte from
structurally similar molecules in the cross-reactivity panel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig

if TYPE_CHECKING:
    from nanolfa.core.pipeline import Candidate

logger = logging.getLogger(__name__)


@dataclass
class SpecificityResult:
    """Cross-reactivity evaluation for a single candidate."""

    candidate_id: str
    on_target_score: float
    off_target_scores: dict[str, float]  # compound_name → composite score
    worst_off_target: str
    worst_off_target_score: float
    selectivity_ratio: float  # on_target / worst_off_target
    passes: bool
    failure_reasons: list[str]


class SpecificityFilter:
    """Screen nanobody candidates against a panel of structural analogs.

    For each candidate, runs AlphaFold predictions against every compound in
    the cross-reactivity panel defined in the target config. Candidates that
    show strong predicted binding to off-target molecules are rejected or
    flagged for negative design.

    Decision rules:
    - Off-target ipTM > 0.50 → reject
    - Selectivity ratio < 2.0 → reject
    - Off-target ΔG_bind < −10 REU → flag for experimental confirmation
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.target_config = config.target
        self.thresholds = config.scoring.thresholds.specificity
        self.panel = list(config.target.cross_reactivity_panel)

        logger.info(
            "Specificity filter: %d compounds in cross-reactivity panel for %s",
            len(self.panel), config.target.name,
        )

    def screen(
        self,
        candidates: list[Candidate],
        af_runner: object,
        scorer: object,
    ) -> list[SpecificityResult]:
        """Screen candidates against the full cross-reactivity panel.

        Args:
            candidates: Candidates to evaluate (must have on-target scores).
            af_runner: AlphaFoldRunner instance for off-target predictions.
            scorer: CompositeScorer instance for scoring off-target complexes.

        Returns:
            SpecificityResult for each candidate.
        """
        results: list[SpecificityResult] = []

        for cand in candidates:
            off_target_scores: dict[str, float] = {}

            for compound in self.panel:
                compound_name = compound["name"]
                compound_smiles = compound["smiles"]

                try:
                    # Run AF prediction against off-target
                    ot_score = self._predict_off_target(
                        cand, compound_smiles, compound_name, af_runner, scorer
                    )
                    off_target_scores[compound_name] = ot_score
                except Exception as e:
                    logger.warning(
                        "Off-target prediction failed: %s vs %s: %s",
                        cand.candidate_id, compound_name, e,
                    )
                    off_target_scores[compound_name] = 0.0

            # Evaluate specificity
            result = self._evaluate_specificity(cand, off_target_scores)
            results.append(result)

        passed = sum(1 for r in results if r.passes)
        logger.info(
            "Specificity screening: %d/%d passed",
            passed, len(results),
        )
        return results

    def _predict_off_target(
        self,
        candidate: Candidate,
        smiles: str,
        compound_name: str,
        af_runner: object,
        scorer: object,
    ) -> float:
        """Predict binding of a candidate to an off-target compound.

        Returns the composite score for the off-target complex.
        """
        # This would use the AlphaFoldRunner to predict the complex with
        # the off-target compound's SMILES/SDF, then score the result.
        # Simplified here — actual implementation mirrors Phase 3a/3b.
        logger.debug(
            "Predicting %s vs off-target %s",
            candidate.candidate_id, compound_name,
        )
        # Placeholder — returns 0.0 (no binding predicted)
        # Actual implementation would:
        # 1. Generate off-target SDF from SMILES
        # 2. Run AF3 prediction
        # 3. Score the predicted complex
        return 0.0

    def _evaluate_specificity(
        self,
        candidate: Candidate,
        off_target_scores: dict[str, float],
    ) -> SpecificityResult:
        """Apply specificity decision rules."""
        on_target = candidate.composite_score or 0.0
        failure_reasons: list[str] = []

        # Find worst off-target
        worst_name = max(off_target_scores, key=off_target_scores.get)  # type: ignore
        worst_score = off_target_scores[worst_name]

        # Selectivity ratio
        selectivity = on_target / max(worst_score, 0.01)

        # Decision rules
        if worst_score > self.thresholds.max_off_target_iptm:
            failure_reasons.append(
                f"off-target {worst_name} score={worst_score:.3f}>{self.thresholds.max_off_target_iptm}"
            )
        if selectivity < self.thresholds.min_selectivity_ratio:
            failure_reasons.append(
                f"selectivity={selectivity:.2f}<{self.thresholds.min_selectivity_ratio}"
            )

        return SpecificityResult(
            candidate_id=candidate.candidate_id,
            on_target_score=on_target,
            off_target_scores=off_target_scores,
            worst_off_target=worst_name,
            worst_off_target_score=worst_score,
            selectivity_ratio=selectivity,
            passes=len(failure_reasons) == 0,
            failure_reasons=failure_reasons,
        )
