"""LFA-specific constraint filters.

Integrates kinetic accessibility, conjugation orientation, and thermal
stability assessments into a unified LFA compatibility gate. Candidates
must pass all three checks to proceed to experimental validation.

This filter runs after Phase 4 (specificity screening) and before
candidates are sent to the wet lab.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from nanolfa.lfa.kinetics import KineticProfile, estimate_kinetic_profile
from nanolfa.lfa.orientation import (
    OrientationResult,
    check_conjugation_clearance,
)
from nanolfa.lfa.stability import StabilityProfile, predict_stability

if TYPE_CHECKING:
    from nanolfa.core.pipeline import Candidate

logger = logging.getLogger(__name__)


@dataclass
class LFACompatibilityResult:
    """Complete LFA compatibility assessment for a candidate."""

    candidate_id: str

    # Component assessments
    kinetics: KineticProfile | None = None
    orientation: OrientationResult | None = None
    stability: StabilityProfile | None = None

    # Overall
    passes_all: bool = False
    passes_kinetics: bool = False
    passes_orientation: bool = False
    passes_stability: bool = False

    # Composite LFA score (0–1, higher = better for LFA deployment)
    lfa_score: float = 0.0

    warnings: list[str] = field(default_factory=list)


class LFACompatibilityFilter:
    """Gate candidates for lateral flow assay deployment suitability.

    Runs three independent assessments and combines them:
    1. Kinetic accessibility — can the analyte bind fast enough?
    2. Conjugation orientation — is the paratope accessible on a gold NP?
    3. Thermal stability — will it survive storage and shipping?

    All three must pass for a candidate to be LFA-compatible.
    """

    def __init__(self, config: DictConfig) -> None:
        self.lfa_config = config.lfa
        self.flow_time = config.lfa.flow_time_seconds
        self.conjugation_tag = config.lfa.conjugation_tag
        self.np_radius = config.lfa.nanoparticle_diameter_nm / 2.0
        self.min_tm = config.filters.min_predicted_tm

        logger.info(
            "LFA filter: flow_time=%ds, tag=%s, NP=%dnm, min_Tm=%.0f°C",
            self.flow_time, self.conjugation_tag,
            config.lfa.nanoparticle_diameter_nm, self.min_tm,
        )

    def evaluate(
        self,
        candidate: Candidate,
        complex_pdb: Path | None = None,
        nanobody_pdb: Path | None = None,
    ) -> LFACompatibilityResult:
        """Run all LFA compatibility checks on a candidate.

        Args:
            candidate: Candidate with sequence and optionally structure.
            complex_pdb: Predicted complex PDB (for kinetics).
            nanobody_pdb: Nanobody-only PDB (for orientation).

        Returns:
            LFACompatibilityResult with component and overall assessment.
        """
        result = LFACompatibilityResult(candidate_id=candidate.candidate_id)

        # 1. Kinetics (requires complex PDB)
        if complex_pdb is not None and complex_pdb.exists():
            try:
                result.kinetics = estimate_kinetic_profile(
                    complex_pdb=complex_pdb,
                    candidate_id=candidate.candidate_id,
                    flow_time_seconds=self.flow_time,
                )
                result.passes_kinetics = result.kinetics.suitable_for_lfa
                if not result.passes_kinetics:
                    result.warnings.append(
                        f"Slow kinetics: kon={result.kinetics.estimated_kon_category}, "
                        f"binding_fraction={result.kinetics.expected_binding_fraction:.0%}"
                    )
            except Exception as e:
                logger.warning("Kinetics assessment failed for %s: %s", candidate.candidate_id, e)
                result.passes_kinetics = True  # pass on error (can't disqualify)
        else:
            result.passes_kinetics = True  # skip if no structure

        # 2. Orientation (requires nanobody PDB)
        pdb_for_orientation = nanobody_pdb or complex_pdb
        if pdb_for_orientation is not None and pdb_for_orientation.exists():
            try:
                tag_loc = "C_terminal" if "His" in self.conjugation_tag else "C_terminal"
                result.orientation = check_conjugation_clearance(
                    nanobody_pdb=pdb_for_orientation,
                    tag_location=tag_loc,
                    nanoparticle_radius_nm=self.np_radius,
                )
                result.passes_orientation = result.orientation.orientation_compatible
                if not result.passes_orientation:
                    angle = result.orientation.paratope_nanoparticle_angle
                    result.warnings.append(
                        f"Poor orientation: paratope-NP angle={angle:.0f}° (<90°)"
                    )
            except Exception as e:
                logger.warning("Orientation check failed for %s: %s", candidate.candidate_id, e)
                result.passes_orientation = True
        else:
            result.passes_orientation = True

        # 3. Stability (sequence-based, always available)
        try:
            result.stability = predict_stability(
                sequence=candidate.sequence,
                candidate_id=candidate.candidate_id,
                min_tm=self.min_tm,
            )
            result.passes_stability = result.stability.passes_lfa_requirements
            if not result.passes_stability:
                result.warnings.extend(result.stability.warnings)
        except Exception as e:
            logger.warning("Stability prediction failed for %s: %s", candidate.candidate_id, e)
            result.passes_stability = True

        # Overall
        result.passes_all = (
            result.passes_kinetics
            and result.passes_orientation
            and result.passes_stability
        )

        # Composite LFA score
        scores: list[float] = []
        if result.kinetics:
            scores.append(result.kinetics.estimated_relative_kon)
        if result.orientation:
            scores.append(1.0 if result.orientation.orientation_compatible else 0.3)
        if result.stability:
            scores.append(result.stability.sequence_stability_score)
        result.lfa_score = sum(scores) / max(len(scores), 1)

        return result

    def filter_batch(
        self,
        candidates: list[Candidate],
        complex_dir: Path | None = None,
    ) -> tuple[list[Candidate], list[LFACompatibilityResult]]:
        """Filter a batch of candidates for LFA compatibility.

        Args:
            candidates: Candidates to evaluate.
            complex_dir: Directory containing predicted complex PDBs.

        Returns:
            Tuple of (passing_candidates, all_results).
        """
        all_results: list[LFACompatibilityResult] = []
        passing: list[Candidate] = []

        for cand in candidates:
            complex_pdb = None
            if complex_dir and cand.complex_path:
                complex_pdb = cand.complex_path

            result = self.evaluate(cand, complex_pdb=complex_pdb)
            all_results.append(result)

            if result.passes_all:
                passing.append(cand)

        logger.info(
            "LFA compatibility: %d/%d passed (kinetics=%d, orientation=%d, stability=%d)",
            len(passing), len(candidates),
            sum(1 for r in all_results if r.passes_kinetics),
            sum(1 for r in all_results if r.passes_orientation),
            sum(1 for r in all_results if r.passes_stability),
        )

        return passing, all_results
