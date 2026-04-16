"""Developability filters for nanobody candidates.

Evaluates aggregation propensity, charge, hydrophobicity, sequence liabilities,
and predicted thermal stability. These properties determine whether a nanobody
is manufacturable and suitable for lateral flow assay deployment.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from omegaconf import DictConfig

if TYPE_CHECKING:
    from nanolfa.core.pipeline import Candidate

logger = logging.getLogger(__name__)

# Kyte-Doolittle hydrophobicity scale
KD_HYDROPHOBICITY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "D": -3.5, "E": -3.5, "N": -3.5, "Q": -3.5, "K": -3.9, "R": -4.5,
}

# Amino acid charges at pH 7.0
AA_CHARGE = {
    "D": -1, "E": -1, "K": +1, "R": +1, "H": +0.1,  # His partially protonated
}


@dataclass
class DevelopabilityReport:
    """Results of developability evaluation for a single candidate."""

    candidate_id: str

    # Aggregation
    aggregation_score: float  # 0–1; lower is better

    # Charge
    cdr3_net_charge: float
    total_net_charge: float

    # Hydrophobicity
    cdr_mean_hydrophobicity: float  # Kyte-Doolittle mean over CDR residues

    # Sequence liabilities
    liability_count: int
    liabilities: list[str]  # e.g., ["NG@position45", "M@position72"]

    # Thermal stability
    predicted_tm: float  # °C

    # Composite developability score [0, 1]
    composite_score: float

    # Pass/fail
    passes: bool
    failure_reasons: list[str]


class DevelopabilityFilter:
    """Evaluate nanobody candidates for manufacturability and LFA compatibility.

    Combines multiple orthogonal assessments:
    1. Aggregation propensity (Aggrescan3D or CamSol)
    2. Charge balance in CDR loops and overall
    3. Hydrophobicity of CDR loops
    4. Sequence liability motifs (deamidation, isomerization, oxidation)
    5. Predicted thermal stability

    The composite developability score feeds into the main scoring function
    with weight w6 (default 0.10).
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.max_agg = config.max_aggregation_score
        self.charge_range = tuple(config.cdr3_net_charge_range)
        self.total_charge_range = tuple(config.total_net_charge_range)
        self.max_hydrophobicity = config.max_cdr_hydrophobicity
        self.flag_motifs = list(config.flag_motifs)
        self.max_liabilities = config.max_liability_count
        self.min_tm = config.min_predicted_tm

        # Compile liability regex patterns
        self._liability_patterns = self._compile_patterns(self.flag_motifs)

    def evaluate(self, candidate: Candidate) -> DevelopabilityReport:
        """Run full developability evaluation on a candidate.

        Args:
            candidate: Candidate with sequence (and optionally structure).

        Returns:
            DevelopabilityReport with all metrics and pass/fail status.
        """
        seq = candidate.sequence
        failure_reasons: list[str] = []

        # 1. Aggregation propensity
        agg_score = self._compute_aggregation(candidate)
        if agg_score > self.max_agg:
            failure_reasons.append(f"aggregation={agg_score:.2f}>{self.max_agg}")

        # 2. Charge analysis
        cdr3_charge = self._compute_cdr3_charge(seq)
        total_charge = self._compute_total_charge(seq)
        if not (self.charge_range[0] <= cdr3_charge <= self.charge_range[1]):
            failure_reasons.append(f"CDR3_charge={cdr3_charge:.1f}")
        if not (self.total_charge_range[0] <= total_charge <= self.total_charge_range[1]):
            failure_reasons.append(f"total_charge={total_charge:.1f}")

        # 3. CDR hydrophobicity
        cdr_hydro = self._compute_cdr_hydrophobicity(seq)
        if cdr_hydro > self.max_hydrophobicity:
            failure_reasons.append(f"CDR_hydrophobicity={cdr_hydro:.2f}")

        # 4. Sequence liabilities
        liabilities = self._scan_liabilities(seq)
        if len(liabilities) > self.max_liabilities:
            failure_reasons.append(f"liabilities={len(liabilities)}")

        # 5. Predicted thermal stability
        predicted_tm = self._predict_tm(candidate)
        if predicted_tm < self.min_tm:
            failure_reasons.append(f"Tm={predicted_tm:.0f}°C<{self.min_tm:.0f}°C")

        # Composite score
        composite = self._compute_composite(
            agg_score, cdr3_charge, cdr_hydro, len(liabilities), predicted_tm
        )

        return DevelopabilityReport(
            candidate_id=candidate.candidate_id,
            aggregation_score=agg_score,
            cdr3_net_charge=cdr3_charge,
            total_net_charge=total_charge,
            cdr_mean_hydrophobicity=cdr_hydro,
            liability_count=len(liabilities),
            liabilities=liabilities,
            predicted_tm=predicted_tm,
            composite_score=composite,
            passes=len(failure_reasons) == 0,
            failure_reasons=failure_reasons,
        )

    def compute_score(self, candidate: Candidate) -> float:
        """Shortcut to get just the composite developability score."""
        report = self.evaluate(candidate)
        return report.composite_score

    # ------------------------------------------------------------------
    # Component calculations
    # ------------------------------------------------------------------

    def _compute_aggregation(self, candidate: Candidate) -> float:
        """Estimate aggregation propensity.

        Uses Aggrescan3D if a structure is available, otherwise falls back
        to a sequence-based heuristic (mean hydrophobicity of patches).
        """
        if candidate.structure_path is not None:
            try:
                return self._aggrescan3d(candidate.structure_path)
            except Exception:
                pass

        # Sequence-based fallback: sliding window hydrophobicity
        seq = candidate.sequence
        window = 7
        if len(seq) < window:
            return 0.0

        patch_scores = []
        for i in range(len(seq) - window + 1):
            patch = seq[i : i + window]
            score = sum(KD_HYDROPHOBICITY.get(aa, 0) for aa in patch) / window
            patch_scores.append(score)

        # Normalize: high positive patches indicate aggregation-prone regions
        max_patch = max(patch_scores) if patch_scores else 0
        # Map to [0, 1] where 0 = no risk, 1 = high risk
        return max(0.0, min(1.0, (max_patch + 0.5) / 5.0))

    def _aggrescan3d(self, structure_path) -> float:
        """Run Aggrescan3D on a structure file. Returns normalized score."""
        # Placeholder — actual implementation calls Aggrescan3D binary or API
        logger.debug("Aggrescan3D not available, using sequence fallback")
        raise NotImplementedError

    def _compute_cdr3_charge(self, sequence: str) -> float:
        """Compute net charge of CDR3 residues at pH 7.0."""
        # CDR3 in IMGT numbering: positions 105–117 (approximate for VHH)
        cdr3_start = 104  # 0-indexed
        cdr3_end = min(117, len(sequence))
        cdr3 = sequence[cdr3_start:cdr3_end]
        return sum(AA_CHARGE.get(aa, 0) for aa in cdr3)

    def _compute_total_charge(self, sequence: str) -> float:
        """Compute net charge of the entire nanobody at pH 7.0."""
        return sum(AA_CHARGE.get(aa, 0) for aa in sequence)

    def _compute_cdr_hydrophobicity(self, sequence: str) -> float:
        """Compute mean Kyte-Doolittle hydrophobicity of CDR residues.

        Returns normalized value mapped to [0, 1] where 0 = hydrophilic, 1 = hydrophobic.
        """
        # Extract all CDR residues (approximate IMGT positions)
        cdr_positions = list(range(26, 39)) + list(range(55, 66)) + list(range(104, 118))
        cdr_residues = [
            sequence[i] for i in cdr_positions if i < len(sequence)
        ]

        if not cdr_residues:
            return 0.0

        mean_kd = sum(KD_HYDROPHOBICITY.get(aa, 0) for aa in cdr_residues) / len(cdr_residues)
        # Normalize: KD range is roughly -4.5 to +4.5
        return (mean_kd + 4.5) / 9.0

    def _scan_liabilities(self, sequence: str) -> list[str]:
        """Scan sequence for chemical liability motifs.

        Returns list of strings describing each liability found.
        """
        found: list[str] = []
        for pattern_name, regex in self._liability_patterns:
            for match in regex.finditer(sequence):
                pos = match.start() + 1  # 1-indexed
                found.append(f"{pattern_name}@{pos}({match.group()})")
        return found

    def _predict_tm(self, candidate: Candidate) -> float:
        """Predict apparent melting temperature.

        Uses a simple heuristic based on:
        - Framework conservation (more conserved = more stable)
        - Number of prolines in turns (stabilizing)
        - Absence of destabilizing mutations

        A proper implementation would use a trained ML model or Rosetta ddG.
        """
        seq = candidate.sequence

        # Baseline Tm for a well-folded VHH
        base_tm = 72.0  # °C — typical for alpaca VHH

        # Penalty for unusual framework residues
        # (simplified: count non-standard positions)
        penalty = 0.0

        # Bonus for prolines in turns
        pro_count = seq.count("P")
        bonus = min(pro_count * 0.3, 3.0)

        # Penalty for free cysteines (beyond the canonical disulfide pair)
        cys_count = seq.count("C")
        if cys_count > 2:
            penalty += (cys_count - 2) * 2.0

        # Penalty for glycine-rich CDR3 (flexible, destabilizing)
        cdr3 = seq[104 : min(118, len(seq))]
        gly_fraction = cdr3.count("G") / max(len(cdr3), 1)
        if gly_fraction > 0.3:
            penalty += 3.0

        return base_tm + bonus - penalty

    def _compute_composite(
        self,
        aggregation: float,
        cdr3_charge: float,
        hydrophobicity: float,
        liability_count: int,
        predicted_tm: float,
    ) -> float:
        """Combine sub-scores into a single developability meta-score [0, 1]."""
        # Each component normalized to [0, 1] where 1 = best
        s_agg = 1.0 - aggregation

        # Charge: best at 0, linearly worse
        charge_penalty = abs(cdr3_charge) / 4.0  # 0 at charge=0, 1 at charge=±4
        s_charge = max(0.0, 1.0 - charge_penalty)

        s_hydro = max(0.0, 1.0 - hydrophobicity)

        s_liability = max(0.0, 1.0 - liability_count / max(self.max_liabilities, 1))

        # Tm: normalized between 50°C (bad) and 85°C (excellent)
        s_tm = max(0.0, min(1.0, (predicted_tm - 50.0) / 35.0))

        # Equal weighting within the developability meta-score
        return (s_agg + s_charge + s_hydro + s_liability + s_tm) / 5.0

    @staticmethod
    def _compile_patterns(motifs: list[str]) -> list[tuple[str, re.Pattern]]:
        """Compile motif strings into regex patterns."""
        compiled = []
        for motif in motifs:
            # Handle simple motifs and regex-style patterns
            if motif.startswith("NX"):
                # N-glycosylation: N-X-S/T where X ≠ P
                pattern = re.compile(r"N[^P][ST]")
                compiled.append(("N-glycosylation", pattern))
            elif len(motif) == 1:
                # Single amino acid (e.g., M for oxidation)
                compiled.append((f"{motif}_exposed", re.compile(motif)))
            else:
                compiled.append((motif, re.compile(motif)))
        return compiled
