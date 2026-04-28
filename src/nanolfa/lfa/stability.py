"""Thermal stability prediction for nanobody candidates.

Estimates apparent melting temperature (Tm) and aggregation onset
temperature (Tagg) to ensure candidates survive LFA manufacturing,
storage, and use conditions.

LFA requirements:
- Tm > 65°C (survive shipping/storage at 25°C with safety margin)
- Tagg > 55°C (no aggregation during gold conjugation at 40°C)
- Shelf life > 18 months at 25°C / 60% RH
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Amino acid contributions to VHH stability (empirical, simplified)
# Positive = stabilizing, negative = destabilizing
_STABILITY_CONTRIBUTIONS = {
    "P": 0.5,   # proline in turns stabilizes
    "C": -1.0,  # free cysteines destabilize (beyond canonical pair)
    "G": -0.2,  # glycine increases flexibility
    "W": 0.3,   # tryptophan packs well in core
    "Y": 0.2,   # tyrosine H-bonds
    "I": 0.3,   # isoleucine core packing
    "V": 0.3,   # valine core packing
    "L": 0.2,   # leucine core packing
    "M": -0.5,  # methionine oxidation risk
    "N": -0.3,  # asparagine deamidation risk
    "D": -0.1,  # aspartate isomerization risk
}


@dataclass
class StabilityProfile:
    """Thermal stability assessment for a candidate."""

    candidate_id: str
    predicted_tm: float          # °C; apparent melting temperature
    predicted_tagg: float        # °C; aggregation onset temperature
    sequence_stability_score: float  # 0–1; from amino acid composition
    cdr3_flexibility_penalty: float  # 0–1; higher = more flexible CDR3
    cys_count: int               # total cysteines
    free_cys_count: int          # cysteines beyond canonical pair
    met_count: int               # oxidation-liable residues
    passes_lfa_requirements: bool
    warnings: list[str]


def predict_stability(
    sequence: str,
    candidate_id: str = "unknown",
    min_tm: float = 65.0,
    min_tagg: float = 55.0,
) -> StabilityProfile:
    """Predict thermal stability for a nanobody candidate.

    Uses a sequence-based heuristic model. In production, complement with:
    - Rosetta total score (monomer fold stability)
    - ESM-IF perplexity (designability proxy)
    - Molecular dynamics simulation (expensive but most accurate)

    Args:
        sequence: Nanobody amino acid sequence.
        candidate_id: Identifier.
        min_tm: Minimum Tm for LFA compatibility (°C).
        min_tagg: Minimum Tagg for LFA compatibility (°C).

    Returns:
        StabilityProfile with predicted thermal parameters.
    """
    warnings: list[str] = []
    seq_len = len(sequence)

    # Baseline Tm for well-folded VHH
    base_tm = 72.0  # °C — typical for alpaca VHH

    # Sequence composition analysis
    stability_delta = 0.0
    for aa, contribution in _STABILITY_CONTRIBUTIONS.items():
        count = sequence.count(aa)
        stability_delta += count * contribution * 0.1

    # Cysteine analysis
    cys_count = sequence.count("C")
    # Canonical VHH has exactly 2 cysteines (Cys23-Cys104)
    free_cys = max(0, cys_count - 2)
    if free_cys > 0:
        stability_delta -= free_cys * 2.0
        warnings.append(f"{free_cys} free Cys (aggregation risk)")

    # Met oxidation risk
    met_count = sequence.count("M")
    if met_count > 2:
        stability_delta -= (met_count - 2) * 0.5
        warnings.append(f"{met_count} Met residues (oxidation risk)")

    # CDR3 flexibility penalty
    cdr3_start = min(104, seq_len)
    cdr3_end = min(117, seq_len)
    cdr3 = sequence[cdr3_start:cdr3_end]
    cdr3_gly_fraction = cdr3.count("G") / max(len(cdr3), 1)
    cdr3_flexibility = cdr3_gly_fraction  # higher Gly = more flexible
    if cdr3_flexibility > 0.3:
        stability_delta -= 3.0
        warnings.append(f"CDR3 Gly-rich ({cdr3_flexibility:.0%})")

    # CDR3 length penalty (very long CDR3 can be destabilizing)
    cdr3_len = len(cdr3)
    if cdr3_len > 18:
        stability_delta -= (cdr3_len - 18) * 0.3
        warnings.append(f"Long CDR3 ({cdr3_len} residues)")

    # Sequence stability score (normalized)
    raw_score = base_tm + stability_delta
    seq_stability_score = float(np.clip((raw_score - 50) / 40, 0, 1))

    # Predicted Tm
    predicted_tm = base_tm + stability_delta

    # Tagg is typically 10-15°C below Tm for nanobodies
    predicted_tagg = predicted_tm - 12.0

    # LFA compatibility check
    passes = predicted_tm >= min_tm and predicted_tagg >= min_tagg
    if predicted_tm < min_tm:
        warnings.append(f"Tm={predicted_tm:.0f}°C < {min_tm:.0f}°C requirement")
    if predicted_tagg < min_tagg:
        warnings.append(f"Tagg={predicted_tagg:.0f}°C < {min_tagg:.0f}°C requirement")

    return StabilityProfile(
        candidate_id=candidate_id,
        predicted_tm=predicted_tm,
        predicted_tagg=predicted_tagg,
        sequence_stability_score=seq_stability_score,
        cdr3_flexibility_penalty=cdr3_flexibility,
        cys_count=cys_count,
        free_cys_count=free_cys,
        met_count=met_count,
        passes_lfa_requirements=passes,
        warnings=warnings,
    )
