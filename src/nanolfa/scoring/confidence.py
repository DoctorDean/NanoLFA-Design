"""AlphaFold confidence metric extraction.

Extracts and processes confidence metrics from AlphaFold 2/3 outputs:
ipTM, pTM, per-residue pLDDT, PAE (predicted aligned error), and
pDockQ. These feed into the composite scoring function.

Supports both AF2-Multimer (ranking_debug.json) and AF3 (per-model
JSON) output formats.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AFConfidenceMetrics:
    """Complete confidence metrics from an AlphaFold prediction."""

    model_id: str
    iptm: float                           # interface predicted TM-score [0, 1]
    ptm: float                            # predicted TM-score [0, 1]
    mean_plddt: float                     # overall mean pLDDT [0, 100]
    interface_plddt: float                # mean pLDDT at interface residues
    per_residue_plddt: list[float]        # per-residue pLDDT values
    ranking_confidence: float             # AF ranking metric (iptm*0.8 + ptm*0.2 for AF2)
    pae_mean: float | None = None         # mean predicted aligned error (Å)
    pae_interface_mean: float | None = None  # mean PAE at interface
    pdockq: float | None = None           # pDockQ score [0, 1] (computed from pLDDT + interface)


def extract_confidence(
    prediction_dir: Path,
    complex_pdb: Path | None = None,
    interface_distance: float = 5.0,
) -> AFConfidenceMetrics | None:
    """Extract all confidence metrics from an AlphaFold prediction.

    Tries AF3 format first, then AF2-Multimer format.

    Args:
        prediction_dir: Directory containing AF output files.
        complex_pdb: PDB path for interface pLDDT calculation.
        interface_distance: Cutoff distance for interface residues (Å).

    Returns:
        AFConfidenceMetrics or None if parsing fails.
    """
    # Try AF3 format
    metrics = _parse_af3(prediction_dir)
    if metrics is None:
        metrics = _parse_af2_multimer(prediction_dir)

    if metrics is None:
        logger.warning("No AF confidence data found in %s", prediction_dir)
        return None

    # Compute interface pLDDT if PDB is available
    if complex_pdb is not None and complex_pdb.exists():
        metrics.interface_plddt = compute_interface_plddt(
            complex_pdb, interface_distance,
        )

    # Compute pDockQ
    if metrics.interface_plddt > 0:
        metrics.pdockq = compute_pdockq(
            metrics.interface_plddt, metrics.mean_plddt,
        )

    return metrics


def compute_interface_plddt(
    complex_pdb: Path,
    interface_distance: float = 5.0,
) -> float:
    """Compute mean pLDDT of residues at the binding interface.

    Interface residues are those with any heavy atom within
    `interface_distance` Å of any heavy atom on the partner chain.
    pLDDT values are read from the B-factor column (standard for AF outputs).

    Args:
        complex_pdb: Path to predicted complex PDB.
        interface_distance: Cutoff distance (Å).

    Returns:
        Mean pLDDT of interface residues.
    """
    from nanolfa.utils.pdb import identify_interface_residues

    interface_a, interface_b = identify_interface_residues(
        complex_pdb, distance_cutoff=interface_distance,
    )

    all_interface = interface_a + interface_b
    if not all_interface:
        return 0.0

    bfactors = [r.bfactor for r in all_interface if r.bfactor > 0]
    return float(np.mean(bfactors)) if bfactors else 0.0


def compute_pdockq(
    interface_plddt: float,
    overall_plddt: float,
    n_interface_contacts: int = 50,
) -> float:
    """Compute pDockQ from pLDDT values.

    pDockQ estimates the probability that the predicted complex is a
    true positive. Based on the sigmoid relationship between average
    interface pLDDT and DockQ score (Bryant et al., 2022).

    Simplified formula:
        pDockQ = L / (1 + exp(-k * (x - x0)))
    where x = mean interface pLDDT, and L, k, x0 are fitted parameters.

    Args:
        interface_plddt: Mean pLDDT at the interface.
        overall_plddt: Overall mean pLDDT of the complex.
        n_interface_contacts: Number of inter-chain contacts.

    Returns:
        pDockQ score [0, 1].
    """
    # Sigmoid parameters from pDockQ publication
    x = interface_plddt / 100.0  # normalize to [0, 1]
    l_param = 0.724
    k_param = 12.0
    x0_param = 0.55

    pdockq = l_param / (1.0 + np.exp(-k_param * (x - x0_param)))

    # Adjust for very few contacts (low confidence)
    if n_interface_contacts < 10:
        pdockq *= n_interface_contacts / 10.0

    return float(np.clip(pdockq, 0.0, 1.0))


def extract_pae_matrix(
    prediction_dir: Path,
) -> np.ndarray | None:
    """Extract the predicted aligned error (PAE) matrix from AF output.

    The PAE matrix is an NxN array where PAE[i,j] is the expected error
    in the position of residue j when aligned on residue i.
    Low PAE between chains indicates confident interface prediction.

    Returns:
        PAE matrix as numpy array, or None if not found.
    """
    # AF2 format
    for pae_file in prediction_dir.glob("*pae*.json"):
        try:
            with open(pae_file) as f:
                data = json.load(f)
            if "predicted_aligned_error" in data:
                return np.array(data["predicted_aligned_error"])
            if isinstance(data, list) and len(data) > 0 and "predicted_aligned_error" in data[0]:
                return np.array(data[0]["predicted_aligned_error"])
        except Exception:
            continue

    # AF3 format — PAE stored differently
    for json_file in prediction_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if "pae" in data:
                return np.array(data["pae"])
        except Exception:
            continue

    return None


def compute_interface_pae(
    pae_matrix: np.ndarray,
    chain_a_length: int,
) -> float:
    """Compute mean PAE at the interface between two chains.

    Extracts the off-diagonal blocks of the PAE matrix corresponding
    to inter-chain predictions.

    Args:
        pae_matrix: Full NxN PAE matrix.
        chain_a_length: Number of residues in chain A.

    Returns:
        Mean PAE at the interface (Å). Lower = more confident.
    """
    n = pae_matrix.shape[0]
    if chain_a_length >= n:
        return 30.0  # max PAE, no confidence

    # Off-diagonal blocks: chain A predicting chain B positions and vice versa
    block_ab = pae_matrix[:chain_a_length, chain_a_length:]
    block_ba = pae_matrix[chain_a_length:, :chain_a_length]

    interface_pae = np.concatenate([block_ab.flatten(), block_ba.flatten()])
    return float(np.mean(interface_pae))


# ---------------------------------------------------------------------------
# Format-specific parsers
# ---------------------------------------------------------------------------

def _parse_af3(prediction_dir: Path) -> AFConfidenceMetrics | None:
    """Parse AF3 output format."""
    for json_file in sorted(prediction_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if "iptm" in data or "interface_ptm" in data:
                plddt_list = data.get("plddt", data.get("atom_plddts", []))
                per_res = [float(v) for v in plddt_list] if isinstance(plddt_list, list) else []

                return AFConfidenceMetrics(
                    model_id=json_file.stem,
                    iptm=data.get("iptm", data.get("interface_ptm", 0.0)),
                    ptm=data.get("ptm", 0.0),
                    mean_plddt=float(np.mean(per_res)) if per_res else 0.0,
                    interface_plddt=0.0,  # computed later from PDB
                    per_residue_plddt=per_res,
                    ranking_confidence=data.get("ranking_confidence", 0.0),
                )
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def _parse_af2_multimer(prediction_dir: Path) -> AFConfidenceMetrics | None:
    """Parse AF2-Multimer output format."""
    ranking_file = prediction_dir / "ranking_debug.json"
    if not ranking_file.exists():
        return None

    try:
        with open(ranking_file) as f:
            ranking = json.load(f)

        # Get the best model
        iptm_ptm = ranking.get("iptm+ptm", {})
        if not iptm_ptm:
            return None

        best_model = max(iptm_ptm, key=iptm_ptm.get)
        iptm = ranking.get("iptm", {}).get(best_model, 0.0)
        ptm = ranking.get("ptm", {}).get(best_model, 0.0)
        mean_plddt = ranking.get("plddts", {}).get(best_model, 0.0)

        return AFConfidenceMetrics(
            model_id=best_model,
            iptm=iptm,
            ptm=ptm,
            mean_plddt=mean_plddt,
            interface_plddt=0.0,
            per_residue_plddt=[],
            ranking_confidence=iptm_ptm[best_model],
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to parse AF2 ranking: %s", e)
        return None
