"""Kinetic parameter estimation from structural predictions.

Estimates relative association rate (kon) from structural features of
the predicted nanobody-hapten complex. Fast on-rates are critical for
LFA performance because the analyte transits the test line in under
60 seconds.

Three components contribute to kinetic accessibility:
1. Binding pocket openness — wider entrance = faster diffusion in
2. CDR loop rigidity — rigid loops = lock-and-key (fast) vs.
   conformational selection (slow)
3. Charge complementarity — electrostatic steering accelerates approach
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KineticProfile:
    """Complete kinetic assessment for a candidate."""

    candidate_id: str
    pocket_openness: float          # 0–1; fraction of rays escaping
    cdr_rigidity: float             # 0–1; normalized mean CDR pLDDT
    charge_complementarity: float   # 0–1; electrostatic steering estimate
    estimated_relative_kon: float   # 0–1; composite kinetic accessibility
    estimated_kon_category: str     # "fast", "moderate", "slow"
    suitable_for_lfa: bool

    # LFA-specific timing
    expected_binding_fraction: float  # estimated fraction bound in 60s


def estimate_kinetic_profile(
    complex_pdb: str | Path,
    candidate_id: str = "unknown",
    ligand_chain: str = "B",
    receptor_chain: str = "A",
    flow_time_seconds: float = 60.0,
    probe_radius: float = 1.4,
    n_ray_samples: int = 1000,
) -> KineticProfile:
    """Estimate kinetic accessibility from a predicted complex structure.

    Combines pocket openness, CDR rigidity, and charge complementarity
    into a composite kinetic score. Maps the score to an expected
    binding fraction within the LFA flow time.

    Args:
        complex_pdb: Path to predicted complex PDB.
        candidate_id: Identifier for logging.
        ligand_chain: Chain ID of hapten/ligand.
        receptor_chain: Chain ID of nanobody.
        flow_time_seconds: Analyte transit time across test line.
        probe_radius: Probe radius for accessibility calculation (Å).
        n_ray_samples: Number of random rays for pocket openness.

    Returns:
        KineticProfile with all kinetic estimates.
    """
    # 1. Pocket openness via ray casting
    pocket_openness = _compute_pocket_openness(
        complex_pdb, ligand_chain, receptor_chain,
        probe_radius, n_ray_samples,
    )

    # 2. CDR rigidity from pLDDT (B-factors in AF output)
    cdr_rigidity = _compute_cdr_rigidity(complex_pdb, receptor_chain)

    # 3. Charge complementarity (simplified)
    charge_comp = _estimate_charge_complementarity(
        complex_pdb, ligand_chain, receptor_chain,
    )

    # Composite kinetic accessibility
    # Weights reflect relative importance for LFA kon
    estimated_kon = (
        0.50 * pocket_openness
        + 0.35 * cdr_rigidity
        + 0.15 * charge_comp
    )

    # Categorize
    if estimated_kon >= 0.6:
        category = "fast"
    elif estimated_kon >= 0.35:
        category = "moderate"
    else:
        category = "slow"

    # Estimate binding fraction in flow time
    # Simplified: binding_fraction ≈ 1 - exp(-kon_relative * t / tau)
    # where tau is a characteristic time scale
    tau = 30.0  # seconds; calibration constant
    binding_fraction = float(1.0 - np.exp(-estimated_kon * flow_time_seconds / tau))

    suitable = estimated_kon >= 0.4 and binding_fraction >= 0.5

    return KineticProfile(
        candidate_id=candidate_id,
        pocket_openness=pocket_openness,
        cdr_rigidity=cdr_rigidity,
        charge_complementarity=charge_comp,
        estimated_relative_kon=estimated_kon,
        estimated_kon_category=category,
        suitable_for_lfa=suitable,
        expected_binding_fraction=binding_fraction,
    )


def _compute_pocket_openness(
    complex_pdb: str | Path,
    ligand_chain: str,
    receptor_chain: str,
    probe_radius: float,
    n_samples: int,
) -> float:
    """Estimate pocket openness by ray casting from ligand centroid.

    Shoots random rays from the ligand centroid and counts what fraction
    escape without hitting receptor atoms. Higher = more accessible pocket.
    """
    try:
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", str(complex_pdb))
        model = structure[0]

        ligand_atoms = [
            atom.get_vector().get_array()
            for atom in model[ligand_chain].get_atoms()
        ]
        receptor_atoms = np.array([
            atom.get_vector().get_array()
            for atom in model[receptor_chain].get_atoms()
        ])

        if not ligand_atoms or len(receptor_atoms) == 0:
            return 0.5

        ligand_centroid = np.mean(ligand_atoms, axis=0)
        rng = np.random.default_rng(42)
        escaped = 0

        for _ in range(n_samples):
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)

            hit = False
            for t in np.linspace(1.0, 20.0, 40):
                point = ligand_centroid + t * direction
                distances = np.linalg.norm(receptor_atoms - point, axis=1)
                if np.min(distances) < probe_radius + 1.5:
                    hit = True
                    break

            if not hit:
                escaped += 1

        return escaped / n_samples

    except Exception as e:
        logger.warning("Pocket openness calculation failed: %s", e)
        return 0.5


def _compute_cdr_rigidity(
    complex_pdb: str | Path,
    receptor_chain: str,
) -> float:
    """Estimate CDR loop rigidity from pLDDT values.

    Higher pLDDT in CDR loops suggests more ordered conformation,
    which correlates with lock-and-key binding (faster kon).
    """
    try:
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", str(complex_pdb))
        model = structure[0]

        cdr_plddt: list[float] = []
        for residue in model[receptor_chain]:
            res_num = residue.get_id()[1]
            # CDR regions in IMGT numbering
            if 27 <= res_num <= 38 or 56 <= res_num <= 65 or 105 <= res_num <= 117:
                ca = residue.get("CA")
                if ca is not None:
                    cdr_plddt.append(ca.get_bfactor())

        if not cdr_plddt:
            return 0.5

        mean_plddt = np.mean(cdr_plddt)
        # Normalize: pLDDT 40 → 0.0, pLDDT 90 → 1.0
        return float(np.clip((mean_plddt - 40) / 50, 0, 1))

    except Exception as e:
        logger.warning("CDR rigidity calculation failed: %s", e)
        return 0.5


def _estimate_charge_complementarity(
    complex_pdb: str | Path,
    ligand_chain: str,
    receptor_chain: str,
    interface_distance: float = 5.0,
) -> float:
    """Estimate electrostatic complementarity at the binding interface.

    Simplified version: counts charged residue pairs across the interface.
    Full version would use APBS for electrostatic potential mapping.
    """
    try:
        from Bio.PDB import NeighborSearch, PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", str(complex_pdb))
        model = structure[0]

        receptor_atoms = list(model[receptor_chain].get_atoms())
        ligand_atoms = list(model[ligand_chain].get_atoms())

        if not receptor_atoms or not ligand_atoms:
            return 0.5

        ns = NeighborSearch(ligand_atoms)

        positive_residues = {"ARG", "LYS", "HIS"}
        negative_residues = {"ASP", "GLU"}

        interface_charges: list[int] = []
        seen: set[int] = set()

        for atom in receptor_atoms:
            neighbors = ns.search(atom.get_vector().get_array(), interface_distance)
            if neighbors:
                res = atom.get_parent()
                res_id = res.get_id()[1]
                if res_id in seen:
                    continue
                seen.add(res_id)
                resname = res.get_resname()
                if resname in positive_residues:
                    interface_charges.append(1)
                elif resname in negative_residues:
                    interface_charges.append(-1)

        if not interface_charges:
            return 0.5

        # Glucuronide carboxylate is negative; complementary positive
        # charges on nanobody improve steering
        positive_count = sum(1 for c in interface_charges if c > 0)
        total = len(interface_charges)
        return float(np.clip(positive_count / max(total, 1), 0, 1))

    except Exception as e:
        logger.warning("Charge complementarity calculation failed: %s", e)
        return 0.5
