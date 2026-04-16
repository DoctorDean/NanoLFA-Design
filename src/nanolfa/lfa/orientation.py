"""LFA-specific kinetic accessibility and conjugation orientation analysis.

Evaluates whether nanobody candidates are suitable for lateral flow assay
deployment based on binding pocket accessibility, electrostatic steering,
and gold nanoparticle conjugation geometry.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KineticAccessibilityResult:
    """Assessment of binding kinetics suitability for LFA."""

    pocket_openness: float        # solid angle fraction [0, 1]; wider = faster kon
    electrostatic_score: float    # charge complementarity at entrance [0, 1]
    cdr_flexibility: float        # mean pLDDT of CDR loops; higher = more rigid
    estimated_relative_kon: float # composite kinetic accessibility [0, 1]
    suitable_for_lfa: bool


@dataclass
class OrientationResult:
    """Assessment of conjugation compatibility."""

    paratope_nanoparticle_angle: float  # degrees; >90° is good
    min_clearance_angstrom: float       # Å; minimum distance paratope to NP surface
    tag_accessible: bool                # is C-terminal tag solvent-exposed?
    orientation_compatible: bool


def estimate_pocket_accessibility(
    complex_pdb: str | Path,
    ligand_chain: str = "B",
    receptor_chain: str = "A",
    probe_radius: float = 1.4,
    n_samples: int = 1000,
) -> KineticAccessibilityResult:
    """Estimate kinetic accessibility of the binding pocket.

    Uses ray-casting from the ligand centroid to estimate the solid angle
    of the pocket entrance. A wider opening allows faster analyte access,
    which is critical for LFA where the sample flows past the test line
    in under 60 seconds.

    Args:
        complex_pdb: Path to the predicted complex PDB.
        ligand_chain: Chain ID of the ligand/hapten.
        receptor_chain: Chain ID of the nanobody.
        probe_radius: Probe radius in Å for accessibility calculation.
        n_samples: Number of random rays for solid angle estimation.

    Returns:
        KineticAccessibilityResult with pocket openness metrics.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(complex_pdb))
    model = structure[0]

    # Get ligand centroid
    ligand_atoms = [
        atom.get_vector().get_array()
        for atom in model[ligand_chain].get_atoms()
    ]
    ligand_centroid = np.mean(ligand_atoms, axis=0)

    # Get receptor atom coordinates
    receptor_atoms = np.array([
        atom.get_vector().get_array()
        for atom in model[receptor_chain].get_atoms()
    ])

    # Ray-casting: shoot random rays from ligand centroid,
    # count fraction that escape without hitting receptor atoms
    rng = np.random.default_rng(42)
    escaped = 0

    for _ in range(n_samples):
        # Random direction on unit sphere
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)

        # Check if ray hits any receptor atom (within probe_radius)
        # Simplified: check closest approach along the ray
        hit = False
        for t in np.linspace(0, 20, 50):  # sample along ray up to 20Å
            point = ligand_centroid + t * direction
            distances = np.linalg.norm(receptor_atoms - point, axis=1)
            if np.min(distances) < probe_radius + 1.5:  # atom radius ~1.5Å
                hit = True
                break

        if not hit:
            escaped += 1

    pocket_openness = escaped / n_samples

    # CDR flexibility from pLDDT (stored in B-factors)
    cdr_plddt_values = []
    for residue in model[receptor_chain]:
        res_num = residue.get_id()[1]
        # CDR regions (IMGT approximate)
        if 27 <= res_num <= 38 or 56 <= res_num <= 65 or 105 <= res_num <= 117:
            ca = residue.get("CA")  # type: ignore[union-attr]
            if ca is not None:
                cdr_plddt_values.append(ca.get_bfactor())

    cdr_flexibility = np.mean(cdr_plddt_values) if cdr_plddt_values else 50.0

    # Composite kinetic accessibility
    # High pocket openness + rigid CDRs + good electrostatics = fast kon
    rigidity_score = min(1.0, cdr_flexibility / 90.0)
    estimated_kon = 0.6 * pocket_openness + 0.4 * rigidity_score

    return KineticAccessibilityResult(
        pocket_openness=pocket_openness,
        electrostatic_score=0.5,  # placeholder — requires APBS calculation
        cdr_flexibility=cdr_flexibility,
        estimated_relative_kon=estimated_kon,
        suitable_for_lfa=estimated_kon > 0.4,
    )


def check_conjugation_clearance(
    nanobody_pdb: str | Path,
    tag_location: str = "C_terminal",
    nanoparticle_radius_nm: float = 20.0,
    paratope_residues: list[int] | None = None,
) -> OrientationResult:
    """Verify that the paratope is accessible when conjugated to a gold nanoparticle.

    For LFA detection antibodies, the nanobody is conjugated to a gold nanoparticle
    via a C-terminal tag (His6, SpyTag, etc.). The paratope (CDR loops) must face
    AWAY from the nanoparticle surface to allow antigen binding.

    The key metric is the angle between:
    - Vector from nanobody centroid to paratope centroid
    - Vector from nanobody centroid to C-terminus (nanoparticle attachment point)

    Angles > 90° indicate the paratope faces away from the nanoparticle (good).
    Angles < 90° indicate potential steric occlusion (bad).

    Args:
        nanobody_pdb: Path to nanobody PDB (chain A).
        tag_location: "C_terminal" or "N_terminal".
        nanoparticle_radius_nm: Radius of gold nanoparticle in nm.
        paratope_residues: List of CDR residue numbers (IMGT). If None, uses defaults.

    Returns:
        OrientationResult with angle and clearance metrics.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("nanobody", str(nanobody_pdb))
    model = structure[0]
    chain = list(model.get_chains())[0]

    # Default CDR positions (IMGT)
    if paratope_residues is None:
        paratope_residues = (
            list(range(27, 39)) + list(range(56, 66)) + list(range(105, 118))
        )

    # Get all CA atoms
    all_ca: list[np.ndarray] = []
    paratope_ca: list[np.ndarray] = []
    terminal_ca: np.ndarray | None = None

    residues = list(chain.get_residues())
    for residue in residues:
        ca = residue.get("CA")  # type: ignore[union-attr]
        if ca is None:
            continue
        coord = ca.get_vector().get_array()
        all_ca.append(coord)

        res_num = residue.get_id()[1]
        if res_num in paratope_residues:
            paratope_ca.append(coord)

    # C-terminal CA
    if tag_location == "C_terminal":
        for residue in reversed(residues):
            ca = residue.get("CA")  # type: ignore[union-attr]
            if ca is not None:
                terminal_ca = ca.get_vector().get_array()
                break
    else:
        for residue in residues:
            ca = residue.get("CA")  # type: ignore[union-attr]
            if ca is not None:
                terminal_ca = ca.get_vector().get_array()
                break

    if not all_ca or not paratope_ca or terminal_ca is None:
        logger.warning("Insufficient atoms for orientation analysis")
        return OrientationResult(
            paratope_nanoparticle_angle=0.0,
            min_clearance_angstrom=0.0,
            tag_accessible=False,
            orientation_compatible=False,
        )

    # Compute centroids
    centroid = np.mean(all_ca, axis=0)
    paratope_centroid = np.mean(paratope_ca, axis=0)

    # Vectors
    vec_to_paratope = paratope_centroid - centroid
    vec_to_terminal = terminal_ca - centroid

    # Angle between vectors
    cos_angle = np.dot(vec_to_paratope, vec_to_terminal) / (
        np.linalg.norm(vec_to_paratope) * np.linalg.norm(vec_to_terminal) + 1e-8
    )
    angle_deg = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))

    # Minimum clearance: distance from paratope centroid to the nanoparticle surface
    # NP center is at terminal_ca extended by NP radius
    np_radius_angstrom = nanoparticle_radius_nm * 10  # nm → Å
    vec_terminal_norm = vec_to_terminal / (np.linalg.norm(vec_to_terminal) + 1e-8)
    np_center = terminal_ca + vec_terminal_norm * np_radius_angstrom
    dist_paratope_to_np_center = np.linalg.norm(paratope_centroid - np_center)
    clearance = dist_paratope_to_np_center - np_radius_angstrom

    # Tag accessibility: is the terminal residue on the surface?
    # Check if terminal CA is farther from centroid than median
    distances_from_centroid = [np.linalg.norm(ca - centroid) for ca in all_ca]
    median_dist = np.median(distances_from_centroid)
    terminal_dist = np.linalg.norm(terminal_ca - centroid)
    tag_accessible = terminal_dist >= median_dist * 0.8

    return OrientationResult(
        paratope_nanoparticle_angle=angle_deg,
        min_clearance_angstrom=max(0.0, clearance),
        tag_accessible=tag_accessible,
        orientation_compatible=angle_deg > 90.0 and clearance > 10.0 and tag_accessible,
    )
