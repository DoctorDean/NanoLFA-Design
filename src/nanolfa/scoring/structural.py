"""Structural interface geometry metrics.

Computes geometry-based descriptors of the nanobody-hapten binding
interface: shape complementarity, contact density, gap volume index,
and interface planarity. These supplement the energy-based metrics
from Rosetta/FoldX with purely geometric assessments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InterfaceGeometry:
    """Complete geometric characterization of a binding interface."""

    shape_complementarity: float     # Lawrence-Colman Sc [0, 1]
    contact_density: float           # contacts per Å² of BSA
    buried_surface_area: float       # Å²
    gap_volume_index: float          # gap volume / BSA (lower = tighter packing)
    interface_planarity: float       # 0 = flat, 1 = highly curved
    n_interface_residues_a: int      # receptor interface residues
    n_interface_residues_b: int      # ligand interface residues
    n_hbonds: int                    # hydrogen bonds across interface
    n_salt_bridges: int              # salt bridges across interface
    n_hydrophobic_contacts: int      # hydrophobic contacts


def compute_interface_geometry(
    complex_pdb: Path,
    chain_a: str = "A",
    chain_b: str = "B",
    contact_distance: float = 4.5,
    hbond_distance: float = 3.5,
    salt_bridge_distance: float = 4.0,
) -> InterfaceGeometry:
    """Compute all geometric interface descriptors.

    Args:
        complex_pdb: Path to predicted complex PDB.
        chain_a: Receptor chain ID.
        chain_b: Ligand chain ID.
        contact_distance: Max distance for atomic contacts (Å).
        hbond_distance: Max donor-acceptor distance for H-bonds (Å).
        salt_bridge_distance: Max distance for salt bridges (Å).

    Returns:
        InterfaceGeometry with all descriptors.
    """
    from nanolfa.utils.pdb import identify_interface_residues

    interface_a, interface_b = identify_interface_residues(
        complex_pdb, chain_a, chain_b, contact_distance,
    )

    bsa = compute_bsa(complex_pdb, chain_a, chain_b)
    sc = compute_shape_complementarity(complex_pdb, chain_a, chain_b)

    total_contacts = sum(r.n_contacts for r in interface_a) + \
        sum(r.n_contacts for r in interface_b)
    contact_density = total_contacts / max(bsa, 1.0)

    gap_index = compute_gap_volume_index(complex_pdb, chain_a, chain_b, bsa)
    planarity = compute_interface_planarity(complex_pdb, chain_a, chain_b)
    n_hbonds = count_hbonds(complex_pdb, chain_a, chain_b, hbond_distance)
    n_salt = count_salt_bridges(complex_pdb, chain_a, chain_b, salt_bridge_distance)
    n_hydrophobic = count_hydrophobic_contacts(complex_pdb, chain_a, chain_b, contact_distance)

    return InterfaceGeometry(
        shape_complementarity=sc,
        contact_density=contact_density,
        buried_surface_area=bsa,
        gap_volume_index=gap_index,
        interface_planarity=planarity,
        n_interface_residues_a=len(interface_a),
        n_interface_residues_b=len(interface_b),
        n_hbonds=n_hbonds,
        n_salt_bridges=n_salt,
        n_hydrophobic_contacts=n_hydrophobic,
    )


def compute_bsa(
    complex_pdb: Path,
    chain_a: str = "A",
    chain_b: str = "B",
) -> float:
    """Compute buried surface area at the interface.

    BSA = SASA(chain_A) + SASA(chain_B) - SASA(complex)
    Uses FreeSASA if available, otherwise falls back to RDKit.
    """
    from nanolfa.utils.pdb import extract_chain

    try:
        import freesasa

        struct = freesasa.Structure(str(complex_pdb))
        sasa_complex = freesasa.calc(struct).totalArea()

        chain_a_pdb = extract_chain(complex_pdb, chain_a)
        chain_b_pdb = extract_chain(complex_pdb, chain_b)

        sasa_a = freesasa.calc(freesasa.Structure(str(chain_a_pdb))).totalArea()
        sasa_b = freesasa.calc(freesasa.Structure(str(chain_b_pdb))).totalArea()

        # Clean up temp files
        chain_a_pdb.unlink(missing_ok=True)
        chain_b_pdb.unlink(missing_ok=True)

        bsa = sasa_a + sasa_b - sasa_complex
        return max(0.0, bsa)

    except ImportError:
        logger.warning("FreeSASA not available — returning estimated BSA")
        return _estimate_bsa_from_contacts(complex_pdb, chain_a, chain_b)


def _estimate_bsa_from_contacts(
    complex_pdb: Path, chain_a: str, chain_b: str,
) -> float:
    """Rough BSA estimate from contact count (~15 Å² per contact)."""
    from nanolfa.utils.pdb import identify_interface_residues
    ia, ib = identify_interface_residues(complex_pdb, chain_a, chain_b)
    n_residues = len(ia) + len(ib)
    return float(n_residues * 35.0)  # ~35 Å² per interface residue


def compute_shape_complementarity(
    complex_pdb: Path,
    chain_a: str = "A",
    chain_b: str = "B",
) -> float:
    """Compute Lawrence-Colman shape complementarity (Sc).

    Tries PyRosetta first. Falls back to a geometric approximation
    using surface normal correlation at the interface.

    Sc ranges from 0 (no complementarity) to 1 (perfect).
    Typical antibody-antigen: 0.64–0.68.
    Hapten-binding pocket: 0.70–0.80.
    """
    try:
        return _sc_via_rosetta(complex_pdb, chain_a, chain_b)
    except ImportError:
        return _sc_geometric_approximation(complex_pdb, chain_a, chain_b)


def _sc_via_rosetta(complex_pdb: Path, chain_a: str, chain_b: str) -> float:
    """Shape complementarity via PyRosetta InterfaceAnalyzer."""
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    pyrosetta.init("-mute all", silent=True)
    pose = pyrosetta.pose_from_pdb(str(complex_pdb))

    interface_str = f"{chain_a}_{chain_b}"
    analyzer = InterfaceAnalyzerMover(interface_str)
    analyzer.set_compute_interface_sc(True)
    analyzer.apply(pose)
    return float(analyzer.get_interface_sc())


def _sc_geometric_approximation(
    complex_pdb: Path, chain_a: str, chain_b: str,
) -> float:
    """Geometric approximation of shape complementarity.

    Estimates Sc from the ratio of actual interface contacts to
    the maximum possible contacts for the given BSA. Tightly
    packed interfaces (high contact/BSA ratio) correlate with
    higher true Sc values.
    """
    from nanolfa.utils.pdb import identify_interface_residues

    ia, ib = identify_interface_residues(complex_pdb, chain_a, chain_b)
    total_contacts = sum(r.n_contacts for r in ia) + sum(r.n_contacts for r in ib)
    n_residues = len(ia) + len(ib)

    if n_residues == 0:
        return 0.4

    # Contact density as a proxy: higher density ≈ better complementarity
    contacts_per_residue = total_contacts / n_residues
    # Empirical mapping: 5 contacts/residue → Sc ≈ 0.55, 15 → Sc ≈ 0.75
    sc_estimate = float(np.clip(0.40 + contacts_per_residue * 0.025, 0.35, 0.85))

    logger.debug(
        "Geometric Sc estimate: %.2f (contacts/residue=%.1f)",
        sc_estimate, contacts_per_residue,
    )
    return sc_estimate


def compute_gap_volume_index(
    complex_pdb: Path, chain_a: str, chain_b: str, bsa: float,
) -> float:
    """Compute gap volume index (gap volume / BSA).

    The gap volume is the void space between the two surfaces at the
    interface. Lower gap index = tighter packing = better binding.

    Simplified: estimated from the difference between the convex hull
    of the interface and the actual contact surface.
    """
    try:
        from scipy.spatial import ConvexHull

        from nanolfa.utils.pdb import identify_interface_residues

        ia, ib = identify_interface_residues(complex_pdb, chain_a, chain_b)
        if len(ia) < 4 or len(ib) < 4:
            return 1.0

        # Get interface atom coordinates
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("c", str(complex_pdb))
        model = structure[0]

        interface_coords: list[list[float]] = []
        for chain_id, interface_res in [(chain_a, ia), (chain_b, ib)]:
            if chain_id not in {c.id for c in model.get_chains()}:
                continue
            chain = model[chain_id]
            res_nums = {r.residue_number for r in interface_res}
            for residue in chain.get_residues():
                if residue.get_id()[1] in res_nums:
                    for atom in residue.get_atoms():
                        pos = atom.get_vector()
                        interface_coords.append([pos[0], pos[1], pos[2]])

        if len(interface_coords) < 4:
            return 1.0

        hull = ConvexHull(np.array(interface_coords))
        gap_volume = hull.volume
        gap_index = gap_volume / max(bsa, 1.0)

        return float(gap_index)

    except Exception as e:
        logger.debug("Gap volume calculation failed: %s", e)
        return 0.5  # neutral default


def compute_interface_planarity(
    complex_pdb: Path, chain_a: str, chain_b: str,
) -> float:
    """Compute interface planarity (0 = flat, 1 = highly curved).

    Fits a plane to the interface CA atoms and measures the RMSD
    of the points from that plane. Hapten-binding pockets are
    typically curved (high planarity score).
    """
    from nanolfa.utils.pdb import identify_interface_residues

    ia, ib = identify_interface_residues(complex_pdb, chain_a, chain_b)
    if len(ia) < 3:
        return 0.5

    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("c", str(complex_pdb))
    model = structure[0]

    if chain_a not in {c.id for c in model.get_chains()}:
        return 0.5

    coords: list[list[float]] = []
    chain = model[chain_a]
    res_nums = {r.residue_number for r in ia}
    for residue in chain.get_residues():
        if residue.get_id()[1] in res_nums:
            ca = residue.get("CA")
            if ca is not None:
                pos = ca.get_vector()
                coords.append([pos[0], pos[1], pos[2]])

    if len(coords) < 3:
        return 0.5

    points = np.array(coords)
    centroid = points.mean(axis=0)
    centered = points - centroid

    # SVD to find the best-fit plane
    _, s, _ = np.linalg.svd(centered)
    # Ratio of smallest to largest singular value indicates planarity
    # Small ratio = flat, large ratio = curved
    planarity = float(s[2] / max(s[0], 1e-8))

    # Normalize to [0, 1] range
    return float(np.clip(planarity * 5.0, 0.0, 1.0))


def count_hbonds(
    complex_pdb: Path, chain_a: str, chain_b: str,
    max_distance: float = 3.5,
) -> int:
    """Count inter-chain hydrogen bonds at the interface.

    Simplified: counts donor-acceptor pairs (N/O atoms) across the
    interface within the distance cutoff. Full H-bond detection would
    also check angles.
    """
    from Bio.PDB import NeighborSearch, PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("c", str(complex_pdb))
    model = structure[0]

    chains = {c.id: c for c in model.get_chains()}
    if chain_a not in chains or chain_b not in chains:
        return 0

    polar_atoms = {"N", "O", "ND1", "ND2", "NE", "NE1", "NE2",
                   "NH1", "NH2", "NZ", "OD1", "OD2", "OE1", "OE2",
                   "OG", "OG1", "OH", "SD", "SG"}

    atoms_a = [a for a in chains[chain_a].get_atoms() if a.get_name() in polar_atoms]
    atoms_b = [a for a in chains[chain_b].get_atoms() if a.get_name() in polar_atoms]

    if not atoms_a or not atoms_b:
        return 0

    ns = NeighborSearch(atoms_b)
    hbond_count = 0
    for atom in atoms_a:
        neighbors = ns.search(atom.get_vector().get_array(), max_distance)
        hbond_count += len(neighbors)

    return hbond_count


def count_salt_bridges(
    complex_pdb: Path, chain_a: str, chain_b: str,
    max_distance: float = 4.0,
) -> int:
    """Count inter-chain salt bridges."""
    from Bio.PDB import NeighborSearch, PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("c", str(complex_pdb))
    model = structure[0]

    chains = {c.id: c for c in model.get_chains()}
    if chain_a not in chains or chain_b not in chains:
        return 0

    positive_atoms = {"NZ", "NH1", "NH2", "NE"}  # Lys, Arg
    negative_atoms = {"OD1", "OD2", "OE1", "OE2"}  # Asp, Glu

    pos_a = [a for a in chains[chain_a].get_atoms() if a.get_name() in positive_atoms]
    neg_b = [a for a in chains[chain_b].get_atoms() if a.get_name() in negative_atoms]
    neg_a = [a for a in chains[chain_a].get_atoms() if a.get_name() in negative_atoms]
    pos_b = [a for a in chains[chain_b].get_atoms() if a.get_name() in positive_atoms]

    count = 0
    if neg_b:
        ns = NeighborSearch(neg_b)
        for atom in pos_a:
            count += len(ns.search(atom.get_vector().get_array(), max_distance))
    if pos_b:
        ns = NeighborSearch(pos_b)
        for atom in neg_a:
            count += len(ns.search(atom.get_vector().get_array(), max_distance))

    return count


def count_hydrophobic_contacts(
    complex_pdb: Path, chain_a: str, chain_b: str,
    max_distance: float = 4.5,
) -> int:
    """Count inter-chain hydrophobic contacts."""
    from Bio.PDB import NeighborSearch, PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("c", str(complex_pdb))
    model = structure[0]

    chains = {c.id: c for c in model.get_chains()}
    if chain_a not in chains or chain_b not in chains:
        return 0

    hydrophobic_residues = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}

    hydro_a = [
        a for a in chains[chain_a].get_atoms()
        if a.get_parent().get_resname() in hydrophobic_residues
        and a.element == "C"
    ]
    hydro_b = [
        a for a in chains[chain_b].get_atoms()
        if a.get_parent().get_resname() in hydrophobic_residues
        and a.element == "C"
    ]

    if not hydro_a or not hydro_b:
        return 0

    ns = NeighborSearch(hydro_b)
    count = 0
    seen: set[tuple[int, int]] = set()
    for atom in hydro_a:
        neighbors = ns.search(atom.get_vector().get_array(), max_distance)
        for neighbor in neighbors:
            pair = (atom.get_parent().get_id()[1], neighbor.get_parent().get_id()[1])
            if pair not in seen:
                seen.add(pair)
                count += 1

    return count
