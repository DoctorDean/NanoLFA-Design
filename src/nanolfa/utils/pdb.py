"""PDB I/O utilities for the NanoLFA-Design pipeline.

Common operations on PDB files used across multiple modules:
chain extraction, interface residue identification, B-factor (pLDDT)
extraction, atom coordinate retrieval, and format conversion.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChainInfo:
    """Metadata for a single chain in a PDB file."""

    chain_id: str
    n_residues: int
    n_atoms: int
    sequence: str
    mean_bfactor: float  # mean pLDDT for AF outputs


@dataclass
class InterfaceResidue:
    """A residue at the binding interface between two chains."""

    chain_id: str
    residue_number: int
    residue_name: str  # three-letter code
    bfactor: float  # pLDDT for AF outputs
    n_contacts: int  # number of inter-chain atom contacts
    min_contact_distance: float  # Å


def parse_structure(pdb_path: Path) -> list[ChainInfo]:
    """Parse a PDB file and return chain-level metadata.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        List of ChainInfo objects, one per chain.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", str(pdb_path))
    model = structure[0]

    chains: list[ChainInfo] = []
    aa_map = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }

    for chain in model.get_chains():
        residues = [r for r in chain.get_residues() if r.get_id()[0] == " "]
        atoms = list(chain.get_atoms())
        sequence = "".join(aa_map.get(r.get_resname(), "X") for r in residues)

        bfactors = [a.get_bfactor() for a in atoms if a.get_name() == "CA"]
        mean_bf = float(np.mean(bfactors)) if bfactors else 0.0

        chains.append(ChainInfo(
            chain_id=chain.id,
            n_residues=len(residues),
            n_atoms=len(atoms),
            sequence=sequence,
            mean_bfactor=mean_bf,
        ))

    return chains


def extract_chain(
    pdb_path: Path,
    chain_id: str,
    output_path: Path | None = None,
) -> Path:
    """Extract a single chain from a multi-chain PDB file.

    Args:
        pdb_path: Input PDB path.
        chain_id: Chain ID to extract.
        output_path: Output PDB path. If None, writes to a temp file.

    Returns:
        Path to the extracted chain PDB.
    """
    from Bio.PDB import PDBIO, PDBParser, Select

    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=f"_chain_{chain_id}.pdb"))

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", str(pdb_path))
    target_chain = chain_id

    class ChainSelector(Select):
        def accept_chain(self, chain: object) -> bool:
            return getattr(chain, "id", None) == target_chain

    io = PDBIO()
    io.set_structure(structure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(output_path), ChainSelector())

    return output_path


def identify_interface_residues(
    pdb_path: Path,
    chain_a: str = "A",
    chain_b: str = "B",
    distance_cutoff: float = 5.0,
) -> tuple[list[InterfaceResidue], list[InterfaceResidue]]:
    """Identify residues at the interface between two chains.

    Args:
        pdb_path: Path to the complex PDB.
        chain_a: First chain ID (typically nanobody).
        chain_b: Second chain ID (typically ligand).
        distance_cutoff: Maximum inter-chain distance (Å) to classify as interface.

    Returns:
        Tuple of (chain_a_interface, chain_b_interface) residue lists.
    """
    from Bio.PDB import NeighborSearch, PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(pdb_path))
    model = structure[0]

    chains = {c.id: c for c in model.get_chains()}
    if chain_a not in chains or chain_b not in chains:
        logger.warning("Chains %s/%s not found in %s", chain_a, chain_b, pdb_path)
        return [], []

    atoms_a = list(chains[chain_a].get_atoms())
    atoms_b = list(chains[chain_b].get_atoms())

    def _find_interface(
        query_atoms: list, target_atoms: list, query_chain_id: str,
    ) -> list[InterfaceResidue]:
        ns = NeighborSearch(target_atoms)
        residue_contacts: dict[int, InterfaceResidue] = {}

        for atom in query_atoms:
            neighbors = ns.search(atom.get_vector().get_array(), distance_cutoff)
            if not neighbors:
                continue

            res = atom.get_parent()
            res_num = res.get_id()[1]
            min_dist = min(float(atom - n) for n in neighbors)

            if res_num in residue_contacts:
                entry = residue_contacts[res_num]
                entry.n_contacts += len(neighbors)
                entry.min_contact_distance = min(entry.min_contact_distance, min_dist)
            else:
                ca = res.get("CA")
                bf = ca.get_bfactor() if ca is not None else 0.0
                residue_contacts[res_num] = InterfaceResidue(
                    chain_id=query_chain_id,
                    residue_number=res_num,
                    residue_name=res.get_resname(),
                    bfactor=bf,
                    n_contacts=len(neighbors),
                    min_contact_distance=min_dist,
                )

        return sorted(residue_contacts.values(), key=lambda r: r.residue_number)

    interface_a = _find_interface(atoms_a, atoms_b, chain_a)
    interface_b = _find_interface(atoms_b, atoms_a, chain_b)

    return interface_a, interface_b


def get_atom_coordinates(
    pdb_path: Path,
    chain_id: str | None = None,
    atom_name: str = "CA",
) -> np.ndarray:
    """Extract coordinates of specific atoms from a PDB file.

    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain to extract from. None = all chains.
        atom_name: Atom name to extract (default "CA" for alpha carbons).

    Returns:
        Numpy array of shape (N, 3) with atom coordinates in Å.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", str(pdb_path))
    model = structure[0]

    coords: list[list[float]] = []
    for chain in model.get_chains():
        if chain_id is not None and chain.id != chain_id:
            continue
        for residue in chain.get_residues():
            atom = residue.get(atom_name)
            if atom is not None:
                pos = atom.get_vector()
                coords.append([pos[0], pos[1], pos[2]])

    return np.array(coords) if coords else np.empty((0, 3))


def get_bfactors(
    pdb_path: Path,
    chain_id: str | None = None,
    atom_name: str = "CA",
) -> list[tuple[int, float]]:
    """Extract per-residue B-factors (pLDDT for AF outputs).

    Args:
        pdb_path: PDB file path.
        chain_id: Chain to extract from. None = all chains.
        atom_name: Atom to read B-factor from (default "CA").

    Returns:
        List of (residue_number, bfactor) tuples.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", str(pdb_path))
    model = structure[0]

    bfactors: list[tuple[int, float]] = []
    for chain in model.get_chains():
        if chain_id is not None and chain.id != chain_id:
            continue
        for residue in chain.get_residues():
            if residue.get_id()[0] != " ":
                continue
            atom = residue.get(atom_name)
            if atom is not None:
                bfactors.append((residue.get_id()[1], atom.get_bfactor()))

    return bfactors


def compute_rmsd(
    pdb_a: Path,
    pdb_b: Path,
    chain_id: str | None = None,
    atom_name: str = "CA",
) -> float:
    """Compute RMSD between two PDB structures.

    Structures must have the same number of atoms in the selected chain.
    No alignment is performed — coordinates are compared directly.

    Args:
        pdb_a: First PDB.
        pdb_b: Second PDB.
        chain_id: Chain to compare. None = all chains.
        atom_name: Atom type for comparison.

    Returns:
        RMSD in Å.
    """
    coords_a = get_atom_coordinates(pdb_a, chain_id, atom_name)
    coords_b = get_atom_coordinates(pdb_b, chain_id, atom_name)

    n = min(len(coords_a), len(coords_b))
    if n == 0:
        return 0.0

    diff = coords_a[:n] - coords_b[:n]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
