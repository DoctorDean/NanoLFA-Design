"""Small-molecule chemistry helpers for target preparation.

Provides the molecular infrastructure for Phase 1 of the NanoLFA-Design pipeline:
- 3D conformer generation from SMILES
- Energy minimization (MMFF94 / UFF)
- Hapten-carrier conjugate modeling (linker attachment)
- Solvent-accessible surface area (SASA) epitope mapping
- Pharmacophore feature extraction
- Export to SDF / MOL2 formats for AlphaFold 3 and docking tools

Dependencies:
    - RDKit (rdkit >= 2024.03)
    - FreeSASA (freesasa >= 2.2) — optional, for SASA calculation
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConformerResult:
    """Result of 3D conformer generation and energy minimization."""

    name: str
    smiles: str
    num_conformers_generated: int
    num_conformers_after_pruning: int
    best_conformer_id: int
    best_energy_kcal: float
    mol: Any  # RDKit Mol object (typed as Any to avoid rdkit import at module level)
    output_sdf: Path | None = None
    output_mol2: Path | None = None


@dataclass
class AtomSASA:
    """Per-atom solvent-accessible surface area data."""

    atom_idx: int
    element: str
    sasa: float  # Å²
    is_epitope: bool  # True if SASA > threshold
    atom_name: str = ""
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class PharmacophoreFeature:
    """A pharmacophore feature on the hapten surface."""

    feature_type: str  # "donor", "acceptor", "hydrophobic", "aromatic", "positive", "negative"
    atom_indices: list[int]
    position: tuple[float, float, float]  # centroid of feature atoms
    is_epitope_accessible: bool  # True if any atom in the feature is epitope-exposed


@dataclass
class EpitopeSurfaceMap:
    """Complete epitope surface characterization of a hapten."""

    name: str
    total_sasa: float
    epitope_sasa: float
    epitope_fraction: float
    atom_sasa: list[AtomSASA]
    pharmacophore_features: list[PharmacophoreFeature]
    epitope_atom_indices: list[int]
    buried_atom_indices: list[int]


@dataclass
class TargetPreparationResult:
    """Complete output of Phase 1 for a single target analyte."""

    target_name: str
    analyte: ConformerResult
    cross_reactants: list[ConformerResult] = field(default_factory=list)
    epitope_map: EpitopeSurfaceMap | None = None
    output_dir: Path | None = None


# ---------------------------------------------------------------------------
# Conformer generation
# ---------------------------------------------------------------------------

def generate_conformers(
    smiles: str,
    name: str = "molecule",
    n_conformers: int = 200,
    force_field: str = "MMFF94",
    random_seed: int = 42,
    prune_rms_threshold: float = 0.5,
    optimize: bool = True,
    max_optimize_iterations: int = 1000,
) -> ConformerResult:
    """Generate and optimize 3D conformers from a SMILES string.

    Uses RDKit's ETKDG (v3) conformer generator with the specified force field
    for energy minimization. Selects the lowest-energy conformer as the
    representative structure.

    Args:
        smiles: SMILES string of the molecule.
        name: Human-readable name for logging and output files.
        n_conformers: Number of initial conformers to generate.
        force_field: Force field for minimization ("MMFF94" or "UFF").
        random_seed: Random seed for reproducibility.
        prune_rms_threshold: RMSD threshold (Å) for pruning similar conformers.
        optimize: Whether to energy-minimize conformers.
        max_optimize_iterations: Maximum iterations for optimization.

    Returns:
        ConformerResult with the optimized molecule and metadata.

    Raises:
        ValueError: If SMILES cannot be parsed or embedding fails.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    logger.info("Generating conformers for %s: %s", name, smiles)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES for {name}: {smiles}")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0
    params.pruneRmsThresh = prune_rms_threshold
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    if len(conf_ids) == 0:
        logger.warning("ETKDG failed for %s, retrying with relaxed parameters", name)
        params.useRandomCoords = True
        params.pruneRmsThresh = -1
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
        if len(conf_ids) == 0:
            raise ValueError(f"All conformer embedding attempts failed for {name}")

    n_generated = len(conf_ids)
    logger.info("Generated %d conformers for %s", n_generated, name)

    energies: list[tuple[int, float]] = []

    if optimize:
        if force_field == "MMFF94":
            results = AllChem.MMFFOptimizeMoleculeConfs(
                mol, maxIters=max_optimize_iterations, numThreads=0
            )
            for conf_id, (_, energy) in zip(conf_ids, results, strict=False):
                energies.append((conf_id, energy))
        elif force_field == "UFF":
            results = AllChem.UFFOptimizeMoleculeConfs(
                mol, maxIters=max_optimize_iterations, numThreads=0
            )
            for conf_id, (_, energy) in zip(conf_ids, results, strict=False):
                energies.append((conf_id, energy))
        else:
            raise ValueError(f"Unknown force field: {force_field}")
    else:
        energies = [(cid, 0.0) for cid in conf_ids]

    if not energies:
        raise ValueError(f"No valid conformers after optimization for {name}")

    energies.sort(key=lambda x: x[1])
    best_conf_id, best_energy = energies[0]

    logger.info(
        "%s: %d conformers -> best energy %.2f kcal/mol (conformer %d)",
        name, len(energies), best_energy, best_conf_id,
    )

    return ConformerResult(
        name=name,
        smiles=smiles,
        num_conformers_generated=n_generated,
        num_conformers_after_pruning=len(energies),
        best_conformer_id=best_conf_id,
        best_energy_kcal=best_energy,
        mol=mol,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_molecule(
    result: ConformerResult,
    output_dir: Path,
    formats: tuple[str, ...] = ("sdf", "mol2", "pdb"),
) -> ConformerResult:
    """Export the best conformer to SDF, MOL2, and/or PDB formats.

    Args:
        result: ConformerResult from generate_conformers.
        output_dir: Directory to write output files.
        formats: Tuple of format strings to export.

    Returns:
        Updated ConformerResult with output file paths populated.
    """
    from rdkit import Chem

    output_dir.mkdir(parents=True, exist_ok=True)
    mol = result.mol
    best_id = result.best_conformer_id
    safe_name = result.name.replace(" ", "_").lower()

    if "sdf" in formats:
        sdf_path = output_dir / f"{safe_name}_3d.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol, confId=best_id)
        writer.close()
        result.output_sdf = sdf_path
        logger.info("Exported SDF: %s", sdf_path)

    if "mol2" in formats:
        mol2_path = output_dir / f"{safe_name}_3d.mol2"
        try:
            _export_mol2_via_openbabel(mol, best_id, mol2_path)
            result.output_mol2 = mol2_path
            logger.info("Exported MOL2: %s", mol2_path)
        except ImportError:
            logger.warning(
                "OpenBabel not available — skipping MOL2 export for %s. "
                "Install with: conda install -c conda-forge openbabel",
                result.name,
            )

    if "pdb" in formats:
        pdb_path = output_dir / f"{safe_name}_3d.pdb"
        Chem.MolToPDBFile(mol, str(pdb_path), confId=best_id)
        logger.info("Exported PDB: %s", pdb_path)

    return result


def _export_mol2_via_openbabel(mol: Any, conf_id: int, output_path: Path) -> None:
    """Convert RDKit mol to MOL2 format via OpenBabel."""
    import tempfile

    from rdkit import Chem

    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
        writer = Chem.SDWriter(tmp_path)
        writer.write(mol, confId=conf_id)
        writer.close()

    try:
        from openbabel import openbabel

        conv = openbabel.OBConversion()
        conv.SetInAndOutFormats("sdf", "mol2")
        ob_mol = openbabel.OBMol()
        conv.ReadFile(ob_mol, tmp_path)
        conv.WriteFile(ob_mol, str(output_path))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Molecular descriptors
# ---------------------------------------------------------------------------

def compute_molecular_properties(mol: Any) -> dict[str, float]:
    """Compute key molecular properties relevant to hapten-antibody recognition."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    mol_no_h = Chem.RemoveHs(mol)

    return {
        "mol_weight": float(Descriptors.MolWt(mol_no_h)),
        "logp": float(Descriptors.MolLogP(mol_no_h)),
        "hbd": float(Lipinski.NumHDonors(mol_no_h)),
        "hba": float(Lipinski.NumHAcceptors(mol_no_h)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol_no_h)),
        "tpsa": float(Descriptors.TPSA(mol_no_h)),
        "aromatic_rings": float(Descriptors.NumAromaticRings(mol_no_h)),
        "num_rings": float(Descriptors.RingCount(mol_no_h)),
        "num_heavy_atoms": float(mol_no_h.GetNumHeavyAtoms()),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol_no_h)),
    }


# ---------------------------------------------------------------------------
# Epitope surface mapping
# ---------------------------------------------------------------------------

def compute_sasa(
    mol: Any,
    conf_id: int = 0,
    probe_radius: float = 1.4,
) -> list[AtomSASA]:
    """Compute per-atom SASA using RDKit's Shrake-Rupley implementation.

    Args:
        mol: RDKit Mol with 3D coordinates (including Hs).
        conf_id: Conformer ID to compute SASA for.
        probe_radius: Probe radius in Angstroms (default 1.4 for water).

    Returns:
        List of AtomSASA objects for every heavy atom.
    """
    from rdkit.Chem import rdFreeSASA

    conformer = mol.GetConformer(conf_id)

    radii = rdFreeSASA.classifyAtoms(mol)
    sasa_opts = rdFreeSASA.SASAOpts()
    sasa_opts.probeRadius = probe_radius
    sasa_values = rdFreeSASA.CalcSASA(mol, radii, confIdx=conf_id)

    results: list[AtomSASA] = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        # Skip hydrogens for epitope analysis
        if atom.GetAtomicNum() == 1:
            continue
        pos = conformer.GetAtomPosition(idx)
        results.append(
            AtomSASA(
                atom_idx=idx,
                element=atom.GetSymbol(),
                sasa=float(sasa_values[idx]) if idx < len(sasa_values) else 0.0,
                is_epitope=False,
                atom_name=f"{atom.GetSymbol()}{idx}",
                position=(pos.x, pos.y, pos.z),
            )
        )

    return results


def map_epitope_surface(
    atom_sasa: list[AtomSASA],
    sasa_threshold: float = 5.0,
) -> tuple[list[int], list[int]]:
    """Classify atoms as epitope-accessible or buried based on SASA threshold."""
    epitope: list[int] = []
    buried: list[int] = []

    for atom_data in atom_sasa:
        if atom_data.sasa >= sasa_threshold:
            atom_data.is_epitope = True
            epitope.append(atom_data.atom_idx)
        else:
            buried.append(atom_data.atom_idx)

    logger.info(
        "Epitope mapping: %d exposed, %d buried (threshold=%.1f A^2)",
        len(epitope), len(buried), sasa_threshold,
    )
    return epitope, buried


# ---------------------------------------------------------------------------
# Pharmacophore features
# ---------------------------------------------------------------------------

def extract_pharmacophore_features(
    mol: Any,
    conf_id: int = 0,
    epitope_indices: list[int] | None = None,
) -> list[PharmacophoreFeature]:
    """Extract pharmacophore features from a molecule.

    Identifies H-bond donors, acceptors, hydrophobic regions, aromatic
    rings, and charged groups. Annotates whether each feature is on
    the epitope-exposed surface.
    """
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    epitope_set = set(epitope_indices) if epitope_indices else set()
    features: list[PharmacophoreFeature] = []

    fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)

    family_map = {
        "Donor": "donor",
        "Acceptor": "acceptor",
        "Hydrophobe": "hydrophobic",
        "Aromatic": "aromatic",
        "PosIonizable": "positive",
        "NegIonizable": "negative",
        "LumpedHydrophobe": "hydrophobic",
    }

    mol_feats = factory.GetFeaturesForMol(mol, confId=conf_id)
    for feat in mol_feats:
        family = feat.GetFamily()
        if family not in family_map:
            continue

        atom_ids = list(feat.GetAtomIds())
        pos = feat.GetPos(conf_id)
        is_accessible = any(idx in epitope_set for idx in atom_ids)

        features.append(
            PharmacophoreFeature(
                feature_type=family_map[family],
                atom_indices=atom_ids,
                position=(pos.x, pos.y, pos.z),
                is_epitope_accessible=is_accessible,
            )
        )

    type_counts: dict[str, int] = {}
    for f in features:
        type_counts[f.feature_type] = type_counts.get(f.feature_type, 0) + 1
    logger.info("Pharmacophore features: %s", type_counts)

    return features


# ---------------------------------------------------------------------------
# Linker occlusion modeling
# ---------------------------------------------------------------------------

def identify_linker_attachment_atom(
    mol: Any,
    linker_chemistry: str = "carbodiimide",
) -> int | None:
    """Identify the atom where the linker attaches to the hapten.

    For carbodiimide (EDC) coupling, the linker attaches at the carboxylic
    acid carbon of the glucuronide moiety.
    """
    from rdkit import Chem

    patterns = {
        "carbodiimide": "[CX3](=O)[OX2H1]",  # carboxylic acid
        "maleimide": "[SX2H1]",  # thiol
        "glutaraldehyde": "[NX3H2]",  # primary amine
    }

    smarts = patterns.get(linker_chemistry)
    if smarts is None:
        logger.warning("Unknown linker chemistry: %s", linker_chemistry)
        return None

    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return None

    matches = mol.GetSubstructMatch(pattern)
    if matches:
        return matches[0]

    logger.warning("No attachment point found for %s chemistry", linker_chemistry)
    return None


def model_linker_occlusion(
    atom_sasa: list[AtomSASA],
    mol: Any,
    attachment_atom_idx: int,
    linker_length_atoms: int = 6,
    occlusion_radius: float = 5.0,
) -> list[AtomSASA]:
    """Simulate linker occlusion by reducing SASA near the attachment point.

    When conjugated to a carrier protein, atoms near the linker attachment
    site are partially or fully occluded. Occlusion decays linearly with
    distance from the attachment atom.
    """
    conformer = mol.GetConformer(0)
    attach_pos = conformer.GetAtomPosition(attachment_atom_idx)
    attach_coord = np.array([attach_pos.x, attach_pos.y, attach_pos.z])

    effective_radius = occlusion_radius + linker_length_atoms * 0.5

    for atom_data in atom_sasa:
        atom_coord = np.array(atom_data.position)
        distance = float(np.linalg.norm(atom_coord - attach_coord))

        if distance < effective_radius:
            occlusion_factor = 1.0 - (distance / effective_radius)
            atom_data.sasa *= 1.0 - occlusion_factor * 0.8

    logger.info(
        "Applied linker occlusion: attachment_atom=%d, effective_radius=%.1f A",
        attachment_atom_idx, effective_radius,
    )
    return atom_sasa


# ---------------------------------------------------------------------------
# Full epitope surface map builder
# ---------------------------------------------------------------------------

def build_epitope_surface_map(
    result: ConformerResult,
    linker_chemistry: str = "carbodiimide",
    linker_length_atoms: int = 6,
    sasa_threshold: float = 5.0,
) -> EpitopeSurfaceMap:
    """Build a complete epitope surface characterization.

    Combines SASA calculation, linker occlusion, epitope mapping, and
    pharmacophore extraction into a single result.
    """
    mol = result.mol
    conf_id = result.best_conformer_id

    atom_sasa = compute_sasa(mol, conf_id=conf_id)

    attachment_idx = identify_linker_attachment_atom(mol, linker_chemistry)
    if attachment_idx is not None:
        atom_sasa = model_linker_occlusion(
            atom_sasa, mol, attachment_idx,
            linker_length_atoms=linker_length_atoms,
        )
    else:
        logger.warning("No linker attachment point found — using free-hapten SASA")

    epitope_indices, buried_indices = map_epitope_surface(atom_sasa, sasa_threshold)

    pharm_features = extract_pharmacophore_features(
        mol, conf_id=conf_id, epitope_indices=epitope_indices
    )

    total_sasa = sum(a.sasa for a in atom_sasa)
    epitope_sasa = sum(a.sasa for a in atom_sasa if a.is_epitope)

    return EpitopeSurfaceMap(
        name=result.name,
        total_sasa=total_sasa,
        epitope_sasa=epitope_sasa,
        epitope_fraction=epitope_sasa / max(total_sasa, 0.01),
        atom_sasa=atom_sasa,
        pharmacophore_features=pharm_features,
        epitope_atom_indices=epitope_indices,
        buried_atom_indices=buried_indices,
    )


# ---------------------------------------------------------------------------
# Full target preparation pipeline
# ---------------------------------------------------------------------------

def prepare_target(
    target_config: dict[str, Any],
    output_dir: Path,
    n_conformers: int = 200,
    force_field: str = "MMFF94",
    random_seed: int = 42,
) -> TargetPreparationResult:
    """Run complete Phase 1 target preparation for a single analyte.

    Generates 3D structures, computes epitope surfaces, processes the
    cross-reactivity panel, and saves a JSON summary.
    """
    target_name = target_config["name"]
    smiles = target_config["smiles"]
    linker_chemistry = target_config.get("linker_chemistry", "carbodiimide")
    linker_length = target_config.get("linker_length_atoms", 6)

    target_dir = output_dir / target_name
    structures_dir = target_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 1: Preparing target %s", target_name)
    logger.info("=" * 60)

    # Primary analyte
    analyte_result = generate_conformers(
        smiles=smiles, name=target_name,
        n_conformers=n_conformers, force_field=force_field,
        random_seed=random_seed,
    )
    analyte_result = export_molecule(analyte_result, structures_dir)

    props = compute_molecular_properties(analyte_result.mol)
    logger.info("Molecular properties for %s: %s", target_name, props)

    epitope_map = build_epitope_surface_map(
        analyte_result,
        linker_chemistry=linker_chemistry,
        linker_length_atoms=linker_length,
    )
    logger.info(
        "Epitope surface: %.1f A^2 total, %.1f A^2 exposed (%.0f%%)",
        epitope_map.total_sasa, epitope_map.epitope_sasa,
        epitope_map.epitope_fraction * 100,
    )

    # Cross-reactivity panel
    cross_dir = structures_dir / "cross_reactants"
    cross_dir.mkdir(parents=True, exist_ok=True)
    cross_results: list[ConformerResult] = []
    panel = target_config.get("cross_reactivity_panel", [])

    for compound in panel:
        try:
            cr_result = generate_conformers(
                smiles=compound["smiles"], name=compound["name"],
                n_conformers=n_conformers // 2, force_field=force_field,
                random_seed=random_seed,
            )
            cr_result = export_molecule(cr_result, cross_dir)
            cross_results.append(cr_result)
        except Exception as e:
            logger.error("Failed to prepare cross-reactant %s: %s", compound["name"], e)

    # Save summary
    summary = {
        "target_name": target_name,
        "smiles": smiles,
        "molecular_properties": props,
        "conformer_stats": {
            "n_generated": analyte_result.num_conformers_generated,
            "n_after_pruning": analyte_result.num_conformers_after_pruning,
            "best_energy_kcal": analyte_result.best_energy_kcal,
        },
        "epitope_surface": {
            "total_sasa_A2": round(epitope_map.total_sasa, 1),
            "epitope_sasa_A2": round(epitope_map.epitope_sasa, 1),
            "epitope_fraction": round(epitope_map.epitope_fraction, 3),
            "n_epitope_atoms": len(epitope_map.epitope_atom_indices),
            "n_buried_atoms": len(epitope_map.buried_atom_indices),
        },
        "cross_reactivity_panel": [
            {"name": cr.name, "smiles": cr.smiles, "best_energy_kcal": cr.best_energy_kcal}
            for cr in cross_results
        ],
    }

    summary_path = target_dir / "preparation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    return TargetPreparationResult(
        target_name=target_name, analyte=analyte_result,
        cross_reactants=cross_results, epitope_map=epitope_map,
        output_dir=target_dir,
    )
