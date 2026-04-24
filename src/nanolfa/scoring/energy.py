"""Physics-based energy rescoring (Rosetta/FoldX wrappers).

Standalone module for computing binding free energy estimates from
predicted complex structures. Wraps PyRosetta and FoldX with automatic
fallback to a knowledge-based statistical potential when neither is
available.

The binding energy (ΔG_bind) is one of the six composite scoring
components. More negative = stronger predicted binding.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnergyResult:
    """Result of physics-based binding energy calculation."""

    method: str                          # "rosetta", "foldx", "statistical"
    binding_energy: float                # REU or kcal/mol; more negative = better
    total_energy_complex: float          # total energy of the complex
    total_energy_receptor: float         # receptor alone
    total_energy_ligand: float           # ligand alone
    n_relax_iterations: int
    converged: bool

    @property
    def delta_g(self) -> float:
        """Binding energy (same as binding_energy, for clarity)."""
        return self.binding_energy


def compute_binding_energy(
    complex_pdb: Path,
    chain_a: str = "A",
    chain_b: str = "B",
    engine: str = "auto",
    n_relax: int = 3,
) -> EnergyResult:
    """Compute binding energy using the best available engine.

    Priority: Rosetta > FoldX > statistical potential.

    Args:
        complex_pdb: Path to predicted complex PDB.
        chain_a: Receptor chain ID.
        chain_b: Ligand chain ID.
        engine: "rosetta", "foldx", "statistical", or "auto".
        n_relax: Number of relaxation replicates (Rosetta) or runs (FoldX).

    Returns:
        EnergyResult with binding energy estimate.
    """
    if engine == "auto":
        for method in ["rosetta", "foldx", "statistical"]:
            try:
                return _dispatch(method, complex_pdb, chain_a, chain_b, n_relax)
            except ImportError:
                continue
            except Exception as e:
                logger.warning("%s failed: %s — trying next engine", method, e)
                continue
        return _statistical_energy(complex_pdb, chain_a, chain_b)
    else:
        return _dispatch(engine, complex_pdb, chain_a, chain_b, n_relax)


def _dispatch(
    method: str, pdb: Path, chain_a: str, chain_b: str, n_relax: int,
) -> EnergyResult:
    if method == "rosetta":
        return rosetta_binding_energy(pdb, chain_a, chain_b, n_relax)
    elif method == "foldx":
        return foldx_binding_energy(pdb, chain_a, chain_b, n_relax)
    elif method == "statistical":
        return _statistical_energy(pdb, chain_a, chain_b)
    else:
        raise ValueError(f"Unknown energy engine: {method}")


# ---------------------------------------------------------------------------
# Rosetta
# ---------------------------------------------------------------------------

def rosetta_binding_energy(
    complex_pdb: Path,
    chain_a: str = "A",
    chain_b: str = "B",
    n_relax: int = 3,
    scorefxn_name: str = "ref2015",
) -> EnergyResult:
    """Compute binding energy via PyRosetta constrained relax + InterfaceAnalyzer.

    Protocol:
    1. Load PDB into Rosetta pose
    2. Constrained FastRelax (preserves AF-predicted backbone)
    3. InterfaceAnalyzer: compute dG_separated, packstat, etc.
    4. Average over n_relax replicates for robustness

    Args:
        complex_pdb: PDB path.
        chain_a: Receptor chain.
        chain_b: Ligand chain.
        n_relax: Number of relax replicates.
        scorefxn_name: Rosetta score function name.

    Returns:
        EnergyResult with Rosetta energy units (REU).
    """
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.protocols.relax import FastRelax

    pyrosetta.init("-mute all", silent=True)

    interface_str = f"{chain_a}_{chain_b}"
    sfxn = pyrosetta.create_score_function(scorefxn_name)

    energies: list[float] = []
    total_complex: list[float] = []
    converged_count = 0

    for _i in range(n_relax):
        pose = pyrosetta.pose_from_pdb(str(complex_pdb))

        # Constrained relax
        relax = FastRelax()
        relax.set_scorefxn(sfxn)
        relax.constrain_relax_to_start_coords(True)
        relax.apply(pose)

        # Interface analysis
        analyzer = InterfaceAnalyzerMover(interface_str)
        analyzer.apply(pose)

        dg = analyzer.get_separated_interface_energy()
        energies.append(dg)
        total_complex.append(sfxn(pose))
        converged_count += 1

    mean_dg = float(np.mean(energies))
    mean_total = float(np.mean(total_complex))

    logger.info(
        "Rosetta ΔG: %.1f ± %.1f REU (%d replicates)",
        mean_dg, float(np.std(energies)), n_relax,
    )

    return EnergyResult(
        method="rosetta",
        binding_energy=mean_dg,
        total_energy_complex=mean_total,
        total_energy_receptor=0.0,  # not computed separately
        total_energy_ligand=0.0,
        n_relax_iterations=n_relax,
        converged=converged_count == n_relax,
    )


# ---------------------------------------------------------------------------
# FoldX
# ---------------------------------------------------------------------------

def foldx_binding_energy(
    complex_pdb: Path,
    chain_a: str = "A",
    chain_b: str = "B",
    n_runs: int = 5,
) -> EnergyResult:
    """Compute binding energy via FoldX AnalyseComplex.

    FoldX uses an empirical force field calibrated on experimental
    ΔΔG data. Returns energy in kcal/mol.

    Args:
        complex_pdb: PDB path.
        chain_a: Receptor chain.
        chain_b: Ligand chain.
        n_runs: Number of FoldX runs.

    Returns:
        EnergyResult with kcal/mol energies.
    """
    # First repair the PDB
    repair_cmd = [
        "foldx", "--command=RepairPDB",
        f"--pdb={complex_pdb.name}",
        f"--pdb-dir={complex_pdb.parent}",
        "--output-dir=/tmp/foldx_repair",
    ]

    subprocess.run(repair_cmd, capture_output=True, text=True, timeout=300)

    # Find repaired PDB
    repaired = Path(f"/tmp/foldx_repair/{complex_pdb.stem}_Repair.pdb")
    input_pdb = repaired if repaired.exists() else complex_pdb

    # Analyse complex
    result = subprocess.run(
        [
            "foldx", "--command=AnalyseComplex",
            f"--pdb={input_pdb.name}",
            f"--pdb-dir={input_pdb.parent}",
            "--output-dir=/tmp/foldx_out",
            f"--numberOfRuns={n_runs}",
        ],
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FoldX failed: {result.stderr[:300]}")

    # Parse output
    interaction_energy = _parse_foldx_output(result.stdout, input_pdb.stem)

    return EnergyResult(
        method="foldx",
        binding_energy=interaction_energy,
        total_energy_complex=0.0,
        total_energy_receptor=0.0,
        total_energy_ligand=0.0,
        n_relax_iterations=n_runs,
        converged=True,
    )


def _parse_foldx_output(stdout: str, pdb_stem: str) -> float:
    """Parse FoldX AnalyseComplex output for interaction energy."""
    # Look in stdout and in the output files
    for line in stdout.splitlines():
        if "Interaction Energy" in line:
            parts = line.strip().split()
            for part in reversed(parts):
                try:
                    return float(part)
                except ValueError:
                    continue

    # Try reading from the Interaction file
    interaction_file = Path(f"/tmp/foldx_out/Interaction_{pdb_stem}_AC.fxout")
    if interaction_file.exists():
        with open(interaction_file) as f:
            for line in f:
                if pdb_stem in line:
                    parts = line.strip().split("\t")
                    if len(parts) >= 6:
                        try:
                            return float(parts[5])  # Interaction_Energy column
                        except ValueError:
                            pass

    logger.warning("Could not parse FoldX interaction energy")
    return 0.0


# ---------------------------------------------------------------------------
# Statistical potential fallback
# ---------------------------------------------------------------------------

def _statistical_energy(
    complex_pdb: Path,
    chain_a: str = "A",
    chain_b: str = "B",
) -> EnergyResult:
    """Estimate binding energy from a knowledge-based statistical potential.

    Uses inter-chain contact counts weighted by residue pair preferences
    derived from the PDB. This is much less accurate than Rosetta/FoldX
    but requires no external software.

    The potential is simplified: each inter-chain contact contributes
    approximately -1.5 kJ/mol for hydrophobic contacts, -2.0 for
    hydrogen bonds, and -3.0 for salt bridges.
    """
    from nanolfa.scoring.structural import (
        count_hbonds,
        count_hydrophobic_contacts,
        count_salt_bridges,
    )

    n_hydro = count_hydrophobic_contacts(complex_pdb, chain_a, chain_b)
    n_hbonds = count_hbonds(complex_pdb, chain_a, chain_b)
    n_salt = count_salt_bridges(complex_pdb, chain_a, chain_b)

    # Simplified statistical potential (kJ/mol)
    energy = -1.5 * n_hydro + -2.0 * n_hbonds + -3.0 * n_salt

    # Convert kJ/mol to approximate REU for consistency with Rosetta scale
    # 1 REU ≈ 1 kcal/mol ≈ 4.184 kJ/mol
    energy_reu = energy / 4.184

    logger.info(
        "Statistical ΔG: %.1f REU (hydro=%d, hbond=%d, salt=%d)",
        energy_reu, n_hydro, n_hbonds, n_salt,
    )

    return EnergyResult(
        method="statistical",
        binding_energy=energy_reu,
        total_energy_complex=0.0,
        total_energy_receptor=0.0,
        total_energy_ligand=0.0,
        n_relax_iterations=0,
        converged=True,
    )
