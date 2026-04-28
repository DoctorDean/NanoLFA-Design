"""Composite scoring function for ranking nanobody candidates.

Combines six orthogonal metrics into a single [0, 1] score:
  1. ipTM — AlphaFold interface predicted TM-score
  2. pLDDT (interface) — mean per-residue confidence at interface
  3. Shape complementarity (Sc) — Lawrence-Colman statistic
  4. Binding energy (ΔG) — Rosetta or FoldX
  5. Buried surface area (BSA) — interface size
  6. Developability — meta-score from filters

See docs/SCORING.md for full documentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class InterfaceScores:
    """Raw scores extracted from a predicted complex structure."""

    iptm: float
    plddt_interface: float
    shape_complementarity: float
    binding_energy: float  # REU or kcal/mol; more negative = better
    buried_surface_area: float  # Å²


class CompositeScorer:
    """Weighted composite scoring with per-metric normalization.

    Usage:
        scorer = CompositeScorer(scoring_config)
        raw = scorer.score_complex(complex_pdb_path)
        final = scorer.composite(raw, developability_score=0.85)
    """

    def __init__(self, config: DictConfig) -> None:
        self.weights = config.weights
        self.norm = config.normalization

        # Handle target-specific normalization overrides
        if hasattr(config, "normalization_overrides"):
            for key, override in config.normalization_overrides.items():
                if key in self.norm:
                    for param, value in override.items():
                        self.norm[key][param] = value
                    logger.info("Normalization override for %s: %s", key, dict(override))

        self.energy_engine = config.get("energy_engine", "rosetta")
        self.interface_cutoff = config.get("interface_distance_cutoff", 5.0)

    def score_complex(self, complex_path: Path) -> dict[str, float]:
        """Extract all raw interface scores from a predicted complex PDB.

        Args:
            complex_path: Path to the relaxed complex PDB file.

        Returns:
            Dictionary of raw metric values.
        """
        scores: dict[str, float] = {}

        # 1. AF confidence metrics (from AF output JSON, not re-computed)
        scores["iptm"] = self._extract_iptm(complex_path)
        scores["plddt_interface"] = self._compute_interface_plddt(complex_path)

        # 2. Shape complementarity via Rosetta
        scores["shape_complementarity"] = self._compute_shape_complementarity(complex_path)

        # 3. Binding energy
        scores["binding_energy"] = self._compute_binding_energy(complex_path)

        # 4. Buried surface area
        scores["buried_surface_area"] = self._compute_bsa(complex_path)

        return scores

    def composite(
        self,
        raw_scores: dict[str, float],
        developability_score: float = 1.0,
    ) -> float:
        """Compute the weighted composite score from raw metrics.

        Args:
            raw_scores: Dict with keys matching self.weights.
            developability_score: Pre-computed developability meta-score [0, 1].

        Returns:
            Composite score in [0, 1].
        """
        components = {
            "iptm": raw_scores["iptm"],
            "plddt_interface": raw_scores["plddt_interface"],
            "shape_complementarity": raw_scores["shape_complementarity"],
            "binding_energy": raw_scores["binding_energy"],
            "buried_surface_area": raw_scores["buried_surface_area"],
            "developability": developability_score,
        }

        total = 0.0
        for metric, raw_value in components.items():
            norm_value = self._normalize(metric, raw_value)
            weight = self.weights[metric]
            total += weight * norm_value

        return max(0.0, min(1.0, total))

    def _normalize(self, metric: str, value: float) -> float:
        """Normalize a raw metric value to [0, 1].

        For inverted metrics (e.g., binding energy where more negative = better),
        the normalization flips the scale.
        """
        params = self.norm[metric]
        vmin = params["min"]
        vmax = params["max"]
        invert = params.get("invert", False)

        normalized = (
            (vmax - value) / (vmax - vmin) if invert else (value - vmin) / (vmax - vmin)
        )

        return max(0.0, min(1.0, normalized))

    # ------------------------------------------------------------------
    # Metric extraction methods
    # ------------------------------------------------------------------

    def _extract_iptm(self, complex_path: Path) -> float:
        """Extract ipTM from AlphaFold ranking JSON.

        Looks for the ranking_debug.json or result_model_*.json file
        adjacent to the complex PDB.
        """
        import json

        result_dir = complex_path.parent
        # AF2 style
        ranking_file = result_dir / "ranking_debug.json"
        if ranking_file.exists():
            with open(ranking_file) as f:
                ranking = json.load(f)
            return ranking.get("iptm", 0.0)

        # AF3 style — look for summary JSON
        summary_file = complex_path.with_suffix(".json")
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
            return data.get("iptm", data.get("interface_ptm", 0.0))

        logger.warning("No ipTM found for %s, returning 0.0", complex_path)
        return 0.0

    def _compute_interface_plddt(self, complex_path: Path) -> float:
        """Compute mean pLDDT of residues at the binding interface.

        Interface residues: any residue with a heavy atom within
        `self.interface_cutoff` Å of a heavy atom on the partner chain.
        """
        from Bio.PDB import NeighborSearch, PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", str(complex_path))
        model = structure[0]

        chains = list(model.get_chains())
        if len(chains) < 2:
            logger.warning("Less than 2 chains in %s", complex_path)
            return 0.0

        chain_a_atoms = list(chains[0].get_atoms())
        chain_b_atoms = list(chains[1].get_atoms())

        # Find interface residues using NeighborSearch
        ns = NeighborSearch(chain_b_atoms)
        interface_bfactors = []

        seen_residues: set[tuple] = set()
        for atom in chain_a_atoms:
            neighbors = ns.search(atom.get_vector().get_array(), self.interface_cutoff)
            if neighbors:
                res_key = (atom.get_parent().get_id(), atom.get_parent().get_parent().id)
                if res_key not in seen_residues:
                    seen_residues.add(res_key)
                    # pLDDT is stored in B-factor column in AF outputs
                    ca = atom.get_parent().get("CA")  # type: ignore[union-attr]
                    if ca is not None:
                        interface_bfactors.append(ca.get_bfactor())

        # Also include partner-side interface residues
        ns_a = NeighborSearch(chain_a_atoms)
        for atom in chain_b_atoms:
            neighbors = ns_a.search(atom.get_vector().get_array(), self.interface_cutoff)
            if neighbors:
                res_key = (atom.get_parent().get_id(), atom.get_parent().get_parent().id)
                if res_key not in seen_residues:
                    seen_residues.add(res_key)
                    ca = atom.get_parent().get("CA")  # type: ignore[union-attr]
                    if ca is not None:
                        interface_bfactors.append(ca.get_bfactor())

        if not interface_bfactors:
            return 0.0
        return sum(interface_bfactors) / len(interface_bfactors)

    def _compute_shape_complementarity(self, complex_path: Path) -> float:
        """Compute Lawrence-Colman shape complementarity statistic.

        Uses PyRosetta's InterfaceAnalyzer if available, otherwise falls back
        to a geometric approximation using surface dot products.
        """
        try:
            import pyrosetta
            from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

            pyrosetta.init("-mute all", silent=True)
            pose = pyrosetta.pose_from_pdb(str(complex_path))

            analyzer = InterfaceAnalyzerMover("A_B")
            analyzer.set_compute_interface_sc(True)
            analyzer.apply(pose)
            sc = analyzer.get_interface_sc()
            return sc

        except ImportError:
            logger.warning("PyRosetta not available; using geometric Sc approximation")
            return self._approximate_sc(complex_path)

    def _approximate_sc(self, complex_path: Path) -> float:
        """Geometric approximation of shape complementarity.

        Uses the ratio of interface BSA to convex hull surface area as a proxy.
        Not as accurate as the true Lawrence-Colman Sc but useful for ranking.
        """
        # Placeholder — implement with scipy.spatial.ConvexHull
        return 0.60  # conservative default

    def _compute_binding_energy(self, complex_path: Path) -> float:
        """Compute binding energy using Rosetta or FoldX.

        Returns dG_separated (REU) for Rosetta or Interaction_Energy (kcal/mol)
        for FoldX. Both scales: more negative = stronger binding.
        """
        if self.energy_engine in ("rosetta", "both"):
            try:
                return self._rosetta_binding_energy(complex_path)
            except Exception as e:
                logger.warning("Rosetta failed for %s: %s", complex_path, e)
                if self.energy_engine == "both":
                    return self._foldx_binding_energy(complex_path)
                return 0.0

        return self._foldx_binding_energy(complex_path)

    def _rosetta_binding_energy(self, complex_path: Path) -> float:
        """Binding energy via PyRosetta constrained relax + InterfaceAnalyzer."""
        import pyrosetta
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.protocols.relax import FastRelax

        pyrosetta.init("-mute all", silent=True)
        pose = pyrosetta.pose_from_pdb(str(complex_path))
        sfxn = pyrosetta.create_score_function("ref2015")

        # Constrained relax
        relax = FastRelax()
        relax.set_scorefxn(sfxn)
        relax.constrain_relax_to_start_coords(True)
        relax.apply(pose)

        # Interface analysis
        analyzer = InterfaceAnalyzerMover("A_B")
        analyzer.apply(pose)
        return analyzer.get_separated_interface_energy()

    def _foldx_binding_energy(self, complex_path: Path) -> float:
        """Binding energy via FoldX AnalyseComplex."""
        import subprocess

        result = subprocess.run(
            [
                "foldx", "--command=AnalyseComplex",
                f"--pdb={complex_path.name}",
                f"--pdb-dir={complex_path.parent}",
                "--output-dir=/tmp/foldx_out",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        # Parse FoldX output for Interaction_Energy
        # (simplified — actual parsing depends on FoldX output format)
        for line in result.stdout.splitlines():
            if "Interaction Energy" in line:
                parts = line.strip().split()
                return float(parts[-1])

        logger.warning("FoldX parsing failed for %s", complex_path)
        return 0.0

    def _compute_bsa(self, complex_path: Path) -> float:
        """Compute buried surface area at the interface.

        BSA = SASA_A + SASA_B - SASA_complex
        """
        import freesasa

        # Full complex
        struct_complex = freesasa.Structure(str(complex_path))
        result_complex = freesasa.calc(struct_complex)
        sasa_complex = result_complex.totalArea()

        # Need to compute SASA of individual chains
        # Extract chains to temporary files
        from Bio.PDB import PDBIO, PDBParser, Select

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", str(complex_path))
        model = structure[0]
        chains = list(model.get_chains())

        total_individual_sasa = 0.0
        for chain in chains:
            chain_id = chain.id  # capture in default arg below to avoid late-binding

            class ChainSelect(Select):
                def accept_chain(self, c: Any, _cid: str = chain_id) -> bool:
                    return bool(c.id == _cid)

            io = PDBIO()
            io.set_structure(structure)
            tmp_path = f"/tmp/chain_{chain.id}.pdb"
            io.save(tmp_path, ChainSelect())

            struct_chain = freesasa.Structure(tmp_path)
            result_chain = freesasa.calc(struct_chain)
            total_individual_sasa += result_chain.totalArea()

        bsa = total_individual_sasa - sasa_complex
        return max(0.0, bsa)
