"""RFdiffusion wrapper for CDR3 backbone generation.

Generates diverse CDR3 loop backbone geometries for VHH nanobodies
using RFdiffusion's protein backbone diffusion model. The generated
backbones serve as input for ProteinMPNN sequence design in Phase 2/3.

RFdiffusion designs the CDR3 loop de novo while keeping the framework
residues fixed, producing structurally diverse starting points for the
iterative design loop.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig

if TYPE_CHECKING:
    from nanolfa.utils.sequence import GermlineScaffold

logger = logging.getLogger(__name__)


@dataclass
class RFdiffusionDesign:
    """A single backbone design from RFdiffusion."""

    design_id: str
    pdb_path: Path
    scaffold_name: str
    cdr3_length: int
    score: float  # RFdiffusion self-consistency score (lower = better)
    passed_clash_check: bool = True


class RFdiffusionRunner:
    """Generate CDR3 backbone geometries using RFdiffusion.

    For hapten-binding nanobodies, CDR3 is the primary determinant of
    specificity. This runner generates diverse CDR3 backbone conformations
    on a fixed VHH framework scaffold, producing structurally varied
    starting points for ProteinMPNN sequence design.

    The contig specification tells RFdiffusion which residues to design
    (CDR3 loop) vs. keep fixed (framework + CDR1/CDR2).
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.install_path = Path(config.install_path)
        self.num_designs = config.num_designs
        self.noise_scale = config.noise_scale
        self.diffusion_steps = config.get("diffusion_steps", 50)

    def generate_backbones(
        self,
        scaffold: GermlineScaffold,
        cdr3_length_range: tuple[int, int],
        output_dir: Path,
        target_pdb: Path | None = None,
        hotspot_residues: list[int] | None = None,
    ) -> list[RFdiffusionDesign]:
        """Generate CDR3 backbone designs for a single scaffold.

        Args:
            scaffold: VHH germline scaffold with annotated regions.
            cdr3_length_range: Min/max CDR3 length to design (residues).
            output_dir: Directory for output PDB files.
            target_pdb: Optional target structure for guided diffusion.
            hotspot_residues: Optional residue indices on target to guide
                the CDR3 loop toward.

        Returns:
            List of RFdiffusionDesign objects with backbone PDB paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if scaffold.regions is None:
            raise ValueError(f"Scaffold {scaffold.name} has no annotated regions")

        # Build contig specification
        contig = self._build_contig(scaffold, cdr3_length_range)

        logger.info(
            "RFdiffusion: scaffold=%s, CDR3=%d-%d, n_designs=%d, contig=%s",
            scaffold.name, cdr3_length_range[0], cdr3_length_range[1],
            self.num_designs, contig,
        )

        # Build command
        cmd = [
            "python", str(self.install_path / "scripts" / "run_inference.py"),
            f"inference.output_prefix={output_dir / scaffold.name}",
            f"inference.num_designs={self.num_designs}",
            f"contigmap.contigs=[{contig}]",
            f"denoiser.noise_scale_ca={self.noise_scale}",
            f"denoiser.noise_scale_frame={self.noise_scale}",
            f"diffuser.T={self.diffusion_steps}",
        ]

        # Add target-guided potentials if target PDB is provided
        if target_pdb is not None:
            cmd.append(f"inference.input_pdb={target_pdb}")
            if hotspot_residues:
                hotspot_str = ",".join(f"A{r}" for r in hotspot_residues)
                cmd.append(f"ppi.hotspot_res=[{hotspot_str}]")

        logger.debug("RFdiffusion command: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600,
            )
            if result.returncode != 0:
                logger.error("RFdiffusion failed: %s", result.stderr[:500])
                raise RuntimeError(f"RFdiffusion failed: {result.stderr[:500]}")
        except FileNotFoundError:
            logger.error(
                "RFdiffusion not found at %s. Install from "
                "https://github.com/RosettaCommons/RFdiffusion",
                self.install_path,
            )
            raise

        # Parse output PDBs
        designs = self._parse_outputs(output_dir, scaffold.name, cdr3_length_range)

        logger.info(
            "Generated %d backbone designs for scaffold %s",
            len(designs), scaffold.name,
        )
        return designs

    def generate_for_library(
        self,
        scaffolds: list[GermlineScaffold],
        cdr3_length_range: tuple[int, int],
        output_dir: Path,
        target_pdb: Path | None = None,
    ) -> list[RFdiffusionDesign]:
        """Generate CDR3 backbones for an entire scaffold library.

        Args:
            scaffolds: List of curated VHH scaffolds.
            cdr3_length_range: CDR3 length range.
            output_dir: Root output directory.
            target_pdb: Optional target for guided diffusion.

        Returns:
            All designs across all scaffolds.
        """
        all_designs: list[RFdiffusionDesign] = []

        for scaffold in scaffolds:
            scaffold_dir = output_dir / scaffold.name
            try:
                designs = self.generate_backbones(
                    scaffold=scaffold,
                    cdr3_length_range=cdr3_length_range,
                    output_dir=scaffold_dir,
                    target_pdb=target_pdb,
                )
                all_designs.extend(designs)
            except Exception as e:
                logger.error(
                    "RFdiffusion failed for scaffold %s: %s", scaffold.name, e
                )

        logger.info(
            "Total: %d backbone designs from %d scaffolds",
            len(all_designs), len(scaffolds),
        )
        return all_designs

    def _build_contig(
        self,
        scaffold: GermlineScaffold,
        cdr3_length_range: tuple[int, int],
    ) -> str:
        """Build the RFdiffusion contig specification.

        The contig defines which residues are fixed (framework + CDR1/2)
        and which are designed de novo (CDR3).

        Format: "A1-104/0 {min}-{max}/0 A118-{end}"
        - A1-104: fixed (FR1 through FR3, includes CDR1/CDR2)
        - {min}-{max}: designed CDR3 loop (variable length)
        - A118-{end}: fixed FR4

        For more granular control, CDR1 and CDR2 can also be designed
        in later rounds.
        """
        regions = scaffold.regions
        if regions is None:
            raise ValueError("Scaffold must have annotated regions")

        seq_len = len(scaffold.sequence)

        # Fixed: FR1 + CDR1 + FR2 + CDR2 + FR3 (positions 1-104 in IMGT)
        pre_cdr3_len = len(regions.fr1) + len(regions.cdr1) + len(regions.fr2) + \
            len(regions.cdr2) + len(regions.fr3)

        # Fixed: FR4
        fr4_len = len(regions.fr4)
        fr4_start = seq_len - fr4_len + 1

        cdr3_min, cdr3_max = cdr3_length_range

        contig = f"A1-{pre_cdr3_len}/0 {cdr3_min}-{cdr3_max}/0 A{fr4_start}-{seq_len}"
        return contig

    def _parse_outputs(
        self,
        output_dir: Path,
        scaffold_name: str,
        cdr3_length_range: tuple[int, int],
    ) -> list[RFdiffusionDesign]:
        """Parse RFdiffusion output PDB files."""
        designs: list[RFdiffusionDesign] = []

        for pdb_path in sorted(output_dir.glob(f"{scaffold_name}_*.pdb")):
            design_id = pdb_path.stem

            # Try to determine CDR3 length from the PDB
            cdr3_len = self._count_cdr3_residues(pdb_path, cdr3_length_range)

            # Check for self-consistency score in the trb file
            trb_path = pdb_path.with_suffix(".trb")
            score = self._extract_score(trb_path)

            designs.append(
                RFdiffusionDesign(
                    design_id=design_id,
                    pdb_path=pdb_path,
                    scaffold_name=scaffold_name,
                    cdr3_length=cdr3_len,
                    score=score,
                )
            )

        return designs

    @staticmethod
    def _count_cdr3_residues(
        pdb_path: Path, cdr3_range: tuple[int, int]
    ) -> int:
        """Count residues in the designed CDR3 region from the output PDB."""
        try:
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("design", str(pdb_path))
            residues = list(structure[0].get_residues())
            total = len(residues)
            # CDR3 is the variable-length region; estimate from total
            # minus known fixed framework lengths
            fixed_len = 104 + 11  # FR1-FR3 + FR4 in IMGT
            cdr3_len = max(0, total - fixed_len)
            return cdr3_len
        except Exception:
            # Return midpoint of range as fallback
            return (cdr3_range[0] + cdr3_range[1]) // 2

    @staticmethod
    def _extract_score(trb_path: Path) -> float:
        """Extract self-consistency score from RFdiffusion .trb file."""
        if not trb_path.exists():
            return 0.0
        try:
            import numpy as np
            trb = np.load(str(trb_path), allow_pickle=True)
            # The trb file contains various metrics; look for plddt or score
            if "plddt" in trb:
                return float(np.mean(trb["plddt"]))
            return 0.0
        except Exception:
            return 0.0
