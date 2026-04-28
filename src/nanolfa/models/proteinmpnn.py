"""Wrapper for ProteinMPNN sequence design on nanobody CDR loops.

Handles fixed-backbone sequence design with framework residues frozen and CDR
loops as the design target. Supports amino acid biases to penalize liabilities
and boost residues favorable for hapten binding.
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig

if TYPE_CHECKING:
    from nanolfa.core.pipeline import Candidate

logger = logging.getLogger(__name__)

# IMGT numbering ranges for VHH CDR loops
IMGT_CDR_RANGES = {
    "CDR1": (27, 38),    # IMGT positions 27–38
    "CDR2": (56, 65),    # IMGT positions 56–65
    "CDR3": (105, 117),  # IMGT positions 105–117 (variable length)
}


class ProteinMPNNDesigner:
    """Fixed-backbone sequence design for nanobody CDR loops.

    Uses ProteinMPNN to propose new CDR sequences on a fixed backbone scaffold,
    with framework residues frozen to maintain structural integrity.

    Key features:
    - CDR-only redesign with configurable loop selection
    - Amino acid biases (penalize liabilities, boost favorable residues)
    - Temperature-scheduled sampling for exploration vs. exploitation
    - Round-adaptive strategy (aggressive → conservative)
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.install_path = Path(config.install_path)
        self.weights = config.weights
        self.aa_bias = dict(config.get("aa_bias", {}))

    def diversify(
        self,
        candidates: list[Candidate],
        n_variants: int,
        temperature: float,
        target_cdrs: list[str],
        round_num: int,
    ) -> list[Candidate]:
        """Generate sequence variants for a set of parent candidates.

        Args:
            candidates: Parent candidates (must have structure_path set).
            n_variants: Number of variants to generate per parent.
            temperature: Sampling temperature (lower = more conservative).
            target_cdrs: Which CDR loops to redesign (e.g., ["CDR3", "CDR2"]).
            round_num: Current design round (affects strategy).

        Returns:
            List of new Candidate objects with diversified sequences.
        """
        all_variants: list[Candidate] = []

        for parent in candidates:
            if parent.complex_path is None:
                logger.warning("No structure for %s, skipping", parent.candidate_id)
                continue

            try:
                variants = self._design_sequences(
                    parent=parent,
                    n_sequences=n_variants,
                    temperature=temperature,
                    target_cdrs=target_cdrs,
                )
                all_variants.extend(variants)
            except Exception as e:
                logger.error("MPNN failed for %s: %s", parent.candidate_id, e)

        logger.info(
            "Generated %d variants from %d parents (round %d, T=%.2f, CDRs=%s)",
            len(all_variants), len(candidates), round_num, temperature, target_cdrs,
        )
        return all_variants

    def _design_sequences(
        self,
        parent: Candidate,
        n_sequences: int,
        temperature: float,
        target_cdrs: list[str],
    ) -> list[Candidate]:
        """Run ProteinMPNN on a single parent structure.

        Returns new Candidate objects with designed sequences.
        """
        if parent.complex_path is None:
            raise ValueError(f"Parent {parent.candidate_id} has no complex_path")
        complex_path: Path = parent.complex_path

        work_dir = Path(f"/tmp/mpnn_{parent.candidate_id}_{uuid.uuid4().hex[:8]}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # Determine which positions to fix (framework) vs. design (CDRs)
        fixed_positions = self._compute_fixed_positions(parent, target_cdrs)

        # Write fixed positions JSON
        fixed_pos_jsonl = work_dir / "fixed_positions.jsonl"
        chain_id = "A"
        fixed_dict = {complex_path.stem: {chain_id: fixed_positions}}
        with open(fixed_pos_jsonl, "w") as f:
            f.write(json.dumps(fixed_dict) + "\n")

        # Write amino acid bias JSON
        bias_jsonl = work_dir / "bias_AA.jsonl"
        bias_dict = self._build_aa_bias_dict(parent, target_cdrs)
        with open(bias_jsonl, "w") as f:
            f.write(json.dumps({complex_path.stem: bias_dict}) + "\n")

        # Run ProteinMPNN
        cmd = [
            "python", str(self.install_path / "protein_mpnn_run.py"),
            "--pdb_path", str(complex_path),
            "--out_folder", str(work_dir / "output"),
            "--num_seq_per_target", str(n_sequences),
            "--sampling_temp", str(temperature),
            "--seed", "42",
            "--batch_size", "1",
            "--backbone_noise", str(self.config.backbone_noise),
            "--fixed_positions_jsonl", str(fixed_pos_jsonl),
            "--bias_AA_jsonl", str(bias_jsonl),
        ]

        logger.debug("ProteinMPNN command: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            raise RuntimeError(f"ProteinMPNN failed: {result.stderr[:500]}")

        # Parse output FASTA
        output_fasta = work_dir / "output" / "seqs" / f"{complex_path.stem}.fa"
        variants = self._parse_mpnn_output(output_fasta, parent)

        return variants

    def _compute_fixed_positions(
        self, parent: Candidate, target_cdrs: list[str]
    ) -> list[int]:
        """Compute 1-indexed residue positions to FIX (framework residues).

        Everything NOT in the target CDRs is fixed.
        """
        sequence_length = len(parent.sequence)
        # Get CDR positions to DESIGN
        design_positions: set[int] = set()
        for cdr_name in target_cdrs:
            if cdr_name in IMGT_CDR_RANGES:
                start, end = IMGT_CDR_RANGES[cdr_name]
                # Clamp to sequence length
                design_positions.update(range(start, min(end + 1, sequence_length + 1)))

        # Everything else is fixed
        all_positions = set(range(1, sequence_length + 1))
        fixed = sorted(all_positions - design_positions)
        return fixed

    def _build_aa_bias_dict(
        self, parent: Candidate, target_cdrs: list[str]
    ) -> dict:
        """Build per-position amino acid bias dictionary.

        Applies global biases (penalize Cys, Met, etc.) to all designed positions,
        plus target-specific CDR3 preferences from config.
        """
        bias: dict[str, dict[str, dict[str, float]]] = {}
        chain_id = "A"
        bias[chain_id] = {}

        for cdr_name in target_cdrs:
            if cdr_name in IMGT_CDR_RANGES:
                start, end = IMGT_CDR_RANGES[cdr_name]
                for pos in range(start, min(end + 1, len(parent.sequence) + 1)):
                    pos_bias: dict[str, float] = dict(self.aa_bias)
                    # Add CDR3-specific boosts if configured
                    if cdr_name == "CDR3" and hasattr(self.config, "cdr3_aa_preferences"):
                        for aa, boost in self.config.cdr3_aa_preferences.items():
                            pos_bias[aa] = pos_bias.get(aa, 0.0) + boost
                    bias[chain_id][str(pos)] = pos_bias

        return bias

    def _parse_mpnn_output(
        self, fasta_path: Path, parent: Candidate
    ) -> list[Candidate]:
        """Parse ProteinMPNN output FASTA into Candidate objects."""
        from nanolfa.core.pipeline import Candidate as CandidateClass

        variants: list[Candidate] = []

        if not fasta_path.exists():
            logger.warning("MPNN output not found: %s", fasta_path)
            return variants

        from Bio import SeqIO

        for i, record in enumerate(SeqIO.parse(str(fasta_path), "fasta")):
            seq = str(record.seq)
            # Skip the wild-type (first entry is usually the input)
            if seq == parent.sequence:
                continue

            variant_id = f"{parent.candidate_id}_v{i:03d}"
            mutations = self._identify_mutations(parent.sequence, seq)

            variants.append(
                CandidateClass(
                    candidate_id=variant_id,
                    sequence=seq,
                    round_created=parent.round_created + 1,
                    parent_id=parent.candidate_id,
                    mutations_from_parent=mutations,
                )
            )

        return variants

    @staticmethod
    def _identify_mutations(parent_seq: str, variant_seq: str) -> list[str]:
        """Identify mutations between parent and variant sequences.

        Returns list of strings like ["A45G", "Y102W"].
        """
        mutations = []
        for i, (p, v) in enumerate(zip(parent_seq, variant_seq, strict=False)):
            if p != v:
                mutations.append(f"{p}{i + 1}{v}")
        return mutations
