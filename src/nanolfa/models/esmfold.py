"""ESMFold-based fast prescreening for nanobody candidates.

Provides rapid (no MSA required) structure quality assessment to triage
candidates before committing expensive AlphaFold-Multimer runs.

Enhanced features over basic prescreening:
- Per-region pLDDT analysis (FR1/CDR1/FR2/CDR2/FR3/CDR3/FR4)
- CDR confidence scoring (penalize candidates with floppy CDR loops)
- PDB output for downstream use
- Batch inference with progress tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

if TYPE_CHECKING:
    from nanolfa.core.pipeline import Candidate

logger = logging.getLogger(__name__)


# IMGT region boundaries (0-indexed, for slicing pLDDT arrays)
_REGION_SLICES = {
    "FR1": (0, 26),
    "CDR1": (26, 38),
    "FR2": (38, 55),
    "CDR2": (55, 65),
    "FR3": (65, 104),
    "CDR3": (104, 117),
    "FR4": (117, 128),
}


@dataclass
class ESMFoldResult:
    """Detailed result from ESMFold prediction for a single candidate."""

    candidate_id: str
    sequence: str
    mean_plddt: float
    per_residue_plddt: list[float]

    # Per-region mean pLDDT
    region_plddt: dict[str, float]

    # CDR-specific scores
    cdr1_plddt: float
    cdr2_plddt: float
    cdr3_plddt: float
    mean_cdr_plddt: float

    # Framework stability indicator
    mean_framework_plddt: float

    # Output PDB path (if saved)
    pdb_path: Path | None = None

    @property
    def passes_threshold(self) -> bool:
        """Whether this candidate meets the minimum quality threshold."""
        return self.mean_plddt >= 60.0

    @property
    def cdr_confidence_ratio(self) -> float:
        """Ratio of CDR pLDDT to framework pLDDT.

        Values close to 1.0 indicate CDR loops are as well-defined as the
        framework (good — suggests lock-and-key binding).
        Values << 1.0 indicate floppy CDR loops (may indicate conformational
        selection binding — slower kon, less ideal for LFA).
        """
        if self.mean_framework_plddt == 0:
            return 0.0
        return self.mean_cdr_plddt / self.mean_framework_plddt


class ESMFoldPrescreen:
    """Fast nanobody fold quality check using ESMFold.

    ESMFold predicts structure from sequence alone (no MSA) in seconds.
    While it cannot predict complexes, it can assess whether a nanobody
    sequence folds into a well-defined immunoglobulin domain. Sequences
    with low overall pLDDT are unlikely to form stable folds.

    Enhanced scoring evaluates CDR loop confidence separately from
    framework confidence, since well-ordered CDR loops correlate with
    faster association kinetics (important for LFA).

    Typical throughput: ~5,000 sequences/hour on a single A100.
    """

    def __init__(self, config: DictConfig) -> None:
        self.enabled = config.enabled
        self.min_plddt = config.min_plddt_threshold
        self.top_k = config.top_k_to_full_af
        self._model: Any = None

    def _load_model(self) -> None:
        """Lazy-load ESMFold model (heavy initialization)."""
        if self._model is not None:
            return

        import esm
        import torch

        logger.info("Loading ESMFold model...")
        model = esm.pretrained.esmfold_v1()
        model = model.eval()

        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("ESMFold loaded on GPU")
        else:
            logger.warning("ESMFold running on CPU (slow)")

        self._model = model

    def predict_single(
        self,
        sequence: str,
        candidate_id: str = "unknown",
        save_pdb: Path | None = None,
    ) -> ESMFoldResult:
        """Run ESMFold on a single sequence with detailed analysis.

        Args:
            sequence: Amino acid sequence.
            candidate_id: Identifier for logging.
            save_pdb: If set, write predicted structure to this path.

        Returns:
            ESMFoldResult with per-region pLDDT breakdown.
        """
        self._load_model()
        import torch

        if self._model is None:
            raise RuntimeError("ESMFold model not loaded")

        with torch.no_grad():
            output = self._model.infer(sequence)

        # Extract per-residue pLDDT
        plddt_tensor = output["plddt"].squeeze()
        if plddt_tensor.dim() == 0:
            per_residue = [float(plddt_tensor.item())]
        else:
            per_residue = [float(v) for v in plddt_tensor.cpu().numpy()]

        mean_plddt = sum(per_residue) / len(per_residue) if per_residue else 0.0

        # Per-region analysis
        region_plddt = self._compute_region_plddt(per_residue)

        cdr1_plddt = region_plddt.get("CDR1", 0.0)
        cdr2_plddt = region_plddt.get("CDR2", 0.0)
        cdr3_plddt = region_plddt.get("CDR3", 0.0)

        cdr_values = [cdr1_plddt, cdr2_plddt, cdr3_plddt]
        mean_cdr = sum(cdr_values) / len(cdr_values) if cdr_values else 0.0

        fw_keys = ["FR1", "FR2", "FR3", "FR4"]
        fw_values = [region_plddt.get(k, 0.0) for k in fw_keys if k in region_plddt]
        mean_fw = sum(fw_values) / len(fw_values) if fw_values else 0.0

        # Save PDB if requested
        pdb_path = None
        if save_pdb is not None:
            save_pdb.parent.mkdir(parents=True, exist_ok=True)
            pdb_string = self._model.output_to_pdb(output)[0]
            save_pdb.write_text(pdb_string)
            pdb_path = save_pdb
            logger.debug("Saved PDB: %s", pdb_path)

        return ESMFoldResult(
            candidate_id=candidate_id,
            sequence=sequence,
            mean_plddt=mean_plddt,
            per_residue_plddt=per_residue,
            region_plddt=region_plddt,
            cdr1_plddt=cdr1_plddt,
            cdr2_plddt=cdr2_plddt,
            cdr3_plddt=cdr3_plddt,
            mean_cdr_plddt=mean_cdr,
            mean_framework_plddt=mean_fw,
            pdb_path=pdb_path,
        )

    def filter(
        self,
        candidates: list[Candidate],
        min_plddt: float | None = None,
        top_k: int | None = None,
    ) -> list[Candidate]:
        """Screen candidates by predicted fold quality.

        Args:
            candidates: Nanobody candidates to evaluate.
            min_plddt: Minimum mean pLDDT to pass (default from config).
            top_k: Return at most this many candidates (default from config).

        Returns:
            Filtered and ranked candidates.
        """
        if not self.enabled:
            return candidates

        self._load_model()

        min_plddt = min_plddt or self.min_plddt
        top_k = top_k or self.top_k

        scored: list[tuple[float, Candidate]] = []

        for i, cand in enumerate(candidates):
            if i % 100 == 0 and i > 0:
                logger.info("ESMFold prescreening: %d/%d", i, len(candidates))

            try:
                result = self.predict_single(cand.sequence, cand.candidate_id)
                if result.mean_plddt >= min_plddt:
                    scored.append((result.mean_plddt, cand))
                else:
                    logger.debug(
                        "Rejected %s: pLDDT=%.1f < %.1f",
                        cand.candidate_id, result.mean_plddt, min_plddt,
                    )
            except Exception as e:
                logger.warning("ESMFold failed for %s: %s", cand.candidate_id, e)

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [cand for _, cand in scored[:top_k]]

        logger.info(
            "ESMFold prescreening: %d -> %d (min_pLDDT=%.0f, top_k=%d)",
            len(candidates), len(selected), min_plddt, top_k,
        )
        return selected

    def screen_scaffolds(
        self,
        sequences: list[tuple[str, str]],
        save_dir: Path | None = None,
    ) -> list[ESMFoldResult]:
        """Screen a set of scaffold sequences with detailed per-region analysis.

        Designed for Phase 2 germline scaffold evaluation. Returns full
        ESMFoldResult objects rather than just filtering.

        Args:
            sequences: List of (name, sequence) tuples.
            save_dir: If set, write predicted PDBs to this directory.

        Returns:
            List of ESMFoldResult objects, sorted by mean pLDDT descending.
        """
        self._load_model()

        results: list[ESMFoldResult] = []

        for i, (name, seq) in enumerate(sequences):
            if i % 10 == 0 and i > 0:
                logger.info("Screening scaffolds: %d/%d", i, len(sequences))

            save_path = save_dir / f"{name}.pdb" if save_dir else None

            try:
                result = self.predict_single(seq, candidate_id=name, save_pdb=save_path)
                results.append(result)
            except Exception as e:
                logger.warning("ESMFold failed for scaffold %s: %s", name, e)

        results.sort(key=lambda r: r.mean_plddt, reverse=True)

        logger.info(
            "Screened %d scaffolds. Best: %s (pLDDT=%.1f), Worst: %s (pLDDT=%.1f)",
            len(results),
            results[0].candidate_id if results else "none",
            results[0].mean_plddt if results else 0,
            results[-1].candidate_id if results else "none",
            results[-1].mean_plddt if results else 0,
        )

        return results

    @staticmethod
    def _compute_region_plddt(per_residue: list[float]) -> dict[str, float]:
        """Compute mean pLDDT for each IMGT region."""
        region_plddt: dict[str, float] = {}
        seq_len = len(per_residue)

        for region_name, (start, end) in _REGION_SLICES.items():
            if start >= seq_len:
                continue
            actual_end = min(end, seq_len)
            region_values = per_residue[start:actual_end]
            if region_values:
                region_plddt[region_name] = sum(region_values) / len(region_values)

        return region_plddt
