"""ESMFold-based fast prescreening for nanobody candidates.

Provides rapid (no MSA required) structure quality assessment to triage
candidates before committing expensive AlphaFold-Multimer runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omegaconf import DictConfig

if TYPE_CHECKING:
    from nanolfa.core.pipeline import Candidate

logger = logging.getLogger(__name__)


class ESMFoldPrescreen:
    """Fast nanobody fold quality check using ESMFold.

    ESMFold predicts structure from sequence alone (no MSA) in seconds.
    While it cannot predict complexes, it can assess whether a nanobody
    sequence folds into a well-defined immunoglobulin domain. Sequences
    with low overall pLDDT are unlikely to form stable folds and can be
    eliminated before running full AF-Multimer.

    Typical throughput: ~5,000 sequences/hour on a single A100.
    """

    def __init__(self, config: DictConfig) -> None:
        self.enabled = config.enabled
        self.min_plddt = config.min_plddt_threshold
        self.top_k = config.top_k_to_full_af
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load ESMFold model (heavy initialization)."""
        if self._model is not None:
            return

        import torch
        import esm

        logger.info("Loading ESMFold model...")
        self._model = esm.pretrained.esmfold_v1()
        self._model = self._model.eval()

        if torch.cuda.is_available():
            self._model = self._model.cuda()
            logger.info("ESMFold loaded on GPU")
        else:
            logger.warning("ESMFold running on CPU (slow)")

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
            if i % 100 == 0:
                logger.info("ESMFold prescreening: %d/%d", i, len(candidates))

            try:
                plddt = self._predict_plddt(cand.sequence)
                if plddt >= min_plddt:
                    scored.append((plddt, cand))
                else:
                    logger.debug(
                        "Rejected %s: pLDDT=%.1f < %.1f",
                        cand.candidate_id, plddt, min_plddt,
                    )
            except Exception as e:
                logger.warning("ESMFold failed for %s: %s", cand.candidate_id, e)

        # Sort by pLDDT descending, take top-K
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [cand for _, cand in scored[:top_k]]

        logger.info(
            "ESMFold prescreening: %d → %d (min_pLDDT=%.0f, top_k=%d)",
            len(candidates), len(selected), min_plddt, top_k,
        )
        return selected

    def _predict_plddt(self, sequence: str) -> float:
        """Predict mean pLDDT for a single sequence using ESMFold."""
        import torch

        with torch.no_grad():
            output = self._model.infer(sequence)  # type: ignore[union-attr]

        # pLDDT is in the B-factor field of the output, shape [1, L, 1]
        plddt = output["plddt"].squeeze()

        if plddt.dim() == 0:
            return float(plddt.item())
        return float(plddt.mean().item())
