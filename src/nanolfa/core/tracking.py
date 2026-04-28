"""Weights & Biases experiment tracking for the design pipeline.

Tracks per-round metrics, candidate scores, convergence curves,
and artifacts across the iterative design loop. Provides a live
dashboard for monitoring pipeline runs on HPC clusters.

Falls back gracefully to no-op logging when W&B is not available
or when experiment_tracker is set to "none" in config.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from nanolfa.core.pipeline import Candidate, RoundResult

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracking interface.

    Wraps Weights & Biases (wandb) for metric logging, artifact storage,
    and hyperparameter tracking. Degrades gracefully to local-only logging
    when wandb is unavailable.

    Usage:
        tracker = ExperimentTracker(config)
        tracker.init_run(target="pdg", tags=["phase3", "round1"])
        tracker.log_round(round_result)
        tracker.log_candidates(candidates, round_num=1)
        tracker.finish()
    """

    def __init__(self, config: DictConfig) -> None:
        self.backend = config.pipeline.get("experiment_tracker", "none")
        self.project = config.pipeline.get("wandb_project", "nanolfa-design")
        self._run: Any = None
        self._enabled = False

        if self.backend == "wandb":
            try:
                import wandb
                self._wandb = wandb
                self._enabled = True
                logger.info("W&B tracking enabled (project=%s)", self.project)
            except ImportError:
                logger.warning(
                    "wandb not installed — tracking disabled. "
                    "Install with: pip install wandb"
                )
                self._enabled = False
        else:
            logger.info("Experiment tracking: %s (disabled)", self.backend)

    def init_run(
        self,
        target: str,
        run_name: str | None = None,
        config: DictConfig | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Initialize a new tracking run.

        Args:
            target: Target analyte name (pdg, e3g).
            run_name: Optional custom run name.
            config: Pipeline config to log as hyperparameters.
            tags: Tags for filtering runs in W&B dashboard.
        """
        if not self._enabled:
            return

        run_name = run_name or f"nanolfa_{target}"
        config_dict: dict[str, Any] | None = None
        if config is not None:
            raw = OmegaConf.to_container(config, resolve=True)
            config_dict = dict(raw) if isinstance(raw, dict) else None  # type: ignore[arg-type]

        self._run = self._wandb.init(
            project=self.project,
            name=run_name,
            config=config_dict,
            tags=tags or [target, "design-loop"],
            reinit=True,
        )

        logger.info("W&B run initialized: %s", self._run.url if self._run else "N/A")

    def log_round(self, result: RoundResult) -> None:
        """Log summary metrics for a completed design round.

        Args:
            result: RoundResult from the pipeline.
        """
        if not self._enabled or self._run is None:
            return

        metrics = {
            "round": result.round_number,
            "candidates_entered": result.candidates_entered,
            "candidates_predicted": result.candidates_predicted,
            "candidates_passed_gates": result.candidates_passed_gates,
            "candidates_advanced": result.candidates_advanced,
            "best_composite_score": result.best_composite_score,
            "mean_composite_score": result.mean_composite_score,
            "median_iptm": result.median_iptm,
            "wall_time_hours": result.wall_time_hours,
            "rejection_rate": 1 - (
                result.candidates_passed_gates / max(result.candidates_predicted, 1)
            ),
        }

        self._wandb.log(metrics, step=result.round_number)
        logger.debug("Logged round %d metrics to W&B", result.round_number)

    def log_candidates(
        self,
        candidates: list[Candidate],
        round_num: int,
    ) -> None:
        """Log per-candidate scores as a W&B Table.

        Creates a sortable, filterable table in the W&B dashboard showing
        all candidates with their individual metric scores.
        """
        if not self._enabled or self._run is None:
            return

        columns = [
            "candidate_id", "sequence_length", "composite_score",
            "iptm", "plddt_interface", "shape_complementarity",
            "binding_energy", "buried_surface_area", "developability",
            "tier", "parent_id",
        ]

        data = []
        for c in candidates:
            data.append([
                c.candidate_id,
                len(c.sequence),
                c.composite_score or 0,
                c.iptm or 0,
                c.plddt_interface or 0,
                c.shape_complementarity or 0,
                c.binding_energy or 0,
                c.buried_surface_area or 0,
                c.developability_score or 0,
                c.tier or "unscored",
                c.parent_id or "seed",
            ])

        table = self._wandb.Table(columns=columns, data=data)
        self._wandb.log(
            {f"candidates_round_{round_num:02d}": table},
            step=round_num,
        )

    def log_convergence(
        self,
        round_num: int,
        best_scores: list[float],
        mean_scores: list[float],
    ) -> None:
        """Log convergence curve data.

        Args:
            round_num: Current round.
            best_scores: Best composite score per round so far.
            mean_scores: Mean composite score per round so far.
        """
        if not self._enabled or self._run is None:
            return

        self._wandb.log({
            "convergence/best_score": best_scores[-1] if best_scores else 0,
            "convergence/mean_score": mean_scores[-1] if mean_scores else 0,
            "convergence/score_delta": (
                best_scores[-1] - best_scores[-2]
                if len(best_scores) >= 2 else 0
            ),
        }, step=round_num)

    def log_artifact(
        self,
        file_path: Path,
        artifact_name: str,
        artifact_type: str = "result",
    ) -> None:
        """Upload a file as a W&B artifact.

        Useful for storing top candidate FASTA files, score tables,
        and predicted structures.
        """
        if not self._enabled or self._run is None:
            return

        artifact = self._wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
        )
        artifact.add_file(str(file_path))
        self._run.log_artifact(artifact)
        logger.debug("Logged artifact: %s (%s)", artifact_name, file_path)

    def log_experimental_correlation(
        self,
        metric_name: str,
        predicted: list[float],
        experimental: list[float],
        candidate_ids: list[str],
    ) -> None:
        """Log predicted vs. experimental correlation scatter.

        Used in Phase 6 for scoring function recalibration.
        """
        if not self._enabled or self._run is None:
            return

        data = [
            [cid, pred, exp]
            for cid, pred, exp in zip(candidate_ids, predicted, experimental, strict=False)
        ]
        table = self._wandb.Table(
            columns=["candidate_id", f"predicted_{metric_name}", f"experimental_{metric_name}"],
            data=data,
        )
        self._wandb.log({f"correlation/{metric_name}": table})

    def finish(self) -> None:
        """Finalize the tracking run."""
        if self._enabled and self._run is not None:
            self._run.finish()
            logger.info("W&B run finished")
