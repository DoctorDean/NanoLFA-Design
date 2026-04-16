"""Pipeline orchestration for the iterative nanobody design loop."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import click
from omegaconf import DictConfig

from nanolfa.core.config import load_config
from nanolfa.filters.developability import DevelopabilityFilter
from nanolfa.filters.specificity import SpecificityFilter
from nanolfa.models.alphafold import AlphaFoldRunner
from nanolfa.models.esmfold import ESMFoldPrescreen
from nanolfa.models.proteinmpnn import ProteinMPNNDesigner
from nanolfa.scoring.composite import CompositeScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A single nanobody design candidate tracked through the pipeline."""

    candidate_id: str
    sequence: str
    round_created: int
    parent_id: str | None = None

    # Structure (populated after AF prediction)
    structure_path: Path | None = None
    complex_path: Path | None = None

    # Scores (populated after scoring)
    iptm: float | None = None
    plddt_interface: float | None = None
    shape_complementarity: float | None = None
    binding_energy: float | None = None
    buried_surface_area: float | None = None
    developability_score: float | None = None
    composite_score: float | None = None

    # Filters
    passed_hard_gates: bool | None = None
    tier: str | None = None  # "green", "yellow", "red"
    rejection_reason: str | None = None

    # Experimental (populated in Phase 6)
    experimental_kd: float | None = None
    experimental_kon: float | None = None
    experimental_koff: float | None = None
    experimental_tm: float | None = None
    lfa_signal_noise: float | None = None

    # Lineage tracking
    mutations_from_parent: list[str] = field(default_factory=list)


@dataclass
class RoundResult:
    """Summary of a single design round."""

    round_number: int
    candidates_entered: int
    candidates_predicted: int
    candidates_passed_gates: int
    candidates_advanced: int
    best_composite_score: float
    mean_composite_score: float
    median_iptm: float
    wall_time_hours: float
    advanced_candidates: list[Candidate] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class NanoLFAPipeline:
    """Master orchestrator for the iterative nanobody design pipeline.

    Manages the full lifecycle: seed generation → iterative design rounds →
    specificity screening → LFA optimization → experimental feedback.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.target_name: str = config.target.name
        self.output_dir = Path(config.pipeline.output_dir) / self.target_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.af_runner = AlphaFoldRunner(config.alphafold)
        self.mpnn_designer = ProteinMPNNDesigner(config.proteinmpnn)
        self.scorer = CompositeScorer(config.scoring)
        self.dev_filter = DevelopabilityFilter(config.filters)
        self.spec_filter = SpecificityFilter(config)
        self.esmfold_prescreen = ESMFoldPrescreen(config.alphafold.prescreening)

        # State
        self.all_candidates: list[Candidate] = []
        self.round_results: list[RoundResult] = []
        self.current_round: int = 0

        logger.info(
            "Pipeline initialized for target=%s, max_rounds=%d, variants/round=%d",
            self.target_name,
            config.pipeline.max_rounds,
            config.pipeline.variants_per_round,
        )

    # ------------------------------------------------------------------
    # Phase 3: Iterative design loop
    # ------------------------------------------------------------------

    def run_design_loop(self, seed_candidates: list[Candidate]) -> list[Candidate]:
        """Execute the iterative predict → score → diversify loop.

        Args:
            seed_candidates: Initial candidates from Phase 2.

        Returns:
            Final ranked candidates after all rounds.
        """
        current_pool = seed_candidates
        max_rounds = self.config.pipeline.max_rounds
        top_k = self.config.pipeline.top_k_advance
        convergence_thresh = self.config.pipeline.convergence_threshold

        for round_num in range(1, max_rounds + 1):
            self.current_round = round_num
            round_start = time.time()
            round_dir = self.output_dir / f"round_{round_num:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "=== Round %d/%d — %d candidates entering ===",
                round_num, max_rounds, len(current_pool),
            )

            # Step 3a: Predict complexes
            predicted = self._predict_complexes(current_pool, round_dir)

            # Step 3b: Score interfaces
            scored = self._score_candidates(predicted)

            # Step 3e: Developability filter (hard gates)
            filtered = self._apply_hard_gates(scored)

            # Rank by composite score
            filtered.sort(key=lambda c: c.composite_score or 0, reverse=True)

            # Classify tiers
            for cand in filtered:
                cand.tier = self._classify_tier(cand)

            advanced = [c for c in filtered if c.tier in ("green", "yellow")][:top_k]

            # Record round summary
            wall_time = (time.time() - round_start) / 3600
            result = RoundResult(
                round_number=round_num,
                candidates_entered=len(current_pool),
                candidates_predicted=len(predicted),
                candidates_passed_gates=len(filtered),
                candidates_advanced=len(advanced),
                best_composite_score=(
                    advanced[0].composite_score or 0.0 if advanced else 0.0
                ),
                mean_composite_score=self._mean_score(advanced),
                median_iptm=self._median_iptm(advanced),
                wall_time_hours=wall_time,
                advanced_candidates=advanced,
            )
            self.round_results.append(result)
            self._log_round_summary(result)
            self._save_round_artifacts(advanced, round_dir)

            # Convergence check
            if self._check_convergence(convergence_thresh):
                logger.info("Convergence reached at round %d. Stopping.", round_num)
                break

            # Step 3c: Diversify for next round (unless final round)
            if round_num < max_rounds:
                current_pool = self._diversify(advanced, round_num)
            else:
                current_pool = advanced

        return current_pool

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _predict_complexes(
        self, candidates: list[Candidate], round_dir: Path
    ) -> list[Candidate]:
        """Step 3a: Run AlphaFold-Multimer / AF3 on all candidates.

        If ESMFold prescreening is enabled, first triage with ESMFold and only
        send the top-K to full AlphaFold.
        """
        prescreen_cfg = self.config.alphafold.prescreening
        if prescreen_cfg.enabled and len(candidates) > prescreen_cfg.top_k_to_full_af:
            logger.info(
                "ESMFold prescreening: %d candidates → top %d",
                len(candidates), prescreen_cfg.top_k_to_full_af,
            )
            candidates = self.esmfold_prescreen.filter(
                candidates,
                min_plddt=prescreen_cfg.min_plddt_threshold,
                top_k=prescreen_cfg.top_k_to_full_af,
            )

        logger.info("Running AlphaFold on %d candidates", len(candidates))
        predicted = self.af_runner.predict_batch(
            candidates=candidates,
            target_config=self.config.target,
            output_dir=round_dir / "predictions",
        )
        return predicted

    def _score_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        """Step 3b: Compute composite score for each candidate."""
        logger.info("Scoring %d candidates", len(candidates))
        for cand in candidates:
            if cand.complex_path is None:
                cand.composite_score = 0.0
                continue
            scores = self.scorer.score_complex(cand.complex_path)
            cand.iptm = scores["iptm"]
            cand.plddt_interface = scores["plddt_interface"]
            cand.shape_complementarity = scores["shape_complementarity"]
            cand.binding_energy = scores["binding_energy"]
            cand.buried_surface_area = scores["buried_surface_area"]
            cand.developability_score = self.dev_filter.compute_score(cand)
            cand.composite_score = self.scorer.composite(scores, cand.developability_score)
        return candidates

    def _apply_hard_gates(self, candidates: list[Candidate]) -> list[Candidate]:
        """Step 3e: Apply hard-gate filters. Returns only passing candidates."""
        thresholds = self.config.scoring.thresholds.hard_gates
        passed = []
        for cand in candidates:
            if (cand.iptm or 0) < thresholds.iptm_min:
                cand.passed_hard_gates = False
                cand.rejection_reason = f"ipTM={cand.iptm:.3f} < {thresholds.iptm_min}"
                continue
            if (cand.plddt_interface or 0) < thresholds.plddt_interface_min:
                cand.passed_hard_gates = False
                cand.rejection_reason = (
                    f"pLDDT_if={cand.plddt_interface:.1f} < {thresholds.plddt_interface_min}"
                )
                continue
            if (cand.binding_energy or 0) > thresholds.binding_energy_max:
                cand.passed_hard_gates = False
                cand.rejection_reason = (
                    f"ΔG={cand.binding_energy:.1f} > {thresholds.binding_energy_max}"
                )
                continue

            dev_report = self.dev_filter.evaluate(cand)
            if dev_report.aggregation_score > thresholds.aggregation_score_max:
                cand.passed_hard_gates = False
                cand.rejection_reason = f"Agg={dev_report.aggregation_score:.2f}"
                continue
            if dev_report.liability_count > thresholds.liability_count_max:
                cand.passed_hard_gates = False
                cand.rejection_reason = f"Liabilities={dev_report.liability_count}"
                continue

            cand.passed_hard_gates = True
            passed.append(cand)

        logger.info(
            "Hard gates: %d/%d passed (%.0f%% rejection rate)",
            len(passed), len(candidates),
            100 * (1 - len(passed) / max(len(candidates), 1)),
        )
        return passed

    def _diversify(self, candidates: list[Candidate], round_num: int) -> list[Candidate]:
        """Step 3c: Generate variants of top candidates for the next round.

        Diversification strategy adapts by round:
        - Early rounds: aggressive CDR redesign, higher temperature
        - Late rounds: conservative point mutations, lower temperature
        """
        variants_per_round = self.config.pipeline.variants_per_round
        n_per_candidate = max(1, variants_per_round // len(candidates))

        # Adaptive temperature schedule
        temp_schedule = {1: 0.20, 2: 0.15, 3: 0.10, 4: 0.05, 5: 0.05}
        temperature = temp_schedule.get(round_num, 0.10)

        # Adaptive CDR targeting
        cdr_schedule = {
            1: ["CDR3"],
            2: ["CDR3", "CDR2"],
            3: ["CDR3", "CDR2", "CDR1"],
            4: ["CDR3", "CDR2", "CDR1"],
            5: ["CDR3", "CDR2", "CDR1"],
        }
        target_cdrs = cdr_schedule.get(round_num, ["CDR3"])

        logger.info(
            "Diversifying %d candidates × %d variants (T=%.2f, CDRs=%s)",
            len(candidates), n_per_candidate, temperature, target_cdrs,
        )

        new_candidates = self.mpnn_designer.diversify(
            candidates=candidates,
            n_variants=n_per_candidate,
            temperature=temperature,
            target_cdrs=target_cdrs,
            round_num=round_num,
        )
        return new_candidates

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _classify_tier(self, cand: Candidate) -> str:
        """Assign green/yellow/red tier based on composite score."""
        thresholds = self.config.scoring.thresholds
        score = cand.composite_score or 0
        if score >= thresholds.advance.composite_score_min:
            return "green"
        elif score >= thresholds.borderline.composite_score_min:
            return "yellow"
        return "red"

    def _check_convergence(self, threshold: float) -> bool:
        """Check if the best score has plateaued across recent rounds."""
        window = self.config.scoring.convergence.window
        min_rounds = self.config.scoring.convergence.min_rounds
        if len(self.round_results) < max(window + 1, min_rounds):
            return False
        current_best = self.round_results[-1].best_composite_score
        previous_best = self.round_results[-(window + 1)].best_composite_score
        delta = current_best - previous_best
        logger.info("Convergence check: Δ=%.4f (threshold=%.4f)", delta, threshold)
        return delta < threshold

    def _mean_score(self, candidates: list[Candidate]) -> float:
        scores = [c.composite_score for c in candidates if c.composite_score]
        return sum(scores) / len(scores) if scores else 0.0

    def _median_iptm(self, candidates: list[Candidate]) -> float:
        iptms = sorted(c.iptm for c in candidates if c.iptm is not None)
        if not iptms:
            return 0.0
        mid = len(iptms) // 2
        return iptms[mid] if len(iptms) % 2 else (iptms[mid - 1] + iptms[mid]) / 2

    def _log_round_summary(self, result: RoundResult) -> None:
        logger.info(
            "Round %d summary: entered=%d predicted=%d gated=%d advanced=%d "
            "best=%.3f mean=%.3f median_ipTM=%.3f time=%.1fh",
            result.round_number,
            result.candidates_entered,
            result.candidates_predicted,
            result.candidates_passed_gates,
            result.candidates_advanced,
            result.best_composite_score,
            result.mean_composite_score,
            result.median_iptm,
            result.wall_time_hours,
        )

    def _save_round_artifacts(self, candidates: list[Candidate], round_dir: Path) -> None:
        """Save candidate sequences, scores, and structures for a round."""
        # Save FASTA of advanced candidates
        fasta_path = round_dir / "top_candidates.fasta"
        with open(fasta_path, "w") as f:
            for cand in candidates:
                f.write(f">{cand.candidate_id} score={cand.composite_score:.4f} "
                        f"ipTM={cand.iptm:.3f} tier={cand.tier}\n")
                f.write(f"{cand.sequence}\n")

        # Save scores as TSV
        scores_path = round_dir / "scores.tsv"
        with open(scores_path, "w") as f:
            header = (
                "candidate_id\tsequence\tcomposite_score\tiptm\tplddt_interface\t"
                "shape_complementarity\tbinding_energy\tburied_surface_area\t"
                "developability\ttier\tparent_id\n"
            )
            f.write(header)
            for cand in candidates:
                f.write(
                    f"{cand.candidate_id}\t{cand.sequence}\t"
                    f"{cand.composite_score:.4f}\t{cand.iptm:.3f}\t"
                    f"{cand.plddt_interface:.1f}\t{cand.shape_complementarity:.3f}\t"
                    f"{cand.binding_energy:.1f}\t{cand.buried_surface_area:.1f}\t"
                    f"{cand.developability_score:.3f}\t{cand.tier}\t"
                    f"{cand.parent_id or 'seed'}\n"
                )
        logger.info("Saved %d candidates to %s", len(candidates), round_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="YAML config path")
@click.option("--rounds", default=None, type=int, help="Override max rounds")
@click.option("--seed-fasta", default=None, type=click.Path(), help="Seed sequences FASTA")
def main(config: str, rounds: int | None, seed_fasta: str | None) -> None:
    """Run the NanoLFA-Design iterative pipeline."""
    cfg = load_config(config)
    if rounds is not None:
        cfg.pipeline.max_rounds = rounds

    logging.basicConfig(
        level=getattr(logging, cfg.pipeline.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = NanoLFAPipeline(cfg)

    # Load or generate seed candidates
    if seed_fasta:
        seeds = _load_seeds_from_fasta(seed_fasta)
    else:
        logger.info("No seed FASTA provided — running Phase 2 seed generation")
        seeds = pipeline.generate_seeds()  # type: ignore[attr-defined]

    # Run the iterative design loop
    final_candidates = pipeline.run_design_loop(seeds)

    # Report
    logger.info(
        "Pipeline complete. %d final candidates. Best score: %.4f",
        len(final_candidates),
        final_candidates[0].composite_score if final_candidates else 0,
    )


def _load_seeds_from_fasta(fasta_path: str) -> list[Candidate]:
    """Load seed candidates from a FASTA file."""
    candidates = []
    from Bio import SeqIO

    for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        candidates.append(
            Candidate(
                candidate_id=record.id or f"seed_{i:04d}",
                sequence=str(record.seq),
                round_created=0,
            )
        )
    logger.info("Loaded %d seed candidates from %s", len(candidates), fasta_path)
    return candidates


if __name__ == "__main__":
    main()
