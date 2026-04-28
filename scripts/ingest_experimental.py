#!/usr/bin/env python3
"""Phase 6: Ingest experimental data and recalibrate scoring function.

Parses SPR kinetics, thermal shift, and LFA prototype data, correlates
with computational predictions, and recalibrates scoring weights.

Usage:
    python scripts/ingest_experimental.py \
        --spr data/experimental/spr_kinetics.csv \
        --thermal data/experimental/thermal_shift.csv \
        --lfa data/experimental/lfa_signal.csv \
        --scores data/results/round_05/scores.tsv \
        --config configs/targets/pdg.yaml \
        --output data/results/calibration/
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option("--spr", default=None, type=click.Path(exists=True),
              help="SPR kinetics CSV (candidate_id, kon, koff, KD, chi2)")
@click.option("--thermal", default=None, type=click.Path(exists=True),
              help="Thermal shift CSV (candidate_id, Tm_celsius, Tagg_celsius)")
@click.option("--lfa", default=None, type=click.Path(exists=True),
              help="LFA signal CSV (candidate_id, signal_noise_ratio, ...)")
@click.option("--scores", required=True, type=click.Path(exists=True),
              help="Computational scores TSV from pipeline (scores.tsv)")
@click.option("--config", required=True, type=click.Path(exists=True),
              help="Target config YAML")
@click.option("--target-metric", default="log_kd",
              type=click.Choice(["log_kd", "kd", "kon", "tm", "snr"]),
              help="Experimental metric to optimize scoring for")
@click.option("--method", default="ridge",
              type=click.Choice(["ridge", "correlation"]),
              help="Recalibration method")
@click.option("--output", default="data/results/calibration/",
              type=click.Path(), help="Output directory")
@click.option("-v", "--verbose", is_flag=True)
def ingest(
    spr: str | None,
    thermal: str | None,
    lfa: str | None,
    scores: str,
    config: str,
    target_metric: str,
    method: str,
    output: str,
    verbose: bool,
) -> None:
    """Ingest experimental data and recalibrate the scoring function."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from nanolfa.core.calibration import (
        correlate_all_metrics,
        find_hidden_gems,
        ingest_experimental_data,
        recalibrate_weights,
        save_calibration,
    )
    from nanolfa.core.config import load_config

    cfg = load_config(config)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest experimental data
    click.echo("Step 1: Ingesting experimental data...")
    exp_data = ingest_experimental_data(
        spr_path=spr, thermal_path=thermal, lfa_path=lfa,
    )
    click.echo(f"  {len(exp_data)} candidates with experimental data")

    # 2. Load computational scores
    click.echo("Step 2: Loading computational scores...")
    candidates_scores = _load_scores_tsv(scores)
    click.echo(f"  {len(candidates_scores)} candidates with computational scores")

    # 3. Correlation analysis
    click.echo(f"Step 3: Correlating with {target_metric}...")
    correlations = correlate_all_metrics(
        candidates_scores, exp_data, target_metric=target_metric,
    )

    if not correlations:
        click.echo("  Insufficient data for correlation analysis.")
        click.echo("  Need at least 5 candidates with both computational and experimental data.")
        return

    click.echo(f"\n  {'Metric':<25} {'Pearson r':>10} {'Spearman ρ':>11} {'R²':>8}")
    click.echo("  " + "-" * 57)
    for c in correlations:
        click.echo(
            f"  {c.metric_name:<25} {c.pearson_r:>10.3f} "
            f"{c.spearman_rho:>11.3f} {c.r_squared:>8.3f}"
        )

    # 4. Recalibrate weights
    click.echo(f"\nStep 4: Recalibrating weights (method={method})...")
    current_weights = dict(cfg.scoring.weights)
    recal = recalibrate_weights(
        correlations, current_weights, method=method,
    )

    click.echo(f"\n  {'Metric':<25} {'Old':>8} {'New':>8} {'Change':>8}")
    click.echo("  " + "-" * 52)
    for metric in current_weights:
        old = recal.old_weights.get(metric, 0)
        new = recal.new_weights.get(metric, 0)
        change = recal.weight_changes.get(metric, 0)
        arrow = "↑" if change > 0.01 else "↓" if change < -0.01 else "="
        click.echo(f"  {metric:<25} {old:>8.3f} {new:>8.3f} {change:>+7.3f} {arrow}")

    click.echo(f"\n  Training R²: {recal.training_r_squared:.3f}")
    click.echo(f"  Cross-val R²: {recal.cross_val_r_squared:.3f}")

    # 5. Hidden gems
    click.echo("\nStep 5: Searching for hidden gems...")
    gems = find_hidden_gems(
        candidates_scores, recal.old_weights, recal.new_weights,
        exp_data, top_n=10,
    )

    if gems:
        click.echo(f"\n  {'Candidate':<25} {'Old Rank':>9} {'New Rank':>9} {'Improvement':>12}")
        click.echo("  " + "-" * 58)
        for g in gems[:10]:
            click.echo(
                f"  {g.candidate_id:<25} {g.old_rank:>9} "
                f"{g.new_rank:>9} {g.rank_improvement:>+12}"
            )
    else:
        click.echo("  No hidden gems found (rankings stable under recalibration)")

    # 6. Save
    click.echo("\nStep 6: Saving calibration results...")
    cal_path = save_calibration(recal, correlations, output_dir)
    click.echo(f"  Calibration JSON: {cal_path}")

    click.echo(f"\nPhase 6 complete. Update configs/scoring.yaml with new weights from {cal_path}")


def _load_scores_tsv(path: str) -> dict[str, dict[str, float]]:
    """Load computational scores from a pipeline scores.tsv file."""
    scores: dict[str, dict[str, float]] = {}
    metric_columns = [
        "iptm", "plddt_interface", "shape_complementarity",
        "binding_energy", "buried_surface_area", "developability",
    ]

    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cid = row.get("candidate_id", "").strip()
            if not cid:
                continue
            metrics: dict[str, float] = {}
            for col in metric_columns:
                val = row.get(col, "0")
                try:
                    metrics[col] = float(val)
                except ValueError:
                    metrics[col] = 0.0
            scores[cid] = metrics

    return scores


if __name__ == "__main__":
    ingest()
