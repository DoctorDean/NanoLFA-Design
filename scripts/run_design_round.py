#!/usr/bin/env python3
"""Run a single design round of the NanoLFA pipeline.

Useful for restarting from a specific round or running rounds in parallel
across different HPC jobs.

Usage:
    python scripts/run_design_round.py \
        --target pdg \
        --round 2 \
        --input data/results/round_01/top_candidates.fasta \
        --n-variants 300 \
        --config configs/targets/pdg.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
from omegaconf import OmegaConf

from nanolfa.core.config import load_config
from nanolfa.core.pipeline import Candidate, NanoLFAPipeline

logger = logging.getLogger(__name__)


@click.command()
@click.option("--target", required=True, type=click.Choice(["pdg", "e3g"]))
@click.option("--round", "round_num", required=True, type=int)
@click.option("--input", "input_fasta", required=True, type=click.Path(exists=True))
@click.option("--n-variants", default=300, type=int)
@click.option("--config", default=None, type=click.Path(exists=True))
def run_single_round(
    target: str,
    round_num: int,
    input_fasta: str,
    n_variants: int,
    config: str | None,
) -> None:
    """Execute a single design round."""
    config_path = config or f"configs/targets/{target}.yaml"
    cfg = load_config(config_path)
    cfg.pipeline.variants_per_round = n_variants

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load candidates from previous round
    from Bio import SeqIO

    candidates = []
    for i, record in enumerate(SeqIO.parse(input_fasta, "fasta")):
        candidates.append(
            Candidate(
                candidate_id=record.id or f"r{round_num - 1}_{i:04d}",
                sequence=str(record.seq),
                round_created=round_num - 1,
            )
        )

    logger.info(
        "Loaded %d candidates for round %d of target %s",
        len(candidates), round_num, target,
    )

    pipeline = NanoLFAPipeline(cfg)
    pipeline.current_round = round_num

    # Run just one iteration of the design loop steps
    round_dir = pipeline.output_dir / f"round_{round_num:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # Predict
    predicted = pipeline._predict_complexes(candidates, round_dir)

    # Score
    scored = pipeline._score_candidates(predicted)

    # Filter
    filtered = pipeline._apply_hard_gates(scored)

    # Rank and save
    filtered.sort(key=lambda c: c.composite_score or 0, reverse=True)
    top_k = cfg.pipeline.top_k_advance
    advanced = filtered[:top_k]

    for cand in advanced:
        cand.tier = pipeline._classify_tier(cand)

    pipeline._save_round_artifacts(advanced, round_dir)

    logger.info(
        "Round %d complete: %d candidates → %d predicted → %d passed → %d advanced",
        round_num, len(candidates), len(predicted), len(filtered), len(advanced),
    )

    if advanced:
        logger.info(
            "Best candidate: %s (score=%.4f, ipTM=%.3f)",
            advanced[0].candidate_id,
            advanced[0].composite_score,
            advanced[0].iptm,
        )


if __name__ == "__main__":
    run_single_round()
