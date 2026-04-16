#!/usr/bin/env python3
"""Screen top candidates against the cross-reactivity panel.

Usage:
    python scripts/screen_crossreactivity.py \
        --candidates data/results/round_03/top_candidates.fasta \
        --panel configs/targets/pdg.yaml \
        --output data/results/specificity/
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from nanolfa.core.config import load_config
from nanolfa.core.pipeline import Candidate
from nanolfa.filters.specificity import SpecificityFilter
from nanolfa.models.alphafold import AlphaFoldRunner
from nanolfa.scoring.composite import CompositeScorer

logger = logging.getLogger(__name__)


@click.command()
@click.option("--candidates", required=True, type=click.Path(exists=True))
@click.option("--panel", required=True, type=click.Path(exists=True), help="Target config YAML")
@click.option("--output", default="data/results/specificity/", type=click.Path())
def screen(candidates: str, panel: str, output: str) -> None:
    """Run cross-reactivity screening on candidate nanobodies."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(panel)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load candidates
    from Bio import SeqIO

    cands = []
    for record in SeqIO.parse(candidates, "fasta"):
        cand = Candidate(
            candidate_id=record.id,
            sequence=str(record.seq),
            round_created=-1,
        )
        # Try to parse score from FASTA header
        desc = record.description
        if "score=" in desc:
            try:
                score_str = desc.split("score=")[1].split()[0]
                cand.composite_score = float(score_str)
            except (IndexError, ValueError):
                pass
        cands.append(cand)

    logger.info("Screening %d candidates against cross-reactivity panel", len(cands))

    # Initialize tools
    spec_filter = SpecificityFilter(cfg)
    af_runner = AlphaFoldRunner(cfg.alphafold)
    scorer = CompositeScorer(cfg.scoring)

    # Run screening
    results = spec_filter.screen(cands, af_runner, scorer)

    # Save results
    results_file = output_dir / "specificity_results.tsv"
    with open(results_file, "w") as f:
        f.write(
            "candidate_id\ton_target_score\tworst_off_target\t"
            "worst_off_target_score\tselectivity_ratio\tpasses\tfailure_reasons\n"
        )
        for r in results:
            f.write(
                f"{r.candidate_id}\t{r.on_target_score:.4f}\t{r.worst_off_target}\t"
                f"{r.worst_off_target_score:.4f}\t{r.selectivity_ratio:.2f}\t"
                f"{r.passes}\t{'; '.join(r.failure_reasons)}\n"
            )

    passed = [r for r in results if r.passes]
    logger.info(
        "Specificity screening complete: %d/%d passed. Results saved to %s",
        len(passed), len(results), results_file,
    )


if __name__ == "__main__":
    screen()
