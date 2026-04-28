#!/usr/bin/env python3
"""Phase 4: Screen top candidates against the cross-reactivity panel.

Runs AlphaFold predictions of each candidate against every compound in the
panel, scores the predicted complexes, and classifies candidates as specific,
borderline, or cross-reactive. Optionally attempts negative-design rescue
for borderline candidates.

Usage:
    # Basic screening
    python scripts/screen_crossreactivity.py \
        --candidates data/results/round_05/top_candidates.fasta \
        --config configs/targets/pdg.yaml

    # With negative design rescue
    python scripts/screen_crossreactivity.py \
        --candidates data/results/round_05/top_candidates.fasta \
        --config configs/targets/pdg.yaml \
        --rescue

    # Custom output directory
    python scripts/screen_crossreactivity.py \
        --candidates data/results/round_05/top_candidates.fasta \
        --config configs/targets/pdg.yaml \
        --output data/results/specificity/pdg/
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--candidates", required=True, type=click.Path(exists=True),
    help="FASTA file of candidates to screen.",
)
@click.option(
    "--config", required=True, type=click.Path(exists=True),
    help="Target config YAML (contains cross-reactivity panel).",
)
@click.option(
    "--output", default="data/results/specificity/",
    type=click.Path(), help="Output directory.",
)
@click.option(
    "--rescue", is_flag=True, default=False,
    help="Attempt negative-design rescue of borderline candidates.",
)
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
def screen(
    candidates: str,
    config: str,
    output: str,
    rescue: bool,
    verbose: bool,
) -> None:
    """Screen nanobody candidates for cross-reactivity."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from nanolfa.core.config import load_config
    from nanolfa.core.pipeline import Candidate
    from nanolfa.filters.specificity import SpecificityFilter
    from nanolfa.models.alphafold import AlphaFoldRunner
    from nanolfa.scoring.composite import CompositeScorer

    cfg = load_config(config)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load candidates
    from Bio import SeqIO

    cands: list[Candidate] = []
    for record in SeqIO.parse(candidates, "fasta"):
        cand = Candidate(
            candidate_id=record.id,
            sequence=str(record.seq),
            round_created=-1,
        )
        desc = record.description
        if "score=" in desc:
            with contextlib.suppress(IndexError, ValueError):
                cand.composite_score = float(desc.split("score=")[1].split()[0])
        cands.append(cand)

    click.echo(f"Loaded {len(cands)} candidates")
    click.echo(f"Cross-reactivity panel: {len(cfg.target.cross_reactivity_panel)} compounds")

    # Initialize tools
    spec_filter = SpecificityFilter(cfg)
    af_runner = AlphaFoldRunner(cfg.alphafold)
    scorer = CompositeScorer(cfg.scoring)

    # Run screening
    results = spec_filter.screen(cands, af_runner, scorer, output_dir)

    # Save results
    spec_filter.save_results(results, output_dir)

    # Summary
    specific = [r for r in results if r.tier == "specific"]
    borderline = [r for r in results if r.tier == "borderline"]
    cross_reactive = [r for r in results if r.tier == "cross-reactive"]
    flagged = [r for r in results if r.flagged_for_experimental]

    click.echo(f"\n{'=' * 60}")
    click.echo("Cross-Reactivity Screening Results")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Specific:       {len(specific):>3} (pass)")
    click.echo(f"  Borderline:     {len(borderline):>3} (rescue eligible)")
    click.echo(f"  Cross-reactive: {len(cross_reactive):>3} (reject)")
    click.echo(f"  Flagged for experimental: {len(flagged)}")

    # Print top candidates
    if specific:
        click.echo("\nTop specific candidates:")
        for r in sorted(specific, key=lambda x: x.selectivity_ratio, reverse=True)[:10]:
            click.echo(
                f"  {r.candidate_id}: selectivity={r.selectivity_ratio:.1f}x, "
                f"worst_OT={r.worst_off_target} ({r.worst_off_target_score:.3f})"
            )

    # Negative design rescue
    if rescue and borderline:
        click.echo(f"\nAttempting negative-design rescue for {len(borderline)} candidates...")

        from nanolfa.models.proteinmpnn import ProteinMPNNDesigner

        mpnn = ProteinMPNNDesigner(cfg.proteinmpnn)

        rescue_results = spec_filter.rescue_borderline(
            results=results,
            candidates=cands,
            af_runner=af_runner,
            scorer=scorer,
            mpnn_designer=mpnn,
            output_dir=output_dir / "rescue",
        )

        accepted = [r for r in rescue_results if r.accepted]
        click.echo(f"  Rescue attempts: {len(rescue_results)}")
        click.echo(f"  Accepted: {len(accepted)}")

        for r in accepted:
            click.echo(
                f"    {r.original_candidate_id} -> {r.rescue_candidate_id}: "
                f"on-target {r.on_target_delta_pct:+.1f}%, "
                f"off-target {r.off_target_delta_pct:+.1f}% "
                f"({len(r.mutations)} mutations)"
            )

    click.echo(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    screen()
