#!/usr/bin/env python3
"""Phase 5: LFA-specific optimization and compatibility screening.

Evaluates candidates for lateral flow assay deployment suitability:
kinetic accessibility, gold nanoparticle conjugation orientation, and
thermal stability.

Usage:
    python scripts/optimize_lfa.py \
        --candidates data/results/specificity/specific_candidates.fasta \
        --config configs/targets/pdg.yaml \
        --complex-dir data/results/round_05/predictions/ \
        --output data/results/lfa_optimization/
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--candidates", required=True, type=click.Path(exists=True),
    help="FASTA of candidates passing Phase 4.",
)
@click.option(
    "--config", required=True, type=click.Path(exists=True),
    help="Target config YAML.",
)
@click.option(
    "--complex-dir", default=None, type=click.Path(),
    help="Directory with predicted complex PDBs (for kinetics/orientation).",
)
@click.option(
    "--output", default="data/results/lfa_optimization/",
    type=click.Path(), help="Output directory.",
)
@click.option("-v", "--verbose", is_flag=True)
def optimize_lfa(
    candidates: str,
    config: str,
    complex_dir: str | None,
    output: str,
    verbose: bool,
) -> None:
    """Screen candidates for LFA deployment suitability."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from nanolfa.core.config import load_config
    from nanolfa.core.pipeline import Candidate
    from nanolfa.filters.lfa_compat import LFACompatibilityFilter

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
        cands.append(cand)

    click.echo(f"Loaded {len(cands)} candidates for LFA optimization")

    # Initialize filter
    lfa_filter = LFACompatibilityFilter(cfg)

    # Run assessment
    complex_path = Path(complex_dir) if complex_dir else None
    passing, all_results = lfa_filter.filter_batch(cands, complex_dir=complex_path)

    # Save results
    tsv_path = output_dir / "lfa_compatibility.tsv"
    with open(tsv_path, "w") as f:
        f.write(
            "candidate_id\tpasses_all\tlfa_score\t"
            "kinetics_ok\torientation_ok\tstability_ok\t"
            "predicted_tm\tkon_category\twarnings\n"
        )
        for r in all_results:
            tm = r.stability.predicted_tm if r.stability else 0
            kon_cat = r.kinetics.estimated_kon_category if r.kinetics else "n/a"
            f.write(
                f"{r.candidate_id}\t{r.passes_all}\t{r.lfa_score:.3f}\t"
                f"{r.passes_kinetics}\t{r.passes_orientation}\t"
                f"{r.passes_stability}\t{tm:.1f}\t{kon_cat}\t"
                f"{'; '.join(r.warnings)}\n"
            )

    # Summary
    click.echo(f"\n{'=' * 60}")
    click.echo("LFA Compatibility Results")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Total candidates:   {len(all_results)}")
    click.echo(f"  LFA-compatible:     {len(passing)}")
    click.echo(f"  Kinetics pass:      {sum(1 for r in all_results if r.passes_kinetics)}")
    click.echo(f"  Orientation pass:   {sum(1 for r in all_results if r.passes_orientation)}")
    click.echo(f"  Stability pass:     {sum(1 for r in all_results if r.passes_stability)}")

    if passing:
        click.echo("\nTop LFA-compatible candidates:")
        sorted_results = sorted(
            [r for r in all_results if r.passes_all],
            key=lambda r: r.lfa_score, reverse=True,
        )
        for r in sorted_results[:10]:
            tm = r.stability.predicted_tm if r.stability else 0
            kon = r.kinetics.estimated_kon_category if r.kinetics else "n/a"
            click.echo(f"  {r.candidate_id}: LFA={r.lfa_score:.2f}, Tm={tm:.0f}°C, kon={kon}")

    click.echo(f"\nResults saved to {tsv_path}")


if __name__ == "__main__":
    optimize_lfa()
