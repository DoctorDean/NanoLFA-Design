#!/usr/bin/env python3
"""Phase 3.5: MD validation of top candidates from the design loop.

Runs short molecular dynamics simulations on AF-predicted complexes to
validate binding stability. Catches false positives that static scoring
misses: unstable binding poses, floppy CDR loops, and dissociating
complexes.

Usage:
    # Validate top candidates from round 5
    python scripts/run_md_validation.py \
        --candidates data/results/round_05/top_candidates.fasta \
        --complex-dir data/results/round_05/predictions/ \
        --config configs/md.yaml \
        --output data/results/md_validation/

    # Quick 5ns screen
    python scripts/run_md_validation.py \
        --candidates data/results/round_05/top_candidates.fasta \
        --complex-dir data/results/round_05/predictions/ \
        --duration-ns 5

    # With score adjustment (updates composite scores)
    python scripts/run_md_validation.py \
        --candidates data/results/round_05/top_candidates.fasta \
        --complex-dir data/results/round_05/predictions/ \
        --scores data/results/round_05/scores.tsv \
        --adjust-scores
"""

from __future__ import annotations

import contextlib
import csv
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--candidates", required=True, type=click.Path(exists=True),
    help="FASTA of candidates to validate.",
)
@click.option(
    "--complex-dir", required=True, type=click.Path(exists=True),
    help="Directory containing predicted complex PDBs (one per candidate).",
)
@click.option(
    "--config", default="configs/md.yaml", type=click.Path(),
    help="MD configuration YAML.",
)
@click.option(
    "--scores", default=None, type=click.Path(exists=True),
    help="Scores TSV from Phase 3 (for --adjust-scores).",
)
@click.option(
    "--adjust-scores", is_flag=True, default=False,
    help="Apply MD-based adjustments to composite scores.",
)
@click.option(
    "--duration-ns", default=None, type=float,
    help="Override production MD duration (ns).",
)
@click.option(
    "--top-n", default=50, type=int,
    help="Validate only the top N candidates.",
)
@click.option(
    "--output", default="data/results/md_validation/",
    type=click.Path(), help="Output directory.",
)
@click.option("-v", "--verbose", is_flag=True)
def run_md(
    candidates: str,
    complex_dir: str,
    config: str,
    scores: str | None,
    adjust_scores: bool,
    duration_ns: float | None,
    top_n: int,
    output: str,
    verbose: bool,
) -> None:
    """Run MD validation on predicted nanobody-hapten complexes."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from omegaconf import OmegaConf

    from nanolfa.models.md_validation import MDConfig, validate_batch
    from nanolfa.scoring.md_scores import apply_md_adjustments

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MD config
    md_cfg = MDConfig()
    if Path(config).exists():
        yaml_cfg = OmegaConf.load(config)
        if "md_validation" in yaml_cfg:
            mc = yaml_cfg.md_validation
            md_cfg.forcefield = mc.get("forcefield", md_cfg.forcefield)
            md_cfg.water_model = mc.get("water_model", md_cfg.water_model)
            md_cfg.production_steps = mc.get("production_steps", md_cfg.production_steps)
            md_cfg.temperature_kelvin = mc.get("temperature_kelvin", md_cfg.temperature_kelvin)

    if duration_ns is not None:
        md_cfg.production_steps = int(duration_ns * 1e6 / md_cfg.timestep_fs)

    sim_ns = md_cfg.production_steps * md_cfg.timestep_fs / 1e6
    click.echo(f"MD validation: {sim_ns:.0f}ns production, {md_cfg.temperature_kelvin:.0f}K")

    # Find candidate PDBs
    from Bio import SeqIO

    complex_path = Path(complex_dir)
    candidate_pdbs: list[tuple[str, Path]] = []

    for record in SeqIO.parse(candidates, "fasta"):
        cid = record.id
        # Search for complex PDB in various naming conventions
        possible = [
            complex_path / cid / "ranked_0.pdb",
            complex_path / f"{cid}.pdb",
            complex_path / cid / f"{cid}_relaxed.pdb",
            complex_path / cid / "best_model.pdb",
        ]
        pdb_found = None
        for p in possible:
            if p.exists():
                pdb_found = p
                break

        if pdb_found:
            candidate_pdbs.append((cid, pdb_found))
        else:
            logger.warning("No PDB found for %s in %s", cid, complex_path)

    # Limit to top-N
    candidate_pdbs = candidate_pdbs[:top_n]
    click.echo(f"Found {len(candidate_pdbs)} candidate PDBs to validate")

    if not candidate_pdbs:
        click.echo("No candidates to validate. Check --complex-dir.")
        return

    # Run validation
    results = validate_batch(
        candidates=candidate_pdbs,
        config=md_cfg,
        output_dir=output_dir,
    )

    # Save results TSV
    tsv_path = output_dir / "md_validation_results.tsv"
    with open(tsv_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "candidate_id", "passes", "md_score", "ligand_rmsd_mean",
            "ligand_rmsd_std", "contact_persistence", "binding_stable",
            "cdr3_rmsf", "framework_rmsf", "cdr_rigidity_ratio",
            "binding_fe_kj", "ligand_escaped", "sim_time_ns",
        ])
        for r in results:
            writer.writerow([
                r.candidate_id, r.passes_validation,
                f"{r.md_validation_score:.3f}", f"{r.ligand_rmsd_mean:.2f}",
                f"{r.ligand_rmsd_std:.2f}", f"{r.contact_persistence:.3f}",
                r.binding_stable, f"{r.cdr3_rmsf:.2f}", f"{r.framework_rmsf:.2f}",
                f"{r.cdr_rigidity_ratio:.2f}",
                f"{r.estimated_binding_free_energy_kj:.1f}",
                r.ligand_escaped, f"{r.simulation_time_ns:.1f}",
            ])

    click.echo(f"\nResults saved to {tsv_path}")

    # Summary
    passed = [r for r in results if r.passes_validation]
    escaped = [r for r in results if r.ligand_escaped]

    click.echo(f"\n{'=' * 60}")
    click.echo("MD Validation Results")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Total candidates:  {len(results)}")
    click.echo(f"  Passed:            {len(passed)}")
    click.echo(f"  Ligand escaped:    {len(escaped)}")
    click.echo(f"  Mean MD score:     {np.mean([r.md_validation_score for r in results]):.3f}")
    click.echo(f"  Mean RMSD:         {np.mean([r.ligand_rmsd_mean for r in results]):.2f} Å")
    click.echo(f"  Mean contacts:     {np.mean([r.contact_persistence for r in results]):.2f}")

    # Score adjustment
    if adjust_scores and scores:
        click.echo("\nApplying MD score adjustments...")

        score_lookup: dict[str, float] = {}
        with open(scores) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                cid = row.get("candidate_id", "").strip()
                cs = row.get("composite_score", "0")
                with contextlib.suppress(ValueError):
                    score_lookup[cid] = float(cs)

        adjustments = apply_md_adjustments(score_lookup, results)

        adj_path = output_dir / "md_adjusted_scores.tsv"
        with open(adj_path, "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                "candidate_id", "static_score", "md_score", "md_modifier",
                "adjusted_score", "recommendation",
            ])
            for a in adjustments:
                writer.writerow([
                    a.candidate_id, f"{a.static_score:.4f}",
                    f"{a.md_validation_score:.3f}", f"{a.md_modifier:.3f}",
                    f"{a.adjusted_score:.4f}", a.recommendation,
                ])

        promoted = sum(1 for a in adjustments if a.recommendation == "promote")
        rejected = sum(1 for a in adjustments if a.recommendation == "reject")
        click.echo(f"  Promoted:  {promoted}")
        click.echo(f"  Rejected:  {rejected}")
        click.echo(f"  Adjusted scores: {adj_path}")


# Needed for summary stats
import numpy as np  # noqa: E402

if __name__ == "__main__":
    run_md()
