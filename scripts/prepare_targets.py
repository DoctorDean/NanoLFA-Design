#!/usr/bin/env python3
"""Phase 1: Prepare target molecule structures from SMILES.

Generates 3D conformers, computes epitope surface maps, and processes
cross-reactivity panels for PdG and/or E3G targets.

Usage:
    # Single target
    python scripts/prepare_targets.py --config configs/targets/pdg.yaml

    # Both targets
    python scripts/prepare_targets.py --config configs/targets/pdg.yaml \
                                       --config configs/targets/e3g.yaml

    # Custom parameters
    python scripts/prepare_targets.py --config configs/targets/pdg.yaml \
        --n-conformers 500 --force-field MMFF94 --output data/targets
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config", "config_paths",
    required=True, multiple=True, type=click.Path(exists=True),
    help="Target config YAML(s). Can specify multiple.",
)
@click.option(
    "--output", "output_dir",
    default="data/targets", type=click.Path(),
    help="Output directory for prepared structures.",
)
@click.option(
    "--n-conformers", default=200, type=int,
    help="Number of conformers to generate per molecule.",
)
@click.option(
    "--force-field", default="MMFF94", type=click.Choice(["MMFF94", "UFF"]),
    help="Force field for energy minimization.",
)
@click.option(
    "--seed", default=42, type=int,
    help="Random seed for reproducibility.",
)
@click.option(
    "--formats", default="sdf,pdb", type=str,
    help="Export formats (comma-separated: sdf,mol2,pdb).",
)
@click.option(
    "--skip-cross-reactants", is_flag=True, default=False,
    help="Skip cross-reactivity panel processing.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def prepare_targets(
    config_paths: tuple[str, ...],
    output_dir: str,
    n_conformers: int,
    force_field: str,
    seed: int,
    formats: str,
    skip_cross_reactants: bool,
    verbose: bool,
) -> None:
    """Prepare 3D structures and epitope surface maps for target analytes."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from nanolfa.utils.chemistry import (
        prepare_target,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []

    for config_path in config_paths:
        logger.info("Loading config: %s", config_path)
        cfg = OmegaConf.load(config_path)

        # Extract target section (handle inheritance if needed)
        if isinstance(cfg, DictConfig) and "target" in cfg:
            target_cfg = OmegaConf.to_container(cfg.target, resolve=True)
        else:
            logger.error("Config %s has no 'target' section, skipping", config_path)
            continue

        if skip_cross_reactants:
            target_cfg["cross_reactivity_panel"] = []

        result = prepare_target(
            target_config=target_cfg,
            output_dir=out_path,
            n_conformers=n_conformers,
            force_field=force_field,
            random_seed=seed,
        )
        results.append(result)

        # Print summary
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Target: {result.target_name}")
        click.echo(f"{'=' * 60}")
        click.echo(f"  Conformers generated: {result.analyte.num_conformers_generated}")
        click.echo(f"  Best energy: {result.analyte.best_energy_kcal:.2f} kcal/mol")
        if result.analyte.output_sdf:
            click.echo(f"  SDF output: {result.analyte.output_sdf}")
        if result.epitope_map:
            em = result.epitope_map
            click.echo(f"  Total SASA: {em.total_sasa:.1f} A^2")
            click.echo(f"  Epitope SASA: {em.epitope_sasa:.1f} A^2 ({em.epitope_fraction:.0%})")
            click.echo(f"  Epitope atoms: {len(em.epitope_atom_indices)}")
            click.echo(f"  Buried atoms: {len(em.buried_atom_indices)}")

            epi_feats = [f for f in em.pharmacophore_features if f.is_epitope_accessible]
            click.echo(f"  Epitope pharmacophore features: {len(epi_feats)}")
            for ft in sorted({f.feature_type for f in epi_feats}):
                count = len([f for f in epi_feats if f.feature_type == ft])
                click.echo(f"    {ft}: {count}")

        click.echo(f"  Cross-reactants processed: {len(result.cross_reactants)}")
        for cr in result.cross_reactants:
            click.echo(f"    - {cr.name}: {cr.best_energy_kcal:.2f} kcal/mol")

    click.echo(f"\nPhase 1 complete. {len(results)} target(s) prepared in {out_path}")


if __name__ == "__main__":
    prepare_targets()
