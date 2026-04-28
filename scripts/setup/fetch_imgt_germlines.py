#!/usr/bin/env python3
"""Fetch and curate VHH germline scaffolds for nanobody design.

Provides two modes:
1. Bundled mode (default): uses the pre-curated representative sequences
   shipped with the nanolfa package. No network access required.
2. Custom mode: reads user-provided FASTA files of VHH sequences (e.g.,
   downloaded from IMGT/GENE-DB or extracted from PDB nanobody structures).

Usage:
    # Use bundled germlines (no network needed)
    python scripts/setup/fetch_imgt_germlines.py \
        --output data/templates/germline_vhh/

    # Use bundled germlines filtered to alpaca only
    python scripts/setup/fetch_imgt_germlines.py \
        --species "Vicugna pacos" \
        --output data/templates/germline_vhh/

    # Import from custom FASTA (e.g., IMGT download)
    python scripts/setup/fetch_imgt_germlines.py \
        --input my_vhh_sequences.fasta \
        --cluster-identity 0.90 \
        --output data/templates/germline_vhh/

Note on IMGT access:
    IMGT/GENE-DB does not provide a public bulk-download API for germline
    sequences. To obtain sequences beyond the bundled set:
    1. Visit http://www.imgt.org/genedb/
    2. Select Species: Vicugna pacos or Camelus dromedarius
    3. Select Molecular component: IG, Chain type: IGH
    4. Download FASTA for IGHV, IGHD, IGHJ genes
    5. Provide the downloaded FASTA via --input
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input", "input_fasta",
    default=None, type=click.Path(exists=True),
    help="Custom FASTA file of VHH sequences. If omitted, uses bundled germlines.",
)
@click.option(
    "--species", "species_filter",
    multiple=True, type=str,
    help="Filter to specific species (e.g., 'Vicugna pacos').",
)
@click.option(
    "--cluster-identity", default=0.90, type=float,
    help="Sequence identity threshold for clustering.",
)
@click.option(
    "--min-vhh-score", default=0.6, type=float,
    help="Minimum VHH-likeness score to retain.",
)
@click.option(
    "--output", default="data/templates/germline_vhh/", type=click.Path(),
    help="Output directory for curated scaffolds.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def fetch_germlines(
    input_fasta: str | None,
    species_filter: tuple[str, ...],
    cluster_identity: float,
    min_vhh_score: float,
    output: str,
    verbose: bool,
) -> None:
    """Fetch and curate VHH germline scaffolds."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from nanolfa.utils.sequence import (
        curate_scaffold_library,
        load_bundled_germlines,
    )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scaffolds
    if input_fasta:
        click.echo(f"Loading custom sequences from {input_fasta}")
        scaffolds = _load_from_fasta(input_fasta)
    else:
        click.echo("Using bundled representative VHH germlines")
        species_list = list(species_filter) if species_filter else None
        scaffolds = load_bundled_germlines(species=species_list)

    click.echo(f"Loaded {len(scaffolds)} raw scaffolds")

    # Curate
    curated = curate_scaffold_library(
        scaffolds,
        min_vhh_score=min_vhh_score,
        cluster_identity=cluster_identity,
    )

    click.echo(f"Curated to {len(curated)} representative scaffolds")

    # Save FASTA
    fasta_path = output_dir / "scaffolds.fasta"
    with open(fasta_path, "w") as f:
        for scaffold in curated:
            header = (
                f">{scaffold.name} species={scaffold.species} "
                f"v_gene={scaffold.v_gene} j_gene={scaffold.j_gene} "
                f"cluster={scaffold.cluster_id}"
            )
            if scaffold.validation:
                header += f" vhh_score={scaffold.validation.vhh_score:.2f}"
            f.write(header + "\n")
            # Wrap sequence at 80 chars
            seq = scaffold.sequence
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")

    click.echo(f"Saved FASTA: {fasta_path}")

    # Save detailed JSON
    json_path = output_dir / "scaffolds.json"
    scaffold_data = []
    for s in curated:
        entry = {
            "name": s.name,
            "species": s.species,
            "v_gene": s.v_gene,
            "j_gene": s.j_gene,
            "sequence": s.sequence,
            "length": len(s.sequence),
            "cluster_id": s.cluster_id,
        }
        if s.regions:
            entry["regions"] = {
                "FR1": s.regions.fr1,
                "CDR1": s.regions.cdr1,
                "FR2": s.regions.fr2,
                "CDR2": s.regions.cdr2,
                "FR3": s.regions.fr3,
                "CDR3": s.regions.cdr3,
                "FR4": s.regions.fr4,
                "CDR3_length": s.regions.cdr3_length,
            }
        if s.validation:
            entry["validation"] = {
                "vhh_score": s.validation.vhh_score,
                "has_canonical_disulfide": s.validation.has_canonical_disulfide,
                "hallmark_matches": s.validation.hallmark_matches,
                "hallmark_total": s.validation.hallmark_total,
                "warnings": s.validation.warnings,
            }
        scaffold_data.append(entry)

    with open(json_path, "w") as f:
        json.dump(scaffold_data, f, indent=2)
    click.echo(f"Saved JSON:  {json_path}")

    # Print summary table
    click.echo(f"\n{'Name':<30} {'Species':<22} {'Len':>4} {'VHH':>5} {'CDR3':>5} {'Cluster':>7}")
    click.echo("-" * 100)
    for s in curated:
        cdr3_len = s.regions.cdr3_length if s.regions else 0
        vhh_score = s.validation.vhh_score if s.validation else 0
        click.echo(
            f"{s.name:<30} {s.species:<22} {len(s.sequence):>4} "
            f"{vhh_score:>5.2f} {cdr3_len:>5} {s.cluster_id:>7}"
        )

    click.echo(f"\nDone. {len(curated)} scaffolds ready for Phase 2.")


def _load_from_fasta(fasta_path: str) -> list:
    """Load VHH sequences from a FASTA file and validate each."""
    from nanolfa.utils.sequence import (
        GermlineScaffold,
        annotate_regions,
        validate_vhh,
    )

    scaffolds = []
    try:
        from Bio import SeqIO

        for record in SeqIO.parse(fasta_path, "fasta"):
            seq = str(record.seq)
            name = record.id

            # Try to parse species from header
            desc = record.description
            species = "unknown"
            if "Vicugna" in desc or "alpaca" in desc.lower():
                species = "Vicugna pacos"
            elif "Camelus" in desc or "dromedary" in desc.lower():
                species = "Camelus dromedarius"
            elif "Lama" in desc or "llama" in desc.lower():
                species = "Lama glama"

            regions = annotate_regions(seq, name=name)
            validation = validate_vhh(seq, name=name)

            scaffolds.append(
                GermlineScaffold(
                    name=name,
                    sequence=seq,
                    species=species,
                    v_gene="unknown",
                    j_gene="unknown",
                    regions=regions,
                    validation=validation,
                )
            )

    except ImportError:
        logger.error("BioPython required for FASTA parsing. Install: pip install biopython")
        raise

    return scaffolds


if __name__ == "__main__":
    fetch_germlines()
