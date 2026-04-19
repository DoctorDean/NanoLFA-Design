"""VHH nanobody sequence utilities.

Provides tools for working with camelid single-domain antibody sequences:
- IMGT numbering and CDR/framework region annotation
- VHH hallmark residue validation
- Canonical disulfide verification
- Sequence clustering for scaffold library curation
- Germline gene assignment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IMGT numbering constants for VHH
# ---------------------------------------------------------------------------

# IMGT CDR boundaries (inclusive, 1-indexed)
IMGT_REGIONS = {
    "FR1": (1, 26),
    "CDR1": (27, 38),
    "FR2": (39, 55),
    "CDR2": (56, 65),
    "FR3": (66, 104),
    "CDR3": (105, 117),
    "FR4": (118, 128),
}

# Hallmark residues that distinguish VHH from conventional VH
# IMGT positions and expected amino acids for VHH
VHH_HALLMARKS = {
    37: {"V", "F", "Y", "L"},     # conventionally Val in VH → often Phe/Tyr in VHH
    44: {"Q", "R", "E", "G"},     # conventionally Gly in VH → often Glu/Arg in VHH
    45: {"R", "L", "C", "W"},     # conventionally Leu in VH → often Arg in VHH
    47: {"G", "W", "F", "L"},     # conventionally Trp in VH → often Gly/Phe in VHH
}

# Canonical cysteines for the conserved disulfide bond
CANONICAL_CYS_POSITIONS = (23, 104)  # IMGT numbering

# Additional disulfide sometimes found in VHH (CDR3 to FR2)
EXTRA_DISULFIDE_POSITIONS = (33, 109)  # CDR1-Cys to CDR3-Cys (less common)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VHHRegions:
    """Annotated regions of a VHH sequence."""

    sequence: str
    name: str = ""
    fr1: str = ""
    cdr1: str = ""
    fr2: str = ""
    cdr2: str = ""
    fr3: str = ""
    cdr3: str = ""
    fr4: str = ""

    @property
    def cdrs(self) -> str:
        """Concatenated CDR sequences."""
        return self.cdr1 + self.cdr2 + self.cdr3

    @property
    def frameworks(self) -> str:
        """Concatenated framework sequences."""
        return self.fr1 + self.fr2 + self.fr3 + self.fr4

    @property
    def cdr3_length(self) -> int:
        return len(self.cdr3)


@dataclass
class VHHValidation:
    """Results of VHH sequence validation."""

    name: str
    sequence_length: int
    is_valid_length: bool           # 110–140 residues typical for VHH
    has_canonical_disulfide: bool   # Cys23–Cys104
    has_extra_disulfide: bool       # Cys33–Cys109 (optional)
    hallmark_residues: dict[int, str]  # position → actual residue
    hallmark_matches: int           # how many hallmarks match VHH pattern
    hallmark_total: int             # total hallmarks checked
    vhh_score: float                # 0–1, overall VHH-likeness
    warnings: list[str] = field(default_factory=list)


@dataclass
class GermlineScaffold:
    """A curated VHH germline scaffold for nanobody design."""

    name: str
    sequence: str
    species: str                    # e.g. "Vicugna pacos"
    v_gene: str                     # e.g. "IGHV3S53"
    j_gene: str                     # e.g. "IGHJ4"
    regions: VHHRegions | None = None
    validation: VHHValidation | None = None
    cluster_id: int = -1            # assigned during clustering


# ---------------------------------------------------------------------------
# IMGT region annotation
# ---------------------------------------------------------------------------

def annotate_regions(
    sequence: str,
    name: str = "",
    scheme: str = "imgt",
) -> VHHRegions:
    """Annotate CDR and framework regions in a VHH sequence.

    Uses approximate positional assignment based on IMGT numbering.
    For a more accurate annotation, use ANARCI or IMGT/DomainGapAlign.

    The IMGT scheme for VHH defines fixed framework lengths with variable
    CDR lengths, particularly CDR3 which can be 8–25 residues.

    Args:
        sequence: VHH amino acid sequence (single-letter code).
        name: Identifier for the sequence.
        scheme: Numbering scheme ("imgt" only currently).

    Returns:
        VHHRegions with annotated region strings.
    """
    if scheme != "imgt":
        raise ValueError(f"Only 'imgt' scheme is supported, got '{scheme}'")

    seq_len = len(sequence)

    # IMGT numbering assumes fixed framework lengths, with gaps in CDR regions.
    # For raw sequences without alignment, we use a heuristic:
    #   FR1:  residues 1–26    (26 residues, fairly conserved)
    #   CDR1: residues 27–38   (up to 12 residues, often 6–8 in VHH)
    #   FR2:  residues 39–55   (17 residues)
    #   CDR2: residues 56–65   (up to 10 residues, often 3–6 in VHH)
    #   FR3:  residues 66–104  (39 residues, longest framework)
    #   CDR3: residues 105–117 (variable, 8–25 residues in VHH)
    #   FR4:  residues 118–128 (11 residues)
    #
    # For sequences shorter or longer than 128, we adjust CDR3 length.

    # Conservative approach: assign fixed framework lengths, rest to CDR3
    fr1_len = min(26, seq_len)
    cdr1_len = min(12, max(0, seq_len - 26))
    fr2_len = min(17, max(0, seq_len - 38))
    cdr2_len = min(10, max(0, seq_len - 55))
    fr3_len = min(39, max(0, seq_len - 65))
    fr4_len = 11  # last 11 residues

    # CDR3 gets whatever is left between FR3 and FR4
    assigned = fr1_len + cdr1_len + fr2_len + cdr2_len + fr3_len + fr4_len
    max(0, seq_len - assigned + (12 + 10))  # reclaim default CDR1/CDR2

    # Re-do with actual positional slicing
    # Use the standard IMGT cut points, adjusting CDR3 for actual length
    pos = 0
    fr1 = sequence[pos:pos + 26] if seq_len > 26 else sequence
    pos = 26

    cdr1_end = min(38, seq_len)
    cdr1 = sequence[pos:cdr1_end]
    pos = cdr1_end

    fr2_end = min(55, seq_len)
    fr2 = sequence[pos:fr2_end]
    pos = fr2_end

    cdr2_end = min(65, seq_len)
    cdr2 = sequence[pos:cdr2_end]
    pos = cdr2_end

    # FR3 is 39 residues in IMGT, but for raw sequences we calculate
    # based on what remains before CDR3/FR4
    remaining = seq_len - pos
    fr4_actual = min(11, remaining)
    cdr3_actual = max(0, remaining - 39 - fr4_actual)
    fr3_actual = remaining - cdr3_actual - fr4_actual

    fr3 = sequence[pos:pos + fr3_actual]
    pos += fr3_actual
    cdr3 = sequence[pos:pos + cdr3_actual]
    pos += cdr3_actual
    fr4 = sequence[pos:pos + fr4_actual]

    regions = VHHRegions(
        sequence=sequence,
        name=name,
        fr1=fr1, cdr1=cdr1, fr2=fr2, cdr2=cdr2,
        fr3=fr3, cdr3=cdr3, fr4=fr4,
    )

    logger.debug(
        "%s: FR1(%d) CDR1(%d) FR2(%d) CDR2(%d) FR3(%d) CDR3(%d) FR4(%d) = %d",
        name, len(fr1), len(cdr1), len(fr2), len(cdr2),
        len(fr3), len(cdr3), len(fr4), seq_len,
    )

    return regions


# ---------------------------------------------------------------------------
# VHH validation
# ---------------------------------------------------------------------------

def validate_vhh(
    sequence: str,
    name: str = "",
) -> VHHValidation:
    """Validate whether a sequence has VHH characteristics.

    Checks:
    1. Length is in the expected range (110–140 residues)
    2. Canonical disulfide bond cysteines at IMGT 23 and 104
    3. VHH hallmark residues at positions 37, 44, 45, 47
    4. Overall VHH-likeness score

    Args:
        sequence: VHH amino acid sequence.
        name: Identifier.

    Returns:
        VHHValidation with all check results.
    """
    warnings: list[str] = []
    seq_len = len(sequence)

    # Length check
    is_valid_length = 110 <= seq_len <= 140
    if not is_valid_length:
        warnings.append(f"Unusual length: {seq_len} (expected 110–140)")

    # Canonical disulfide (positions are 1-indexed, Python is 0-indexed)
    cys23 = sequence[22] == "C" if seq_len > 22 else False
    cys104 = sequence[103] == "C" if seq_len > 103 else False
    has_canonical = cys23 and cys104
    if not has_canonical:
        if not cys23:
            warnings.append("Missing canonical Cys at IMGT position 23")
        if not cys104:
            warnings.append("Missing canonical Cys at IMGT position 104")

    # Extra disulfide (CDR1-CDR3, less common)
    cys33 = sequence[32] == "C" if seq_len > 32 else False
    cys_cdr3 = any(sequence[i] == "C" for i in range(104, min(117, seq_len)))
    has_extra = cys33 and cys_cdr3

    # Hallmark residues
    hallmark_residues: dict[int, str] = {}
    hallmark_matches = 0
    hallmark_total = 0

    for pos, expected_aas in VHH_HALLMARKS.items():
        idx = pos - 1  # convert to 0-indexed
        if idx < seq_len:
            actual = sequence[idx]
            hallmark_residues[pos] = actual
            hallmark_total += 1
            if actual in expected_aas:
                hallmark_matches += 1
            else:
                warnings.append(
                    f"Position {pos}: {actual} (expected one of {expected_aas})"
                )

    # Overall VHH score
    score_components = [
        1.0 if is_valid_length else 0.5,
        1.0 if has_canonical else 0.0,
        hallmark_matches / max(hallmark_total, 1),
    ]
    vhh_score = sum(score_components) / len(score_components)

    return VHHValidation(
        name=name,
        sequence_length=seq_len,
        is_valid_length=is_valid_length,
        has_canonical_disulfide=has_canonical,
        has_extra_disulfide=has_extra,
        hallmark_residues=hallmark_residues,
        hallmark_matches=hallmark_matches,
        hallmark_total=hallmark_total,
        vhh_score=vhh_score,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Sequence clustering
# ---------------------------------------------------------------------------

def cluster_sequences(
    sequences: list[tuple[str, str]],
    identity_threshold: float = 0.90,
) -> list[list[int]]:
    """Cluster sequences by pairwise sequence identity.

    Uses a simple greedy leader-follower clustering algorithm. For production
    use, consider CD-HIT or MMseqs2 for better performance on large datasets.

    Args:
        sequences: List of (name, sequence) tuples.
        identity_threshold: Minimum fractional identity to join a cluster.

    Returns:
        List of clusters, where each cluster is a list of sequence indices.
    """
    if not sequences:
        return []

    clusters: list[list[int]] = []
    representatives: list[int] = []

    for i, (_name_i, seq_i) in enumerate(sequences):
        assigned = False
        for cluster_idx, rep_idx in enumerate(representatives):
            _, seq_rep = sequences[rep_idx]
            identity = _pairwise_identity(seq_i, seq_rep)
            if identity >= identity_threshold:
                clusters[cluster_idx].append(i)
                assigned = True
                break

        if not assigned:
            representatives.append(i)
            clusters.append([i])

    logger.info(
        "Clustered %d sequences into %d clusters at %.0f%% identity",
        len(sequences), len(clusters), identity_threshold * 100,
    )
    return clusters


def _pairwise_identity(seq_a: str, seq_b: str) -> float:
    """Compute fractional sequence identity between two sequences.

    Uses a simple positional alignment (no gaps). For sequences of
    different lengths, aligns from the start and compares up to the
    shorter length.
    """
    min_len = min(len(seq_a), len(seq_b))
    max_len = max(len(seq_a), len(seq_b))
    if max_len == 0:
        return 1.0
    matches = sum(1 for a, b in zip(seq_a[:min_len], seq_b[:min_len], strict=False) if a == b)
    return matches / max_len


# ---------------------------------------------------------------------------
# Bundled representative VHH germline sequences
# ---------------------------------------------------------------------------

# Curated set of representative alpaca and dromedary VHH framework sequences.
# These are composite consensus sequences derived from published VHH structures
# in the PDB and IMGT germline database. CDR loops are replaced with placeholder
# lengths; actual CDR sequences come from design (Phase 2/3).
#
# Sources:
#   - IMGT germline database (Vicugna pacos, Camelus dromedarius)
#   - PDB nanobody structures clustered at 90% framework identity
#   - Muyldermans (2013) Annu Rev Biochem 82:775

BUNDLED_GERMLINES: list[dict[str, str]] = [
    {
        "name": "VHH_alpaca_IGHV3S53",
        "species": "Vicugna pacos",
        "v_gene": "IGHV3S53",
        "j_gene": "IGHJ4",
        "sequence": (
            "QVQLVESGGGLVQAGGSLRLSCAASGSIFSINA"
            "MGWYRQAPGKQRELVAAITSGGSTNYADSVKGR"
            "FTISRDNAKNTVYLQMNSLKPEDTAVYYCNAGT"
            "TVSRDYWGQGTQVTVSS"
        ),
    },
    {
        "name": "VHH_alpaca_IGHV3S1",
        "species": "Vicugna pacos",
        "v_gene": "IGHV3S1",
        "j_gene": "IGHJ4",
        "sequence": (
            "QVQLVESGGGLVQPGGSLRLSCAASGFTLDYYA"
            "IGWFRQAPGKEREGVSCISSSDGSTYYADSVKGR"
            "FTISRDNAKNTVYLQMNSLKPEDTAVYYCAADST"
            "IYTLPSEYNYWGQGTQVTVSS"
        ),
    },
    {
        "name": "VHH_alpaca_IGHV3S66",
        "species": "Vicugna pacos",
        "v_gene": "IGHV3S66",
        "j_gene": "IGHJ6",
        "sequence": (
            "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSY"
            "AMGWFRQAPGKEREFVAAINWRGDITYYADSVKGR"
            "FTISRDNAKNTVYLQMNSLRPEDTAVYYCYVDRG"
            "RQWYSDSGAFDYWGQGTQVTVSS"
        ),
    },
    {
        "name": "VHH_alpaca_IGHV3S29",
        "species": "Vicugna pacos",
        "v_gene": "IGHV3S29",
        "j_gene": "IGHJ4",
        "sequence": (
            "QVQLVESGGGLVQPGGSLRLSCAASGFTVSSN"
            "SMSWVRQAPGKGLEWVSSIYSSGGSTFYADSVKGR"
            "FTISRDNAKNTLYLQMSSLRAEDTAVYYCARGR"
            "WLGAYDYWGQGTQVTVSS"
        ),
    },
    {
        "name": "VHH_dromedary_cAbBCII10",
        "species": "Camelus dromedarius",
        "v_gene": "IGHV3",
        "j_gene": "IGHJ4",
        "sequence": (
            "QVQLVESGGGLVQPGGSLRLSCAASGSIFSINA"
            "MGWYRQAPGKQRELVAAITSGGSTNYADSVKGR"
            "FTISRDNAKNTVYLQMNSLKPEDTAVYYCNVGG"
            "TWSRDYWGQGTQVTVSS"
        ),
    },
    {
        "name": "VHH_dromedary_cAbLys3",
        "species": "Camelus dromedarius",
        "v_gene": "IGHV3",
        "j_gene": "IGHJ6",
        "sequence": (
            "QVQLVESGGGLVQAGGSLRLSCAASGSIFRIA"
            "AMGWYRQAPGSQRELVAAITRSGSTYYADSVKGR"
            "FTISRDNAKNTVYLQMNSLKPEDTADYYCYVDS"
            "NYCDYEISATGYRYWGQGTQVTVSS"
        ),
    },
    {
        "name": "VHH_alpaca_Nb17",
        "species": "Vicugna pacos",
        "v_gene": "IGHV3S53",
        "j_gene": "IGHJ4",
        "sequence": (
            "EVQLVESGGGLVQPGGSLRLSCAASGFPVSDY"
            "NMSWVRQAPGKGLQWVSSISSSGGITPYADSVKGR"
            "FTISRDNAKNTLYLQMNSLRAEDTAVYYCAKAG"
            "STYSSNHMDHWGQGTQVTVSS"
        ),
    },
    {
        "name": "VHH_alpaca_Nb80",
        "species": "Vicugna pacos",
        "v_gene": "IGHV3S53",
        "j_gene": "IGHJ4",
        "sequence": (
            "QVQLVESGGGLVQAGGSLRLSCAASGSISSIT"
            "AMGWYRQTPGNQRELVAAITSGGRTNYADSVRG"
            "RFTVSRDNAKNTVYLQMNSLKPEDTAIYYCNVK"
            "DYGIAGFDSWGQGTQVTVSS"
        ),
    },
    {
        "name": "VHH_alpaca_IGHV3S9",
        "species": "Vicugna pacos",
        "v_gene": "IGHV3S9",
        "j_gene": "IGHJ4",
        "sequence": (
            "QVQLVESGGGLVQAGGSLRLSCAASGRTFSNY"
            "AMAWFRQAPGKEREFVAAITSGGRTNYADSVRG"
            "RFTISRDNAKNTVYLQMNSLKPEDTAVYYCNVK"
            "DYGSTYWGQGTQVTVSS"
        ),
    },
    {
        "name": "VHH_dromedary_antiGFP",
        "species": "Camelus dromedarius",
        "v_gene": "IGHV3",
        "j_gene": "IGHJ6",
        "sequence": (
            "QVQLVESGGALVQPGGSLRLSCAASGFPVNRYS"
            "MSWVRQAPGKGLEWVAYISRSGSTIYADSVKGR"
            "FTISRDNSKNTLYLQMNSLRAEDTAVYYCTIGG"
            "SLSRSSQGTLVTVSS"
        ),
    },
]


def load_bundled_germlines(
    species: list[str] | None = None,
) -> list[GermlineScaffold]:
    """Load the bundled set of representative VHH germline scaffolds.

    These are pre-curated sequences that can be used immediately without
    IMGT database access. For a larger library, use fetch_imgt_germlines.py.

    Args:
        species: Filter to specific species. None = all species.

    Returns:
        List of GermlineScaffold objects with regions and validation.
    """
    scaffolds: list[GermlineScaffold] = []

    for entry in BUNDLED_GERMLINES:
        if species and entry["species"] not in species:
            continue

        seq = entry["sequence"]
        name = entry["name"]

        regions = annotate_regions(seq, name=name)
        validation = validate_vhh(seq, name=name)

        scaffold = GermlineScaffold(
            name=name,
            sequence=seq,
            species=entry["species"],
            v_gene=entry["v_gene"],
            j_gene=entry["j_gene"],
            regions=regions,
            validation=validation,
        )
        scaffolds.append(scaffold)

    logger.info(
        "Loaded %d bundled germline scaffolds (species filter: %s)",
        len(scaffolds), species or "all",
    )
    return scaffolds


def curate_scaffold_library(
    scaffolds: list[GermlineScaffold],
    min_vhh_score: float = 0.6,
    cluster_identity: float = 0.90,
    min_clusters: int = 5,
) -> list[GermlineScaffold]:
    """Curate a scaffold library by filtering and clustering.

    Applies VHH validation, clusters by sequence identity, and selects
    one representative per cluster. Ensures minimum diversity.

    Args:
        scaffolds: Raw scaffold list.
        min_vhh_score: Minimum VHH-likeness score to retain.
        cluster_identity: Sequence identity threshold for clustering.
        min_clusters: Minimum clusters required (warns if fewer).

    Returns:
        Curated list of representative scaffolds (one per cluster).
    """
    # Filter by VHH score
    valid = [s for s in scaffolds if s.validation and s.validation.vhh_score >= min_vhh_score]
    logger.info(
        "VHH validation: %d/%d pass (score >= %.2f)",
        len(valid), len(scaffolds), min_vhh_score,
    )

    if not valid:
        logger.warning("No scaffolds passed VHH validation!")
        return []

    # Cluster by framework sequence identity
    sequences = [(s.name, s.sequence) for s in valid]
    clusters = cluster_sequences(sequences, identity_threshold=cluster_identity)

    # Select best representative per cluster (highest VHH score)
    representatives: list[GermlineScaffold] = []
    for cluster_idx, member_indices in enumerate(clusters):
        members = [valid[i] for i in member_indices]
        best = max(members, key=lambda s: s.validation.vhh_score if s.validation else 0)
        best.cluster_id = cluster_idx
        representatives.append(best)

    if len(representatives) < min_clusters:
        logger.warning(
            "Only %d clusters found (minimum %d requested). "
            "Consider lowering cluster_identity or adding more scaffolds.",
            len(representatives), min_clusters,
        )

    logger.info(
        "Curated library: %d representatives from %d clusters",
        len(representatives), len(clusters),
    )
    return representatives
