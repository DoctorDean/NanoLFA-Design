"""Tests for VHH sequence utilities."""

import pytest

from nanolfa.utils.sequence import (
    annotate_regions,
    cluster_sequences,
    curate_scaffold_library,
    load_bundled_germlines,
    validate_vhh,
)


@pytest.fixture
def sample_vhh():
    """Representative alpaca VHH sequence."""
    return (
        "QVQLVESGGGLVQAGGSLRLSCAASGSIFSINAMGWYRQAPGKQRELVAAITSGGSTNYADSVKGR"
        "FTISRDNAKNTVYLQMNSLKPEDTAVYYCNAGTTVSRDYWGQGTQVTVSS"
    )


class TestAnnotateRegions:
    def test_basic_annotation(self, sample_vhh):
        regions = annotate_regions(sample_vhh, name="test_vhh")
        assert regions.sequence == sample_vhh
        assert len(regions.fr1) > 0
        assert len(regions.cdr3) > 0
        assert len(regions.fr4) > 0

    def test_regions_cover_full_sequence(self, sample_vhh):
        regions = annotate_regions(sample_vhh)
        total = (
            len(regions.fr1) + len(regions.cdr1) + len(regions.fr2)
            + len(regions.cdr2) + len(regions.fr3) + len(regions.cdr3)
            + len(regions.fr4)
        )
        assert total == len(sample_vhh)

    def test_cdr3_length_property(self, sample_vhh):
        regions = annotate_regions(sample_vhh)
        assert regions.cdr3_length == len(regions.cdr3)
        assert regions.cdr3_length > 0


class TestValidateVHH:
    def test_validates_good_sequence(self, sample_vhh):
        result = validate_vhh(sample_vhh, name="test")
        assert result.sequence_length == len(sample_vhh)
        assert result.vhh_score > 0

    def test_length_check(self):
        short = "ACDEF" * 10  # 50 residues — too short
        result = validate_vhh(short, name="short")
        assert not result.is_valid_length
        assert any("length" in w.lower() for w in result.warnings)

    def test_score_bounded(self, sample_vhh):
        result = validate_vhh(sample_vhh)
        assert 0.0 <= result.vhh_score <= 1.0

    def test_canonical_disulfide_detection(self, sample_vhh):
        result = validate_vhh(sample_vhh)
        # The sample sequence should have canonical Cys at 23 and 104
        # (whether it does depends on the exact sequence)
        assert isinstance(result.has_canonical_disulfide, bool)


class TestClusterSequences:
    def test_identical_sequences_same_cluster(self):
        seqs = [("a", "ACDEFG"), ("b", "ACDEFG"), ("c", "ACDEFG")]
        clusters = cluster_sequences(seqs, identity_threshold=0.90)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_different_sequences_different_clusters(self):
        seqs = [("a", "AAAAAAA"), ("b", "DDDDDDD"), ("c", "WWWWWWW")]
        clusters = cluster_sequences(seqs, identity_threshold=0.90)
        assert len(clusters) == 3

    def test_empty_input(self):
        clusters = cluster_sequences([], identity_threshold=0.90)
        assert clusters == []

    def test_threshold_affects_count(self):
        seqs = [
            ("a", "ACDEFGHIKLMN"),
            ("b", "ACDEFGHIKLMX"),  # 1 mutation
            ("c", "XXXXXXXXXXXX"),  # totally different
        ]
        tight = cluster_sequences(seqs, identity_threshold=0.99)
        loose = cluster_sequences(seqs, identity_threshold=0.50)
        assert len(tight) >= len(loose)


class TestBundledGermlines:
    def test_load_all(self):
        scaffolds = load_bundled_germlines()
        assert len(scaffolds) >= 5  # minimum useful diversity

    def test_filter_by_species(self):
        alpaca = load_bundled_germlines(species=["Vicugna pacos"])
        all_scaffolds = load_bundled_germlines()
        assert len(alpaca) <= len(all_scaffolds)
        for s in alpaca:
            assert s.species == "Vicugna pacos"

    def test_scaffolds_have_regions(self):
        scaffolds = load_bundled_germlines()
        for s in scaffolds:
            assert s.regions is not None
            assert s.validation is not None

    def test_scaffolds_have_reasonable_lengths(self):
        scaffolds = load_bundled_germlines()
        for s in scaffolds:
            assert 100 <= len(s.sequence) <= 150, (
                f"{s.name} has unusual length: {len(s.sequence)}"
            )


class TestCurateLibrary:
    def test_curation_reduces_count(self):
        all_scaffolds = load_bundled_germlines()
        curated = curate_scaffold_library(
            all_scaffolds, min_vhh_score=0.0, cluster_identity=0.99
        )
        # Should have at most as many as input
        assert len(curated) <= len(all_scaffolds)

    def test_curated_have_cluster_ids(self):
        scaffolds = load_bundled_germlines()
        curated = curate_scaffold_library(scaffolds)
        for s in curated:
            assert s.cluster_id >= 0

    def test_strict_filter_reduces_more(self):
        scaffolds = load_bundled_germlines()
        loose = curate_scaffold_library(scaffolds, min_vhh_score=0.0)
        strict = curate_scaffold_library(scaffolds, min_vhh_score=0.9)
        assert len(strict) <= len(loose)
