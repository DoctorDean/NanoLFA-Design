"""Tests for developability filters."""

import pytest
from omegaconf import OmegaConf

from nanolfa.filters.developability import DevelopabilityFilter, KD_HYDROPHOBICITY


@pytest.fixture
def filter_config():
    return OmegaConf.create({
        "max_aggregation_score": 0.5,
        "aggregation_tool": "aggrescan3d",
        "cdr3_net_charge_range": [-2, 2],
        "total_net_charge_range": [-5, 5],
        "max_cdr_hydrophobicity": 0.55,
        "flag_motifs": ["NG", "NS", "DG", "DS", "M", "NX[ST]"],
        "max_liability_count": 2,
        "min_predicted_tm": 65.0,
        "min_humanness_score": 0.5,
    })


@pytest.fixture
def dev_filter(filter_config):
    return DevelopabilityFilter(filter_config)


@pytest.fixture
def mock_candidate():
    from nanolfa.core.pipeline import Candidate
    return Candidate(
        candidate_id="test_001",
        sequence="QVQLVESGGGLVQAGGSLRLSCAASGSIFSINAMGWYRQAPGKQRELVAAITSGGSTNYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCNAGTTVSRDYWGQGTQVTVSS",
        round_created=1,
    )


class TestLiabilityScanning:
    """Test sequence liability motif detection."""

    def test_detects_ng_motif(self, dev_filter):
        seq = "AAAAANGAAAA"
        liabilities = dev_filter._scan_liabilities(seq)
        assert any("NG" in l for l in liabilities)

    def test_detects_ds_motif(self, dev_filter):
        seq = "AAAADSAAAA"
        liabilities = dev_filter._scan_liabilities(seq)
        assert any("DS" in l for l in liabilities)

    def test_detects_met_oxidation(self, dev_filter):
        seq = "AAAAMAAAA"
        liabilities = dev_filter._scan_liabilities(seq)
        assert any("M" in l for l in liabilities)

    def test_clean_sequence_no_liabilities(self, dev_filter):
        seq = "QVQLVESGGGLVQAGG"  # no liability motifs
        liabilities = dev_filter._scan_liabilities(seq)
        # May find single M, but no NG/NS/DG/DS
        pattern_liabilities = [l for l in liabilities if "NG" in l or "DS" in l or "DG" in l]
        assert len(pattern_liabilities) == 0


class TestChargeCalculation:
    """Test charge computation."""

    def test_neutral_sequence(self, dev_filter):
        seq = "AAAAAAAAAAA"
        charge = dev_filter._compute_total_charge(seq)
        assert charge == 0.0

    def test_positive_charge(self, dev_filter):
        seq = "AAAKKKAAAA"
        charge = dev_filter._compute_total_charge(seq)
        assert charge == 3.0

    def test_negative_charge(self, dev_filter):
        seq = "AAADDDAAAA"
        charge = dev_filter._compute_total_charge(seq)
        assert charge == -3.0


class TestHydrophobicity:
    """Test hydrophobicity calculation."""

    def test_hydrophobic_sequence(self, dev_filter):
        # Build a sequence with known CDR positions being hydrophobic
        seq = "A" * 130  # long enough for all CDR positions
        # CDR residues include positions 26-38, 55-65, 104-117
        hydro = dev_filter._compute_cdr_hydrophobicity(seq)
        # Alanine has KD=1.8, normalized: (1.8+4.5)/9.0 ≈ 0.7
        assert 0.6 < hydro < 0.8

    def test_hydrophilic_sequence(self, dev_filter):
        seq = "D" * 130  # all Asp
        hydro = dev_filter._compute_cdr_hydrophobicity(seq)
        # Asp has KD=-3.5, normalized: (-3.5+4.5)/9.0 ≈ 0.11
        assert hydro < 0.2


class TestTmPrediction:
    """Test thermal stability prediction."""

    def test_baseline_tm(self, dev_filter, mock_candidate):
        tm = dev_filter._predict_tm(mock_candidate)
        # Should be close to baseline (72°C) for a normal VHH
        assert 65 < tm < 80

    def test_cys_penalty(self, dev_filter):
        from nanolfa.core.pipeline import Candidate
        cand = Candidate(
            candidate_id="cys_heavy",
            sequence="C" * 120,  # all cysteine — extreme case
            round_created=1,
        )
        tm = dev_filter._predict_tm(cand)
        # Should be heavily penalized
        assert tm < 65


class TestCompositeDevScore:
    """Test the developability meta-score."""

    def test_perfect_candidate(self, dev_filter):
        score = dev_filter._compute_composite(
            aggregation=0.0,
            cdr3_charge=0.0,
            hydrophobicity=0.0,
            liability_count=0,
            predicted_tm=85.0,
        )
        assert score == pytest.approx(1.0, abs=0.01)

    def test_terrible_candidate(self, dev_filter):
        score = dev_filter._compute_composite(
            aggregation=1.0,
            cdr3_charge=4.0,
            hydrophobicity=1.0,
            liability_count=5,
            predicted_tm=50.0,
        )
        assert score < 0.2

    def test_score_bounded(self, dev_filter):
        """Score must be in [0, 1]."""
        import random
        random.seed(42)
        for _ in range(50):
            score = dev_filter._compute_composite(
                aggregation=random.uniform(0, 1),
                cdr3_charge=random.uniform(-5, 5),
                hydrophobicity=random.uniform(0, 1),
                liability_count=random.randint(0, 10),
                predicted_tm=random.uniform(40, 90),
            )
            assert 0.0 <= score <= 1.0
