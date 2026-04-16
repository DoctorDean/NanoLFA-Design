"""Tests for the composite scoring function."""

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def scoring_config():
    """Minimal scoring configuration for testing."""
    return OmegaConf.create({
        "weights": {
            "iptm": 0.25,
            "plddt_interface": 0.20,
            "shape_complementarity": 0.15,
            "binding_energy": 0.20,
            "buried_surface_area": 0.10,
            "developability": 0.10,
        },
        "normalization": {
            "iptm": {"min": 0.4, "max": 0.95},
            "plddt_interface": {"min": 50.0, "max": 95.0},
            "shape_complementarity": {"min": 0.40, "max": 0.80},
            "binding_energy": {"min": -50.0, "max": 0.0, "invert": True},
            "buried_surface_area": {"min": 300.0, "max": 900.0},
            "developability": {"min": 0.0, "max": 1.0},
        },
        "energy_engine": "rosetta",
        "interface_distance_cutoff": 5.0,
    })


@pytest.fixture
def scorer(scoring_config):
    from nanolfa.scoring.composite import CompositeScorer
    return CompositeScorer(scoring_config)


class TestNormalization:
    """Test metric normalization."""

    def test_normalize_standard(self, scorer):
        """Normal metric within range maps to [0, 1]."""
        # ipTM=0.675 should map to (0.675-0.4)/(0.95-0.4) = 0.5
        assert abs(scorer._normalize("iptm", 0.675) - 0.5) < 0.01

    def test_normalize_at_min(self, scorer):
        """Value at min maps to 0."""
        assert scorer._normalize("iptm", 0.4) == 0.0

    def test_normalize_at_max(self, scorer):
        """Value at max maps to 1."""
        assert scorer._normalize("iptm", 0.95) == 1.0

    def test_normalize_below_min_clips(self, scorer):
        """Value below min clips to 0."""
        assert scorer._normalize("iptm", 0.2) == 0.0

    def test_normalize_above_max_clips(self, scorer):
        """Value above max clips to 1."""
        assert scorer._normalize("iptm", 1.0) == 1.0

    def test_normalize_inverted(self, scorer):
        """Inverted metric: more negative = higher score."""
        # binding_energy: min=-50, max=0, inverted
        # -50 REU should → 1.0 (best)
        assert scorer._normalize("binding_energy", -50.0) == 1.0
        # 0 REU should → 0.0 (worst)
        assert scorer._normalize("binding_energy", 0.0) == 0.0
        # -25 REU should → 0.5
        assert abs(scorer._normalize("binding_energy", -25.0) - 0.5) < 0.01


class TestCompositeScore:
    """Test the composite scoring function."""

    def test_perfect_scores(self, scorer):
        """All metrics at maximum should give composite ≈ 1.0."""
        raw = {
            "iptm": 0.95,
            "plddt_interface": 95.0,
            "shape_complementarity": 0.80,
            "binding_energy": -50.0,
            "buried_surface_area": 900.0,
        }
        score = scorer.composite(raw, developability_score=1.0)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_worst_scores(self, scorer):
        """All metrics at minimum should give composite ≈ 0.0."""
        raw = {
            "iptm": 0.4,
            "plddt_interface": 50.0,
            "shape_complementarity": 0.40,
            "binding_energy": 0.0,
            "buried_surface_area": 300.0,
        }
        score = scorer.composite(raw, developability_score=0.0)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_mid_range_scores(self, scorer):
        """Mid-range metrics should give composite ≈ 0.5."""
        raw = {
            "iptm": 0.675,
            "plddt_interface": 72.5,
            "shape_complementarity": 0.60,
            "binding_energy": -25.0,
            "buried_surface_area": 600.0,
        }
        score = scorer.composite(raw, developability_score=0.5)
        assert 0.4 < score < 0.6

    def test_weights_sum_to_one(self, scoring_config):
        """Verify weights sum to 1.0."""
        total = sum(scoring_config.weights.values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_score_bounded_zero_one(self, scorer):
        """Composite score must be in [0, 1] for any input."""
        import random
        random.seed(42)
        for _ in range(100):
            raw = {
                "iptm": random.uniform(0, 1),
                "plddt_interface": random.uniform(0, 100),
                "shape_complementarity": random.uniform(0, 1),
                "binding_energy": random.uniform(-80, 20),
                "buried_surface_area": random.uniform(0, 1500),
            }
            score = scorer.composite(raw, developability_score=random.uniform(0, 1))
            assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for {raw}"

    def test_higher_iptm_increases_score(self, scorer):
        """Increasing ipTM should increase composite score, all else equal."""
        base = {
            "iptm": 0.60,
            "plddt_interface": 75.0,
            "shape_complementarity": 0.65,
            "binding_energy": -30.0,
            "buried_surface_area": 500.0,
        }
        improved = dict(base)
        improved["iptm"] = 0.85

        score_base = scorer.composite(base, 0.7)
        score_improved = scorer.composite(improved, 0.7)
        assert score_improved > score_base


class TestDevelopabilityIntegration:
    """Test that developability score integrates correctly."""

    def test_dev_score_affects_composite(self, scorer):
        """Low developability should reduce composite score."""
        raw = {
            "iptm": 0.80,
            "plddt_interface": 85.0,
            "shape_complementarity": 0.70,
            "binding_energy": -35.0,
            "buried_surface_area": 500.0,
        }
        score_good_dev = scorer.composite(raw, developability_score=0.9)
        score_bad_dev = scorer.composite(raw, developability_score=0.1)
        assert score_good_dev > score_bad_dev
        # Developability weight is 0.10, so difference should be ~0.08
        diff = score_good_dev - score_bad_dev
        assert 0.05 < diff < 0.12
