"""Shared pytest fixtures for NanoLFA-Design test suite."""

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def sample_vhh_sequence():
    """A representative alpaca VHH nanobody sequence (~120 residues)."""
    return (
        "QVQLVESGGGLVQAGGSLRLSCAASGSIFSINA"
        "MGWYRQAPGKQRELVAAITSGGSTNYADSVKGR"
        "FTISRDNAKNTVYLQMNSLKPEDTAVYYCNAGT"
        "TVSRDYWGQGTQVTVSS"
    )


@pytest.fixture
def sample_candidate(sample_vhh_sequence):
    """A mock Candidate object for testing."""
    from nanolfa.core.pipeline import Candidate
    return Candidate(
        candidate_id="test_vhh_001",
        sequence=sample_vhh_sequence,
        round_created=1,
    )


@pytest.fixture
def minimal_config():
    """Minimal valid pipeline configuration for unit tests."""
    return OmegaConf.create({
        "pipeline": {
            "max_rounds": 2,
            "variants_per_round": 10,
            "top_k_advance": 5,
            "top_k_experimental": 3,
            "convergence_threshold": 0.02,
            "random_seed": 42,
            "output_dir": "/tmp/nanolfa_test",
            "log_level": "WARNING",
            "experiment_tracker": "none",
        },
        "target": {
            "name": "pdg",
            "smiles": "O=C(O)C1OC(OC2CCC3C4CCC5CC(O)CCC5(C)C4CCC23C)C(O)C(O)C1O",
            "mol_weight": 496.6,
            "carrier_protein": "BSA",
            "linker_chemistry": "carbodiimide",
            "linker_length_atoms": 6,
            "cross_reactivity_panel": [],
        },
        "alphafold": {
            "version": "af3",
            "install_path": "/opt/alphafold",
            "database_path": "/data/alphafold_db",
            "num_models": 1,
            "num_recycles": 3,
            "num_seeds": 1,
            "max_template_date": "2025-01-01",
            "use_templates": False,
            "use_amber_relax": False,
            "gpu_memory_gb": 16,
            "af3": {"diffusion_samples": 1, "ligand_format": "sdf"},
            "prescreening": {
                "enabled": False,
                "engine": "esmfold",
                "min_plddt_threshold": 60,
                "top_k_to_full_af": 10,
            },
        },
        "proteinmpnn": {
            "install_path": "/opt/ProteinMPNN",
            "weights": "v_48_020",
            "sampling_temperature": 0.1,
            "num_sequences": 10,
            "num_batches": 1,
            "backbone_noise": 0.02,
            "cdr_only": True,
            "fixed_positions": "framework",
            "aa_bias": {"C": -5.0, "M": -2.0},
        },
        "scoring": {
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
            "thresholds": {
                "hard_gates": {
                    "iptm_min": 0.60,
                    "plddt_interface_min": 65.0,
                    "binding_energy_max": -15.0,
                    "aggregation_score_max": 0.50,
                    "liability_count_max": 3,
                },
                "advance": {"composite_score_min": 0.70},
                "borderline": {"composite_score_min": 0.50},
                "specificity": {
                    "max_off_target_iptm": 0.50,
                    "min_selectivity_ratio": 2.0,
                    "off_target_binding_energy_max": -10.0,
                },
            },
            "convergence": {
                "metric": "composite_score",
                "window": 2,
                "min_delta": 0.02,
                "min_rounds": 2,
            },
        },
        "filters": {
            "max_aggregation_score": 0.5,
            "aggregation_tool": "aggrescan3d",
            "cdr3_net_charge_range": [-2, 2],
            "total_net_charge_range": [-5, 5],
            "max_cdr_hydrophobicity": 0.55,
            "flag_motifs": ["NG", "NS", "DG", "DS", "M"],
            "max_liability_count": 2,
            "min_predicted_tm": 65.0,
            "min_humanness_score": 0.5,
        },
    })
