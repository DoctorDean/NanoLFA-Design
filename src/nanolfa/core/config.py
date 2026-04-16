"""Configuration loading and validation for NanoLFA-Design."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Required top-level keys in a valid config
_REQUIRED_KEYS = {"pipeline", "target", "alphafold", "proteinmpnn", "scoring", "filters"}


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> DictConfig:
    """Load and validate a hierarchical YAML configuration.

    Supports the `defaults` key for inheritance: a target config can reference
    the master default.yaml and overlay target-specific values.

    Args:
        config_path: Path to the YAML configuration file.
        overrides: Optional dict of dot-separated key overrides.

    Returns:
        Merged and validated OmegaConf DictConfig.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # Handle defaults/inheritance
    if "defaults" in cfg:
        base_names = cfg.pop("defaults")
        if not isinstance(base_names, list):
            base_names = [base_names]
        base_cfgs = []
        for base_name in base_names:
            base_path = config_path.parent / f"{base_name}.yaml"
            if not base_path.exists():
                # Try configs/ directory
                base_path = config_path.parent.parent / f"{base_name}.yaml"
            if base_path.exists():
                base_cfgs.append(OmegaConf.load(base_path))
            else:
                logger.warning("Base config '%s' not found at %s", base_name, base_path)
        # Merge: base configs first, then current config on top
        cfg = OmegaConf.merge(*base_cfgs, cfg)

    # Apply CLI overrides
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Resolve interpolations (e.g., ${oc.env:VAR})
    OmegaConf.resolve(cfg)

    # Validate
    _validate_config(cfg)

    logger.info("Loaded config for target=%s from %s", cfg.target.name, config_path)
    return cfg


def _validate_config(cfg: DictConfig) -> None:
    """Check that required sections and critical parameters are present."""
    missing = _REQUIRED_KEYS - set(cfg.keys())
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")

    if cfg.target.name is None:
        raise ValueError("target.name must be set (e.g., 'pdg' or 'e3g')")

    # Validate scoring weights sum to ~1.0
    weights = cfg.scoring.weights
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Scoring weights must sum to 1.0, got {total:.3f}")

    # Validate threshold consistency
    thresholds = cfg.scoring.thresholds
    if thresholds.advance.composite_score_min <= thresholds.borderline.composite_score_min:
        logger.warning(
            "Advance threshold (%.2f) should be > borderline (%.2f)",
            thresholds.advance.composite_score_min,
            thresholds.borderline.composite_score_min,
        )
