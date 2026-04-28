"""Experimental feedback integration and scoring function recalibration.

Phase 6 of the NanoLFA-Design pipeline. Ingests wet-lab characterization
data (SPR kinetics, thermal shift, LFA signal), correlates with
computational predictions, recalibrates scoring weights, and identifies
hidden gems from prior rounds.

Workflow:
  6.1  Data ingestion — parse SPR, thermal shift, and LFA CSV files
  6.2  Correlation analysis — predicted vs. experimental metrics
  6.3  Weight recalibration — ridge regression or Bayesian optimization
  6.4  Re-ranking — apply recalibrated scoring to all prior candidates
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentalData:
    """Parsed experimental characterization data for one candidate."""

    candidate_id: str

    # SPR/BLI kinetics
    kon: float | None = None        # M⁻¹s⁻¹
    koff: float | None = None       # s⁻¹
    kd: float | None = None         # M (dissociation constant)
    spr_chi2: float | None = None   # goodness of fit

    # Thermal shift
    tm_celsius: float | None = None
    tagg_celsius: float | None = None

    # LFA prototype
    test_line_intensity: float | None = None
    control_line_intensity: float | None = None
    signal_noise_ratio: float | None = None
    detection_limit_ng_ml: float | None = None

    @property
    def log_kd(self) -> float | None:
        if self.kd is not None and self.kd > 0:
            return float(np.log10(self.kd))
        return None

    @property
    def has_kinetics(self) -> bool:
        return self.kd is not None

    @property
    def has_lfa(self) -> bool:
        return self.signal_noise_ratio is not None


@dataclass
class CorrelationResult:
    """Result of correlating a computational metric with experimental data."""

    metric_name: str
    experimental_metric: str
    n_points: int
    pearson_r: float
    spearman_rho: float
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    predicted_values: list[float]
    experimental_values: list[float]
    candidate_ids: list[str]


@dataclass
class RecalibrationResult:
    """Result of scoring function weight recalibration."""

    method: str  # "ridge", "bayesian", "manual"
    old_weights: dict[str, float]
    new_weights: dict[str, float]
    weight_changes: dict[str, float]
    training_r_squared: float
    cross_val_r_squared: float
    n_training_points: int
    feature_importances: dict[str, float]


@dataclass
class HiddenGem:
    """A candidate that was computationally underranked but experimentally strong."""

    candidate_id: str
    old_rank: int
    new_rank: int
    rank_improvement: int
    old_composite_score: float
    new_composite_score: float
    experimental_kd: float | None
    experimental_snr: float | None


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

def ingest_experimental_data(
    spr_path: str | Path | None = None,
    thermal_path: str | Path | None = None,
    lfa_path: str | Path | None = None,
) -> list[ExperimentalData]:
    """Parse experimental characterization CSV files.

    Expected CSV formats:
    - SPR: candidate_id, kon, koff, KD, chi2
    - Thermal: candidate_id, Tm_celsius, Tagg_celsius
    - LFA: candidate_id, test_line_intensity, control_line_intensity,
            signal_noise_ratio, detection_limit_ng_ml

    All files are optional; data is merged by candidate_id.

    Returns:
        List of ExperimentalData objects, one per candidate.
    """
    import csv

    data_by_id: dict[str, ExperimentalData] = {}

    def _get_or_create(cid: str) -> ExperimentalData:
        if cid not in data_by_id:
            data_by_id[cid] = ExperimentalData(candidate_id=cid)
        return data_by_id[cid]

    def _float_or_none(val: str) -> float | None:
        try:
            return float(val.strip())
        except (ValueError, AttributeError):
            return None

    # SPR kinetics
    if spr_path is not None:
        spr_file = Path(spr_path)
        if spr_file.exists():
            with open(spr_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("candidate_id", "").strip()
                    if not cid:
                        continue
                    entry = _get_or_create(cid)
                    entry.kon = _float_or_none(row.get("kon", ""))
                    entry.koff = _float_or_none(row.get("koff", ""))
                    entry.kd = _float_or_none(row.get("KD", row.get("kd", "")))
                    entry.spr_chi2 = _float_or_none(row.get("chi2", ""))
            logger.info("Loaded SPR data: %d candidates from %s", len(data_by_id), spr_file)

    # Thermal shift
    if thermal_path is not None:
        thermal_file = Path(thermal_path)
        if thermal_file.exists():
            with open(thermal_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("candidate_id", "").strip()
                    if not cid:
                        continue
                    entry = _get_or_create(cid)
                    entry.tm_celsius = _float_or_none(row.get("Tm_celsius", ""))
                    entry.tagg_celsius = _float_or_none(row.get("Tagg_celsius", ""))
            logger.info("Loaded thermal data from %s", thermal_file)

    # LFA prototype
    if lfa_path is not None:
        lfa_file = Path(lfa_path)
        if lfa_file.exists():
            with open(lfa_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("candidate_id", "").strip()
                    if not cid:
                        continue
                    entry = _get_or_create(cid)
                    entry.test_line_intensity = _float_or_none(
                        row.get("test_line_intensity", "")
                    )
                    entry.control_line_intensity = _float_or_none(
                        row.get("control_line_intensity", "")
                    )
                    entry.signal_noise_ratio = _float_or_none(
                        row.get("signal_noise_ratio", "")
                    )
                    entry.detection_limit_ng_ml = _float_or_none(
                        row.get("detection_limit_ng_ml", "")
                    )
            logger.info("Loaded LFA data from %s", lfa_file)

    results = list(data_by_id.values())
    logger.info(
        "Total: %d candidates with experimental data (%d with kinetics, %d with LFA)",
        len(results),
        sum(1 for d in results if d.has_kinetics),
        sum(1 for d in results if d.has_lfa),
    )
    return results


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_correlation(
    predicted: list[float],
    experimental: list[float],
    candidate_ids: list[str],
    metric_name: str = "metric",
    experimental_metric: str = "log_KD",
) -> CorrelationResult:
    """Compute correlation between predicted and experimental values.

    Args:
        predicted: Computational metric values.
        experimental: Experimental metric values.
        candidate_ids: Identifiers for each data point.
        metric_name: Name of the computational metric.
        experimental_metric: Name of the experimental metric.

    Returns:
        CorrelationResult with Pearson, Spearman, and linear fit.
    """
    from scipy import stats

    pred_arr = np.array(predicted)
    exp_arr = np.array(experimental)

    # Pearson correlation
    pearson_r, p_value = stats.pearsonr(pred_arr, exp_arr)

    # Spearman rank correlation
    spearman_rho, _ = stats.spearmanr(pred_arr, exp_arr)

    # Linear regression
    slope, intercept, r_value, _, _ = stats.linregress(pred_arr, exp_arr)

    return CorrelationResult(
        metric_name=metric_name,
        experimental_metric=experimental_metric,
        n_points=len(predicted),
        pearson_r=float(pearson_r),
        spearman_rho=float(spearman_rho),
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_value ** 2),
        p_value=float(p_value),
        predicted_values=predicted,
        experimental_values=experimental,
        candidate_ids=candidate_ids,
    )


def correlate_all_metrics(
    candidates_scores: dict[str, dict[str, float]],
    experimental_data: list[ExperimentalData],
    target_metric: str = "log_kd",
) -> list[CorrelationResult]:
    """Correlate all computational metrics with a target experimental metric.

    Args:
        candidates_scores: Dict of {candidate_id: {metric: value}}.
        experimental_data: Parsed experimental data.
        target_metric: Which experimental metric to correlate against.

    Returns:
        List of CorrelationResult, one per computational metric.
    """
    exp_lookup = {d.candidate_id: d for d in experimental_data}

    # Get experimental values
    exp_getter = {
        "log_kd": lambda d: d.log_kd,
        "kd": lambda d: d.kd,
        "kon": lambda d: d.kon,
        "tm": lambda d: d.tm_celsius,
        "snr": lambda d: d.signal_noise_ratio,
    }

    if target_metric not in exp_getter:
        raise ValueError(f"Unknown target metric: {target_metric}")

    get_exp = exp_getter[target_metric]

    # Find candidates with both computational and experimental data
    shared_ids = [
        cid for cid in candidates_scores
        if cid in exp_lookup and get_exp(exp_lookup[cid]) is not None
    ]

    if len(shared_ids) < 5:
        logger.warning(
            "Only %d candidates with both computational and experimental data "
            "(minimum 5 needed for meaningful correlation)", len(shared_ids),
        )
        return []

    # Get computational metric names
    first_scores = next(iter(candidates_scores.values()))
    metric_names = list(first_scores.keys())

    results: list[CorrelationResult] = []
    for metric in metric_names:
        predicted = [candidates_scores[cid][metric] for cid in shared_ids]
        experimental = [get_exp(exp_lookup[cid]) for cid in shared_ids]

        # Filter out None values
        valid = [
            (p, e, cid) for p, e, cid in zip(predicted, experimental, shared_ids, strict=False)
            if e is not None
        ]
        if len(valid) < 5:
            continue

        pred_vals = [v[0] for v in valid]
        exp_vals = [v[1] for v in valid]
        cids = [v[2] for v in valid]

        corr = compute_correlation(
            pred_vals, exp_vals, cids,
            metric_name=metric,
            experimental_metric=target_metric,
        )
        results.append(corr)

    results.sort(key=lambda r: abs(r.pearson_r), reverse=True)

    logger.info(
        "Correlation analysis: %d metrics vs %s (%d data points)",
        len(results), target_metric, len(shared_ids),
    )
    for r in results:
        logger.info(
            "  %s: Pearson=%.3f, Spearman=%.3f, R²=%.3f",
            r.metric_name, r.pearson_r, r.spearman_rho, r.r_squared,
        )

    return results


# ---------------------------------------------------------------------------
# Weight recalibration
# ---------------------------------------------------------------------------

def recalibrate_weights(
    correlations: list[CorrelationResult],
    current_weights: dict[str, float],
    method: str = "ridge",
    min_weight: float = 0.05,
) -> RecalibrationResult:
    """Recalibrate scoring function weights based on experimental correlations.

    Args:
        correlations: CorrelationResults from correlate_all_metrics.
        current_weights: Current scoring weights from config.
        method: Recalibration method ("ridge", "correlation", "manual").
        min_weight: Minimum weight for any metric (prevent zero-ing out).

    Returns:
        RecalibrationResult with old and new weights.
    """
    metric_names = list(current_weights.keys())

    if method == "ridge" and len(correlations) > 0:
        return _ridge_recalibration(
            correlations, current_weights, metric_names, min_weight,
        )
    elif method == "correlation":
        return _correlation_recalibration(
            correlations, current_weights, metric_names, min_weight,
        )
    else:
        logger.warning("Insufficient data for %s recalibration, returning current weights", method)
        return RecalibrationResult(
            method="none",
            old_weights=dict(current_weights),
            new_weights=dict(current_weights),
            weight_changes={k: 0.0 for k in current_weights},
            training_r_squared=0.0,
            cross_val_r_squared=0.0,
            n_training_points=0,
            feature_importances={k: 0.0 for k in current_weights},
        )


def _ridge_recalibration(
    correlations: list[CorrelationResult],
    current_weights: dict[str, float],
    metric_names: list[str],
    min_weight: float,
) -> RecalibrationResult:
    """Recalibrate using ridge regression on experimental data."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score

    # Build feature matrix: one row per candidate, one column per metric
    # Use the first correlation's candidate set
    corr_lookup = {c.metric_name: c for c in correlations}

    # Get shared candidate IDs across all metrics
    all_ids = None
    for c in correlations:
        ids_set = set(c.candidate_ids)
        all_ids = ids_set if all_ids is None else all_ids & ids_set

    if all_ids is None or len(all_ids) < 10:
        logger.warning("Too few shared candidates (%d) for ridge regression",
                        len(all_ids) if all_ids else 0)
        return RecalibrationResult(
            method="ridge_insufficient_data",
            old_weights=dict(current_weights),
            new_weights=dict(current_weights),
            weight_changes={k: 0.0 for k in current_weights},
            training_r_squared=0.0,
            cross_val_r_squared=0.0,
            n_training_points=len(all_ids) if all_ids else 0,
            feature_importances={k: 0.0 for k in current_weights},
        )

    shared_ids = sorted(all_ids)

    # Build feature matrix and target vector
    available_metrics = [m for m in metric_names if m in corr_lookup]
    features = np.zeros((len(shared_ids), len(available_metrics)))
    for j, metric in enumerate(available_metrics):
        corr = corr_lookup[metric]
        id_to_val = dict(zip(corr.candidate_ids, corr.predicted_values, strict=False))
        for i, cid in enumerate(shared_ids):
            features[i, j] = id_to_val.get(cid, 0.0)

    # Target = experimental values from the first correlation
    first_corr = correlations[0]
    id_to_exp = dict(zip(
        first_corr.candidate_ids, first_corr.experimental_values, strict=False,
    ))
    targets = np.array([id_to_exp.get(cid, 0.0) for cid in shared_ids])

    # Ridge regression with cross-validation
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=min(5, len(shared_ids)))
    ridge.fit(features, targets)

    training_r2 = float(ridge.score(features, targets))

    # Cross-validated R²
    cv_folds = min(5, len(shared_ids))
    cv_scores = cross_val_score(ridge, features, targets, cv=cv_folds, scoring="r2")
    cv_r2 = float(np.mean(cv_scores))

    # Convert coefficients to weights (normalize to sum to 1)
    raw_coefs = np.abs(ridge.coef_)
    raw_coefs = np.maximum(raw_coefs, min_weight)
    normalized = raw_coefs / raw_coefs.sum()

    new_weights = dict(current_weights)
    importances = {}
    for j, metric in enumerate(available_metrics):
        new_weights[metric] = float(normalized[j])
        importances[metric] = float(raw_coefs[j])

    # Ensure weights sum to 1
    total = sum(new_weights.values())
    new_weights = {k: v / total for k, v in new_weights.items()}

    weight_changes = {
        k: new_weights.get(k, 0) - current_weights.get(k, 0)
        for k in current_weights
    }

    logger.info(
        "Ridge recalibration: R²=%.3f (train), R²=%.3f (CV), %d points",
        training_r2, cv_r2, len(shared_ids),
    )

    return RecalibrationResult(
        method="ridge",
        old_weights=dict(current_weights),
        new_weights=new_weights,
        weight_changes=weight_changes,
        training_r_squared=training_r2,
        cross_val_r_squared=cv_r2,
        n_training_points=len(shared_ids),
        feature_importances=importances,
    )


def _correlation_recalibration(
    correlations: list[CorrelationResult],
    current_weights: dict[str, float],
    metric_names: list[str],
    min_weight: float,
) -> RecalibrationResult:
    """Simple recalibration using absolute correlation as importance."""
    corr_lookup = {c.metric_name: abs(c.pearson_r) for c in correlations}

    raw_weights: dict[str, float] = {}
    for metric in metric_names:
        raw_weights[metric] = max(corr_lookup.get(metric, min_weight), min_weight)

    total = sum(raw_weights.values())
    new_weights = {k: v / total for k, v in raw_weights.items()}

    weight_changes = {
        k: new_weights.get(k, 0) - current_weights.get(k, 0)
        for k in current_weights
    }

    return RecalibrationResult(
        method="correlation",
        old_weights=dict(current_weights),
        new_weights=new_weights,
        weight_changes=weight_changes,
        training_r_squared=0.0,
        cross_val_r_squared=0.0,
        n_training_points=sum(c.n_points for c in correlations),
        feature_importances=dict(corr_lookup),
    )


# ---------------------------------------------------------------------------
# Re-ranking and hidden gem detection
# ---------------------------------------------------------------------------

def find_hidden_gems(
    candidates_scores: dict[str, dict[str, float]],
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    experimental_data: list[ExperimentalData],
    top_n: int = 10,
) -> list[HiddenGem]:
    """Identify candidates that improve most under recalibrated scoring.

    A "hidden gem" is a candidate that was ranked low by the original
    scoring function but would rank much higher with the recalibrated
    weights — AND has strong experimental data to back it up.

    Args:
        candidates_scores: {candidate_id: {metric: normalized_value}}.
        old_weights: Original scoring weights.
        new_weights: Recalibrated weights.
        experimental_data: Parsed experimental data.
        top_n: Number of hidden gems to return.

    Returns:
        List of HiddenGem objects sorted by rank improvement.
    """
    exp_lookup = {d.candidate_id: d for d in experimental_data}

    # Score all candidates with both weight sets
    old_scores: list[tuple[str, float]] = []
    new_scores: list[tuple[str, float]] = []

    for cid, metrics in candidates_scores.items():
        old_s = sum(old_weights.get(m, 0) * v for m, v in metrics.items())
        new_s = sum(new_weights.get(m, 0) * v for m, v in metrics.items())
        old_scores.append((cid, old_s))
        new_scores.append((cid, new_s))

    # Rank by score (descending)
    old_scores.sort(key=lambda x: x[1], reverse=True)
    new_scores.sort(key=lambda x: x[1], reverse=True)

    old_rank = {cid: rank + 1 for rank, (cid, _) in enumerate(old_scores)}
    new_rank = {cid: rank + 1 for rank, (cid, _) in enumerate(new_scores)}
    old_score_map = dict(old_scores)
    new_score_map = dict(new_scores)

    # Find candidates with largest rank improvement
    gems: list[HiddenGem] = []
    for cid in candidates_scores:
        improvement = old_rank[cid] - new_rank[cid]
        if improvement <= 0:
            continue

        exp = exp_lookup.get(cid)
        gems.append(HiddenGem(
            candidate_id=cid,
            old_rank=old_rank[cid],
            new_rank=new_rank[cid],
            rank_improvement=improvement,
            old_composite_score=old_score_map[cid],
            new_composite_score=new_score_map[cid],
            experimental_kd=exp.kd if exp else None,
            experimental_snr=exp.signal_noise_ratio if exp else None,
        ))

    gems.sort(key=lambda g: g.rank_improvement, reverse=True)

    logger.info(
        "Hidden gems: %d candidates improved rank, top improvement: %d positions",
        len(gems), gems[0].rank_improvement if gems else 0,
    )

    return gems[:top_n]


# ---------------------------------------------------------------------------
# Save calibration
# ---------------------------------------------------------------------------

def save_calibration(
    recalibration: RecalibrationResult,
    correlations: list[CorrelationResult],
    output_dir: Path,
) -> Path:
    """Save calibration results as a JSON file for config update.

    The output JSON can be used to update configs/scoring.yaml with
    the recalibrated weights and correlation metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "method": recalibration.method,
        "n_training_points": recalibration.n_training_points,
        "training_r_squared": round(recalibration.training_r_squared, 4),
        "cross_val_r_squared": round(recalibration.cross_val_r_squared, 4),
        "old_weights": {k: round(v, 4) for k, v in recalibration.old_weights.items()},
        "new_weights": {k: round(v, 4) for k, v in recalibration.new_weights.items()},
        "weight_changes": {k: round(v, 4) for k, v in recalibration.weight_changes.items()},
        "feature_importances": {
            k: round(v, 4) for k, v in recalibration.feature_importances.items()
        },
        "correlations": [
            {
                "metric": c.metric_name,
                "experimental": c.experimental_metric,
                "pearson_r": round(c.pearson_r, 4),
                "spearman_rho": round(c.spearman_rho, 4),
                "r_squared": round(c.r_squared, 4),
                "n_points": c.n_points,
            }
            for c in correlations
        ],
    }

    json_path = output_dir / "calibration.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Calibration saved to %s", json_path)
    return json_path
