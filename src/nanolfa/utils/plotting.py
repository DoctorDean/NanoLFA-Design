"""Visualization helpers for NanoLFA-Design pipeline outputs.

Reusable plotting functions for notebooks and reports. All functions
return matplotlib Figure objects and optionally save to disk. Designed
for consistency across the six pipeline notebooks.

Style: dark theme with IBM Plex Sans typography, matching the
project's visual identity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from nanolfa.core.pipeline import RoundResult
    from nanolfa.models.md_validation import MDTrajectoryMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Color palette
COLORS = {
    "primary": "#2196F3",
    "secondary": "#E91E63",
    "green": "#4CAF50",
    "orange": "#FF9800",
    "red": "#F44336",
    "purple": "#9C27B0",
    "teal": "#009688",
    "grey": "#9E9E9E",
    "light_grey": "#BDBDBD",
}

PHASE_COLORS = {
    1: "#22d3ee",
    2: "#34d399",
    3: "#60a5fa",
    4: "#f97316",
    5: "#a78bfa",
    6: "#f472b6",
}

TIER_COLORS = {
    "green": "#4CAF50",
    "yellow": "#FF9800",
    "red": "#F44336",
    "specific": "#4CAF50",
    "borderline": "#FF9800",
    "cross-reactive": "#F44336",
}


def apply_style() -> None:
    """Apply the NanoLFA plot style globally."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Convergence plots (Phase 3)
# ---------------------------------------------------------------------------

def plot_convergence(
    round_results: list[RoundResult],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot score convergence across design rounds.

    Shows best score, mean score (top-K), and per-round improvement.

    Args:
        round_results: List of RoundResult from the design loop.
        save_path: If set, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    rounds = [r.round_number for r in round_results]
    best = [r.best_composite_score for r in round_results]
    mean = [r.mean_composite_score for r in round_results]

    # Score curves
    ax = axes[0]
    ax.plot(rounds, best, "o-", lw=2, ms=8, label="Best", color=COLORS["secondary"])
    ax.plot(rounds, mean, "s--", lw=2, ms=6, label="Mean (top-K)", color=COLORS["primary"])
    ax.axhline(y=0.70, color=COLORS["green"], ls=":", alpha=0.5, label="Green threshold")
    ax.axhline(y=0.50, color=COLORS["orange"], ls=":", alpha=0.5, label="Yellow threshold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Composite Score")
    ax.set_title("Score Convergence")
    ax.legend(fontsize=9)

    # Tier distribution
    ax = axes[1]
    greens = []
    yellows = []
    reds = []
    for r in round_results:
        g = sum(1 for c in r.advanced_candidates if c.tier == "green")
        y = sum(1 for c in r.advanced_candidates if c.tier == "yellow")
        rd = r.candidates_advanced - g - y
        greens.append(g)
        yellows.append(y)
        reds.append(rd)

    ax.bar(rounds, greens, color=TIER_COLORS["green"], label="Green")
    ax.bar(rounds, yellows, bottom=greens, color=TIER_COLORS["yellow"], label="Yellow")
    bottoms = [g + y for g, y in zip(greens, yellows, strict=False)]
    ax.bar(rounds, reds, bottom=bottoms, color=TIER_COLORS["red"], label="Red")
    ax.set_xlabel("Round")
    ax.set_ylabel("Count")
    ax.set_title("Tier Distribution")
    ax.legend()

    # Per-round delta
    ax = axes[2]
    deltas = [0.0] + [best[i] - best[i - 1] for i in range(1, len(best))]
    bar_colors = [
        COLORS["green"] if d > 0.02
        else COLORS["orange"] if d > 0
        else COLORS["red"]
        for d in deltas
    ]
    ax.bar(rounds, deltas, color=bar_colors)
    ax.axhline(y=0.02, color="black", ls="--", alpha=0.5, label="Convergence threshold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Score Delta")
    ax.set_title("Per-Round Improvement")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metric_distributions(
    round_results: list[RoundResult],
    save_path: Path | None = None,
) -> plt.Figure:
    """Box plots of per-metric distributions across design rounds.

    Args:
        round_results: List of RoundResult.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    apply_style()
    metrics = [
        ("iptm", "ipTM"),
        ("plddt_interface", "Interface pLDDT"),
        ("shape_complementarity", "Shape Complementarity"),
        ("binding_energy", "Binding Energy (REU)"),
        ("buried_surface_area", "BSA (Å²)"),
        ("composite_score", "Composite Score"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    n_rounds = len(round_results)

    for ax, (attr, label) in zip(axes.flat, metrics, strict=False):
        data = []
        for r in round_results:
            vals = [
                getattr(c, attr) for c in r.advanced_candidates
                if getattr(c, attr, None) is not None
            ]
            data.append(vals if vals else [0])

        bp = ax.boxplot(
            data,
            labels=[str(r.round_number) for r in round_results],
            patch_artist=True,
        )
        cmap = plt.cm.Blues(np.linspace(0.3, 0.8, n_rounds))
        for patch, color in zip(bp["boxes"], cmap, strict=False):
            patch.set_facecolor(color)

        ax.set_xlabel("Round")
        ax.set_ylabel(label)
        ax.set_title(label, size=11)

    plt.suptitle("Metric Distributions Across Design Rounds", size=14, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Specificity plots (Phase 4)
# ---------------------------------------------------------------------------

def plot_cross_reactivity_heatmap(
    candidate_ids: list[str],
    compound_names: list[str],
    on_target_scores: list[float],
    off_target_matrix: list[list[float]],
    tiers: list[str],
    save_path: Path | None = None,
) -> plt.Figure:
    """Heatmap of candidate × compound binding scores.

    Args:
        candidate_ids: Candidate identifiers.
        compound_names: Off-target compound names.
        on_target_scores: On-target composite score per candidate.
        off_target_matrix: 2D list [candidates][compounds] of off-target scores.
        tiers: Tier classification per candidate.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    apply_style()
    columns = ["ON-TARGET"] + compound_names
    matrix = np.zeros((len(candidate_ids), len(columns)))
    for i, ot_score in enumerate(on_target_scores):
        matrix[i, 0] = ot_score
        for j, ot_val in enumerate(off_target_matrix[i]):
            matrix[i, j + 1] = ot_val

    fig, ax = plt.subplots(figsize=(12, max(6, len(candidate_ids) * 0.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.8)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right", size=9)
    ax.set_yticks(range(len(candidate_ids)))

    labels = [f"{cid} [{t[0].upper()}]" for cid, t in zip(candidate_ids, tiers, strict=False)]
    ax.set_yticklabels(labels, size=8)

    for i, tier in enumerate(tiers):
        color = TIER_COLORS.get(tier, COLORS["grey"])
        ax.get_yticklabels()[i].set_color(color)

    plt.colorbar(im, ax=ax, label="Composite Score", shrink=0.6)
    ax.set_title("Cross-Reactivity Matrix")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# MD validation plots (Phase 3.5)
# ---------------------------------------------------------------------------

def plot_md_validation_summary(
    md_results: list[MDTrajectoryMetrics],
    save_path: Path | None = None,
) -> plt.Figure:
    """Summary plots for MD validation results.

    Shows ligand RMSD, contact persistence, CDR RMSF, and MD scores.

    Args:
        md_results: List of MDTrajectoryMetrics.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    names = [r.candidate_id for r in md_results]

    # Ligand RMSD
    ax = axes[0, 0]
    rmsd = [r.ligand_rmsd_mean for r in md_results]
    rmsd_err = [r.ligand_rmsd_std for r in md_results]
    colors = [COLORS["red"] if r.ligand_escaped else COLORS["green"] for r in md_results]
    ax.barh(names, rmsd, xerr=rmsd_err, color=colors, height=0.6, alpha=0.8)
    ax.axvline(x=5.0, color="red", ls="--", alpha=0.5, label="Escape threshold")
    ax.set_xlabel("Ligand RMSD (Å)")
    ax.set_title("Binding Pose Stability")
    ax.legend()
    ax.invert_yaxis()

    # Contact persistence
    ax = axes[0, 1]
    contacts = [r.contact_persistence for r in md_results]
    colors = [COLORS["green"] if c > 0.8 else COLORS["orange"] if c > 0.6 else COLORS["red"]
              for c in contacts]
    ax.barh(names, contacts, color=colors, height=0.6, alpha=0.8)
    ax.axvline(x=0.8, color="green", ls="--", alpha=0.5, label="Stable (>0.8)")
    ax.set_xlabel("Contact Persistence")
    ax.set_title("Native Contact Maintenance")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.invert_yaxis()

    # CDR RMSF
    ax = axes[1, 0]
    x = np.arange(len(names))
    w = 0.25
    ax.barh(x - w, [r.cdr1_rmsf for r in md_results], w, label="CDR1", color=PHASE_COLORS[1])
    ax.barh(x, [r.cdr2_rmsf for r in md_results], w, label="CDR2", color=PHASE_COLORS[2])
    ax.barh(x + w, [r.cdr3_rmsf for r in md_results], w, label="CDR3", color=PHASE_COLORS[3])
    ax.set_yticks(x)
    ax.set_yticklabels(names, size=8)
    ax.set_xlabel("RMSF (Å)")
    ax.set_title("CDR Loop Flexibility")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    # MD validation score
    ax = axes[1, 1]
    md_scores = [r.md_validation_score for r in md_results]
    colors = [COLORS["green"] if r.passes_validation else COLORS["red"] for r in md_results]
    ax.barh(names, md_scores, color=colors, height=0.6, alpha=0.8)
    ax.axvline(x=0.5, color="red", ls="--", alpha=0.5, label="Pass threshold")
    ax.set_xlabel("MD Validation Score")
    ax.set_title("Overall MD Assessment")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.invert_yaxis()

    plt.suptitle("Molecular Dynamics Validation", size=14, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Calibration plots (Phase 6)
# ---------------------------------------------------------------------------

def plot_correlation_scatter(
    metric_name: str,
    predicted: list[float],
    experimental: list[float],
    candidate_ids: list[str],
    r_squared: float,
    slope: float,
    intercept: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """Scatter plot of predicted vs experimental values with regression line.

    Args:
        metric_name: Name of the computational metric.
        predicted: Predicted values.
        experimental: Experimental values.
        candidate_ids: Labels for each point.
        r_squared: R² of the linear fit.
        slope: Regression slope.
        intercept: Regression intercept.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        predicted, experimental,
        s=50, alpha=0.7, edgecolors="white", c=COLORS["primary"],
    )

    # Regression line
    x_range = np.linspace(min(predicted), max(predicted), 100)
    ax.plot(x_range, slope * x_range + intercept, "r--", alpha=0.7, lw=2)

    ax.set_xlabel(f"{metric_name} (predicted)", size=12)
    ax.set_ylabel("Experimental", size=12)
    ax.set_title(f"{metric_name} vs Experimental\nR²={r_squared:.3f}", size=13)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
