"""
plot_delta_correlation.py
=========================
Creates a publication-ready figure showing delta (Δ = pred vs actual)
correlation scatter plots with R² values for Ridge and MLP regression.

Outputs:
  results/figures/regression/delta_correlation.png
  paper/figures/delta_correlation.png
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE   = Path(__file__).resolve().parent.parent.parent
RES    = BASE / "results"
FIGS   = BASE / "paper" / "figures"
FIGS.mkdir(exist_ok=True)
(RES / "figures" / "regression").mkdir(parents=True, exist_ok=True)

# ── Color palette (matching existing regression figures) ─────────────────────
TEAL   = "#4ecdc4"
AMBER  = "#f7d794"
VIOLET = "#a29bfe"
CORAL  = "#ff6b6b"
PANEL  = "#f5f5f5"
GRID   = "#d0d0d0"


def load_predictions():
    """Load raw prediction arrays for both models."""
    with open(RES / "ml_models/predictions_ridge_noemb.json") as f:
        ridge_preds = json.load(f)
    with open(RES / "ml_models/predictions_nn_noemb.json") as f:
        nn_preds = json.load(f)
    return ridge_preds, nn_preds


def extract_arrays(predictions, metric="wer"):
    """Extract actual and predicted arrays for a given metric."""
    actual = np.array([p[f"actual_{metric}"] for p in predictions])
    pred   = np.array([p[f"pred_{metric}"]   for p in predictions])
    return actual, pred


def compute_r2(actual, pred):
    """Compute R² (coefficient of determination)."""
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - ss_res / ss_tot


def fig_delta_correlation():
    """
    2×2 scatter grid: rows = WER / CER, columns = Ridge / MLP.
    Each panel shows predicted vs actual with regression line, R², and Pearson r.
    """
    ridge_preds, nn_preds = load_predictions()

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.patch.set_facecolor("white")

    models = [
        ("Ridge Regression (LOO-CV)", ridge_preds, TEAL),
        ("MLP Regression (10-fold CV)", nn_preds, VIOLET),
    ]
    metrics = [("WER", "wer"), ("CER", "cer")]

    for col, (model_name, preds, color) in enumerate(models):
        for row, (metric_label, metric_key) in enumerate(metrics):
            ax = axes[row][col]
            ax.set_facecolor(PANEL)

            actual, pred = extract_arrays(preds, metric_key)

            # Compute statistics
            r2 = compute_r2(actual, pred)
            pearson_r, pearson_p = stats.pearsonr(actual, pred)
            mae = np.mean(np.abs(actual - pred))

            # Scatter plot
            ax.scatter(actual, pred, c=color, alpha=0.45, s=18,
                       edgecolors="none", zorder=3)

            # Perfect prediction line (y=x)
            lims = [
                min(actual.min(), pred.min()) - 0.02,
                max(actual.max(), pred.max()) + 0.02,
            ]
            ax.plot(lims, lims, "--", color="#999999", linewidth=1.0,
                    alpha=0.7, zorder=2, label="y = x")

            # Best-fit regression line
            slope, intercept = np.polyfit(actual, pred, 1)
            x_fit = np.linspace(actual.min(), actual.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, "-", color=CORAL, linewidth=1.8,
                    alpha=0.85, zorder=4, label="Best fit")

            # Annotation box
            txt = (f"R² = {r2:.4f}\n"
                   f"r  = {pearson_r:.4f}\n"
                   f"MAE = {mae:.4f}")
            ax.text(0.04, 0.96, txt, transform=ax.transAxes,
                    ha="left", va="top", fontsize=9, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              alpha=0.85, edgecolor=color, linewidth=1.2))

            # Axes
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(colors="black", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(GRID)
            ax.xaxis.grid(True, color=GRID, linestyle="--", linewidth=0.4, zorder=0)
            ax.yaxis.grid(True, color=GRID, linestyle="--", linewidth=0.4, zorder=0)
            ax.set_axisbelow(True)

            # Labels
            if row == 1:
                ax.set_xlabel(f"Actual {metric_label}", fontsize=10, color="black")
            if col == 0:
                ax.set_ylabel(f"Predicted {metric_label}", fontsize=10, color="black")

            # Column titles (top row only)
            if row == 0:
                ax.set_title(model_name, fontsize=11, fontweight="bold",
                             color="black", pad=8)

            # Legend (bottom-right of each panel)
            ax.legend(loc="lower right", fontsize=7.5, framealpha=0.85,
                      facecolor="white", edgecolor="#cccccc")

    # Row labels on the right side
    for row, (metric_label, _) in enumerate(metrics):
        axes[row][1].annotate(
            f"{metric_label}",
            xy=(1.08, 0.5), xycoords="axes fraction",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="black", rotation=-90,
        )

    plt.tight_layout(rect=[0, 0, 0.96, 1.0])

    # Save
    out_results = RES / "figures/regression/delta_correlation.png"
    out_paper   = FIGS / "delta_correlation.png"
    plt.savefig(out_results, dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(out_paper, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {out_results}")
    print(f"  ✓ {out_paper}")


if __name__ == "__main__":
    print("Generating delta correlation figure …")
    fig_delta_correlation()
    print("\nDone ✓")
