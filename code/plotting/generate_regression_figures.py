"""
generate_regression_figures.py
================================
Creates publication-ready figures for the regression experiments section.

Outputs (all written to results/ and paper/figures/):
  1. regression_comparison.png  — side-by-side model comparison bar chart
  2. regression_scatter.png     — 2×2 scatter plots (WER/CER × Linear/NN)
  3. regression_features.png    — top-10 feature importance comparison
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

BASE   = Path(__file__).resolve().parent.parent.parent
RES    = BASE / "results"
FIGS   = BASE / "paper" / "figures"
FIGS.mkdir(exist_ok=True)

DARK   = "white"
PANEL  = "#f5f5f5"
TEAL   = "#4ecdc4"
AMBER  = "#f7d794"
VIOLET = "#a29bfe"
CORAL  = "#ff6b6b"
PINK   = "#fd79a8"
GRID   = "#d0d0d0"
WHITE  = "black"


def load_results():
    lr = json.loads((RES / "ml_models/linear_regression_pooled_noemb_results.json").read_text())
    nn = json.loads((RES / "ml_models/nn_regression_pooled_noemb_results.json").read_text())
    return lr, nn


# ── Figure 1: Model Comparison Bar Chart ────────────────────────────────────
def fig_comparison(lr, nn):
    metrics = ["R²", "MAE", "RMSE", "Pearson r"]
    wer_lr  = [lr["test_wer"]["R2"], lr["test_wer"]["MAE"],
                lr["test_wer"]["RMSE"], lr["test_wer"]["pearson_r"]]
    cer_lr  = [lr["test_cer"]["R2"], lr["test_cer"]["MAE"],
                lr["test_cer"]["RMSE"], lr["test_cer"]["pearson_r"]]
    wer_nn  = [nn["test_wer"]["R2"], nn["test_wer"]["MAE"],
                nn["test_wer"]["RMSE"], nn["test_wer"]["pearson_r"]]
    cer_nn  = [nn["test_cer"]["R2"], nn["test_cer"]["MAE"],
                nn["test_cer"]["RMSE"], nn["test_cer"]["pearson_r"]]

    x = np.arange(len(metrics))
    w = 0.20

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK)

    for ax, (wer_data, cer_data, title) in zip(axes, [
        (wer_lr, wer_nn, "WER Prediction"),
        (cer_lr, cer_nn, "CER Prediction"),
    ]):
        ax.set_facecolor(PANEL)
        ax.bar(x - w,     wer_data, w*1.8, label="Ridge (WER)", color=TEAL,   alpha=0.9)
        ax.bar(x + w,     cer_data, w*1.8, label="Ridge (CER)", color=AMBER,  alpha=0.9)
        # Reuse for NN
        # ax.set_title removed — caption in LaTeX
        ax.set_xticks(x); ax.set_xticklabels(metrics, color=WHITE, fontsize=10)
        ax.tick_params(colors=WHITE); ax.spines[:].set_color(GRID)
        ax.yaxis.grid(True, color=GRID, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values(): spine.set_color(GRID)

    # Redo properly with all four series
    for ax, (d1, d2, d3, d4, title) in zip(axes, [
        (wer_lr, wer_nn, cer_lr, cer_nn, "WER Prediction"),
        (wer_lr, wer_nn, cer_lr, cer_nn, "CER Prediction"),
    ]):
        ax.cla()
        ax.set_facecolor(PANEL)

    # Rebuild
    for ax, (lr_d, nn_d, metric_name) in zip(axes, [
        (wer_lr, wer_nn, "WER"),
        (cer_lr, cer_nn, "CER"),
    ]):
        ax.set_facecolor(PANEL)
        b1 = ax.bar(x - w*1.1, lr_d, w*2, label="Ridge Regression", color=TEAL,   alpha=0.88, zorder=3)
        b2 = ax.bar(x + w*1.1, nn_d, w*2, label="MLP Regression",   color=VIOLET, alpha=0.88, zorder=3)
        # value labels
        for bar in [*b1, *b2]:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                    ha="center", va="bottom", color="black", fontsize=7.5, fontweight="bold")
        # ax.set_title removed — caption in LaTeX
        ax.set_xticks(x); ax.set_xticklabels(metrics, color=WHITE, fontsize=10)
        ax.tick_params(colors=WHITE)
        for spine in ax.spines.values(): spine.set_color(GRID)
        ax.yaxis.grid(True, color=GRID, linestyle="--", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(facecolor="white", labelcolor="black", edgecolor="#cccccc", fontsize=9)
        ax.set_ylim(0, max(max(lr_d), max(nn_d)) * 1.18)

    plt.tight_layout()
    out = RES / "figures/regression/regression_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.savefig(FIGS / "figures/regression/regression_comparison.png", dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ {out}")


# ── Figure 2: Scatter Grid ───────────────────────────────────────────────────
def fig_scatter(lr, nn):
    """Reproduce scatter from saved result JSONs (we need raw predictions though).
    Since we don't store predictions in JSON, use stored metrics to annotate."""

    # Load raw prediction arrays from plots if available, else skip
    # Instead build an annotation-only figure summarising per-model metrics.
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.patch.set_facecolor(DARK)

    models = [("Ridge Regression", lr), ("MLP Regression", nn)]
    targets = [("WER", TEAL), ("CER", AMBER)]

    for col, (model_name, res) in enumerate(models):
        for row, (target, color) in enumerate(targets):
            ax = axes[row][col]
            ax.set_facecolor(PANEL)
            key = f"test_{target.lower()}"
            m = res[key]
            # Draw a metrics summary box
            txt = (
                f"R² = {m['R2']:.4f}\n"
                f"MAE = {m['MAE']:.4f}\n"
                f"RMSE = {m['RMSE']:.4f}\n"
                f"Pearson r = {m['pearson_r']:.4f}"
            )
            ax.text(0.5, 0.5, txt, transform=ax.transAxes,
                    ha="center", va="center", color="black",
                    fontsize=13, fontweight="bold", linespacing=2.0,
                    bbox=dict(boxstyle="round,pad=0.6", facecolor=color, alpha=0.18,
                              edgecolor=color, linewidth=1.5))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis("off")
            if row == 0:
                ax.set_xlabel(model_name, color="black", fontsize=11,
                              fontweight="bold", labelpad=6)
            if col == 0:
                ax.set_ylabel(f"{target} Metrics", color=color, fontsize=11,
                              labelpad=8, fontweight="bold")
                ax.yaxis.set_label_coords(-0.05, 0.5)
                ax.yaxis.label.set_visible(True)

    # suptitle removed — caption in LaTeX
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = RES / "figures/regression/regression_metrics_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.savefig(FIGS / "figures/regression/regression_metrics_summary.png", dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ {out}")


# ── Figure 3: Feature Importance ────────────────────────────────────────────
def fig_features(lr, nn):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK)

    # Ridge: use WER coefficients
    ridge_top = lr["top_features_wer"][:10]
    nn_top    = nn["top_features_permutation"][:10]

    for ax, (top, color, model) in zip(axes, [
        (ridge_top, TEAL,   "Ridge: Top WER Coefficients (absolute)"),
        (nn_top,    VIOLET, "MLP: Permutation Importance (WER+CER Δmse)"),
    ]):
        ax.set_facecolor(PANEL)
        names = [f[:26] for f, _ in top]
        vals  = [abs(v) if model.startswith("Ridge") else v for _, v in top]
        raw_v = [v for _, v in top]
        bar_colors = [color if v >= 0 else CORAL for v in raw_v]
        bars = ax.barh(range(len(names)), vals, color=bar_colors, edgecolor="none", zorder=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, color=WHITE, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("|Coefficient|" if "Ridge" in model else "Δ MSE", color=WHITE, fontsize=10)
        # ax.set_title removed — caption in LaTeX
        ax.tick_params(colors=WHITE)
        for spine in ax.spines.values(): spine.set_color(GRID)
        ax.xaxis.grid(True, color=GRID, linestyle="--", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        # value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", color="black", fontsize=7.5)

    # suptitle removed — caption in LaTeX
    plt.tight_layout()
    out = RES / "figures/regression/regression_features_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.savefig(FIGS / "figures/regression/regression_features_importance.png", dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ {out}")


if __name__ == "__main__":
    print("Generating regression figures …")
    lr, nn = load_results()
    fig_comparison(lr, nn)
    fig_scatter(lr, nn)
    fig_features(lr, nn)
    print("\nAll figures saved to results/ and paper/figures/ ✓")
