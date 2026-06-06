"""
plot_delta_scatter_spearman.py
================================
Predicted vs Actual CER improvement (Δ_i) scatter plot with Spearman ρ.

Uses LassoCV with 10-fold CV (same as routing frontier) to produce
out-of-sample Δ̂_i predictions, then plots them against the true Δ_i.

Outputs:
  paper/figures/predicted_vs_actual_cer_loo_cv10.pdf
  paper/figures/predicted_vs_actual_cer_loo_cv10.png
"""

import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_gbt_classifier import (
    load_tesseract_records,
    load_llm_corrections,
    build_features,
)

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"
FIGS = BASE / "paper" / "figures"
FIGS.mkdir(exist_ok=True)


def main():
    print("Loading data...")
    records = load_tesseract_records()
    target_file = "corrections/tesseract/tesseract_Full_Expert_Robuste_8_google__gemini-3-flash-preview.json"
    corrections = load_llm_corrections(target_file)
    X = build_features(records)

    # Compute actual Δ_CER = baseline_cer - corrected_cer
    delta_cer = np.array([
        float(r["cer"]) - corrections.get(r["filename"], {}).get("cer", float(r["cer"]))
        for r in records
    ], dtype=np.float32)

    # Train LassoCV with 10-fold CV to get out-of-sample predictions
    print("Training LassoCV delta CER regression (10-fold CV)...")
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', LassoCV(cv=5, max_iter=5000, random_state=42))
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_dc = cross_val_predict(pipe, X, delta_cer, cv=cv)

    # Compute correlation statistics
    spearman_rho, spearman_p = stats.spearmanr(delta_cer, pred_dc)
    pearson_r, pearson_p = stats.pearsonr(delta_cer, pred_dc)

    print(f"  Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.2e})")
    print(f"  Spearman ρ = {spearman_rho:.4f}  (p = {spearman_p:.2e})")

    # ── Classify points into quadrants for coloring ──
    # True positive:  Δ > 0 and Δ̂ > 0  (correctly predicted improvement)  → blue
    # True negative:  Δ ≤ 0 and Δ̂ ≤ 0  (correctly predicted no benefit)  → grey
    # False positive: Δ ≤ 0 and Δ̂ > 0  (predicted help but hurt/same)    → grey
    # False negative: Δ > 0 and Δ̂ ≤ 0  (missed a good doc)               → red/salmon
    # Overcorrection: Δ < 0 (correction hurt)                             → red/salmon
    colors = []
    for actual, predicted in zip(delta_cer, pred_dc):
        if actual < -0.01:
            colors.append("#e74c3c")    # red — correction hurt
        elif actual > 0.01 and predicted > 0:
            colors.append("#3498db")    # blue — correctly predicted improvement
        elif actual > 0.01 and predicted <= 0:
            colors.append("#e74c3c")    # red — missed improvement
        else:
            colors.append("#95a5a6")    # grey — near zero or correctly identified as ~0

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    ax.scatter(delta_cer, pred_dc, c=colors, alpha=0.55, s=22,
               edgecolors="none", zorder=3)

    # Perfect prediction line (y=x)
    lims = [-0.5, 0.5]
    ax.plot(lims, lims, "--", color="#555555", linewidth=1.2,
            alpha=0.7, zorder=2, label="Perfect prediction")

    # Set same limits for both axes and make the aspect ratio square
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    # Quadrant lines at 0
    ax.axhline(0, color="#f39c12", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="#f39c12", linestyle=":", linewidth=0.8, alpha=0.5)

    # Annotation box with Spearman ρ
    txt = f"Spearman ρ = {spearman_rho:.3f}"
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.85, edgecolor="#cccccc", linewidth=0.8))

    ax.set_xlabel(r"Actual CER improvement $\Delta_i$", fontsize=12)
    ax.set_ylabel(r"Predicted $\hat{\Delta}_i$", fontsize=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85,
              facecolor="white", edgecolor="#cccccc")

    ax.tick_params(colors="black", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#d0d0d0")
    ax.grid(True, color="#e8e8e8", linestyle="-", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save
    out_pdf = FIGS / "predicted_vs_actual_cer_loo_cv10.pdf"
    out_png = FIGS / "predicted_vs_actual_cer_loo_cv10.png"
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {out_pdf}")
    print(f"  ✓ {out_png}")
    from PIL import Image
    print("SAVED IMAGE DIMENSIONS:", Image.open(out_png).size)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
