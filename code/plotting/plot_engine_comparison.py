"""
plot_engine_comparison.py
=========================
Generates comparison plots across OCR engines (Tesseract, EasyOCR, PaddleOCR),
now that PaddleOCR results have been added to summary.json.

Outputs (saved to results/ and paper/figures/):
  1. engine_baseline_comparison.png  — Baseline WER/CER per engine (bar)
  2. engine_strategy_heatmap.png     — WER per strategy × engine (heatmap)
  3. engine_improvement_chart.png    — Best-per-strategy WER improvement over baseline
"""
 
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
C_TESS   = "#1565c0"   # navy blue
C_EASY   = "#e65100"   # burnt orange
C_PADDLE = "#2e7d32"   # forest green
C_PANEL  = "#f5f5f5"
C_GRID   = "#d0d0d0"

ENGINE_COLORS = {
    "tesseract": C_TESS,
    "easyocr":   C_EASY,
    "paddle":    C_PADDLE,
}
ENGINE_LABELS = {
    "tesseract": "Tesseract",
    "easyocr":   "EasyOCR",
    "paddle":    "PaddleOCR",
}
ENGINES = ["tesseract", "easyocr", "paddle"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   C_PANEL,
    "axes.edgecolor":   "#aaaaaa",
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "text.color":       "black",
})

# ── Load data ─────────────────────────────────────────────────────────────────
summary = json.loads((RESULTS / "summaries/summary.json").read_text())

# Baseline entry
baseline_entry = next((e for e in summary if e.get("strategy") == "baseline_no_llm"), None)
baseline = {eng: baseline_entry["by_ocr_engine"].get(eng, {}) for eng in ENGINES}

# Non-baseline entries
data = [e for e in summary if e.get("strategy") != "baseline_no_llm"]

# ── Figure 1: Baseline comparison ────────────────────────────────────────────
def fig_baseline_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="white")

    for ax, metric, title in [
        (axes[0], "wer", "Baseline WER (no LLM correction)"),
        (axes[1], "cer", "Baseline CER (no LLM correction)"),
    ]:
        ax.set_facecolor(C_PANEL)
        vals   = [baseline[eng].get(metric, 0) for eng in ENGINES]
        labels = [ENGINE_LABELS[e] for e in ENGINES]
        colors = [ENGINE_COLORS[e] for e in ENGINES]

        bars = ax.bar(labels, vals, color=colors, alpha=0.88, width=0.5,
                      edgecolor="none", zorder=3)

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="black")

        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.yaxis.grid(True, color=C_GRID, linestyle="--", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(C_GRID)

    plt.tight_layout()
    out = RESULTS / "figures/engine_comparison/engine_baseline_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    shutil.copy(out, FIGURES / out.name)
    plt.close()
    print(f"  ✓  {out.name}")


# ── Figure 2: Strategy heatmap (Full strategies only, WER) ───────────────────
def fig_strategy_heatmap():
    full_strategies = sorted(set(
        e["strategy"] for e in data if e["strategy"].startswith("Full_")
    ))

    # Build matrix: rows = strategies, cols = engines
    matrix_wer = np.full((len(full_strategies), len(ENGINES)), np.nan)
    matrix_cer = np.full((len(full_strategies), len(ENGINES)), np.nan)

    for e in data:
        strat = e["strategy"]
        if strat not in full_strategies:
            continue
        row = full_strategies.index(strat)
        for col, eng in enumerate(ENGINES):
            eng_data = e.get("by_ocr_engine", {}).get(eng, {})
            if "wer" in eng_data:
                # Keep the minimum (best) if multiple LLMs exist for same strategy
                cur = matrix_wer[row, col]
                new = eng_data["wer"]
                matrix_wer[row, col] = new if np.isnan(cur) else min(cur, new)
            if "cer" in eng_data:
                cur = matrix_cer[row, col]
                new = eng_data["cer"]
                matrix_cer[row, col] = new if np.isnan(cur) else min(cur, new)

    # Nice strategy labels
    labels = [s.replace("Full_", "").replace("_", " ") for s in full_strategies]
    engine_labels = [ENGINE_LABELS[e] for e in ENGINES]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor="white")

    for ax, matrix, title, baseline_vals in [
        (axes[0], matrix_wer, "Best WER per Strategy & Engine",
         [baseline[eng].get("wer", np.nan) for eng in ENGINES]),
        (axes[1], matrix_cer, "Best CER per Strategy & Engine",
         [baseline[eng].get("cer", np.nan) for eng in ENGINES]),
    ]:
        ax.set_facecolor(C_PANEL)
        # Append baseline row
        baseline_row_wer = np.array([baseline[eng].get("wer", np.nan) for eng in ENGINES])
        baseline_row_cer = np.array([baseline[eng].get("cer", np.nan) for eng in ENGINES])
        baseline_row = baseline_row_wer if "WER" in title else baseline_row_cer

        full_matrix = np.vstack([matrix, baseline_row])
        full_labels = labels + ["(Baseline)"]

        im = ax.imshow(full_matrix, aspect="auto", cmap="RdYlGn_r",
                       vmin=0.0, vmax=max(0.6, np.nanmax(full_matrix)))
        ax.set_xticks(range(len(engine_labels)))
        ax.set_xticklabels(engine_labels, fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(full_labels)))
        ax.set_yticklabels(full_labels, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

        # Annotate cells
        for i in range(full_matrix.shape[0]):
            for j in range(full_matrix.shape[1]):
                val = full_matrix[i, j]
                if not np.isnan(val):
                    txt_color = "white" if val > 0.35 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7.5, color=txt_color, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    plt.tight_layout()
    out = RESULTS / "figures/engine_comparison/engine_strategy_heatmap.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    shutil.copy(out, FIGURES / out.name)
    plt.close()
    print(f"  ✓  {out.name}")


# ── Figure 3: Best improvement over baseline per engine ─────────────────────
def fig_improvement_chart():
    """
    For each engine, show: baseline WER vs best-strategy WER (by that engine).
    Also show the absolute improvement.
    """
    # Find best WER per engine across all strategies
    best = {eng: {"wer": np.inf, "strategy": ""} for eng in ENGINES}
    for e in data:
        for eng in ENGINES:
            eng_data = e.get("by_ocr_engine", {}).get(eng, {})
            if eng_data.get("wer", np.inf) < best[eng]["wer"]:
                best[eng]["wer"]      = eng_data["wer"]
                best[eng]["strategy"] = e["strategy"] + " / " + e.get("llm_model", "")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.set_facecolor(C_PANEL)

    x     = np.arange(len(ENGINES))
    w     = 0.30
    labels = [ENGINE_LABELS[e] for e in ENGINES]
    b_wers = [baseline[eng].get("wer", 0) for eng in ENGINES]
    best_wers = [best[eng]["wer"] for eng in ENGINES]

    b1 = ax.bar(x - w/2, b_wers,    w*1.8, label="Baseline (raw OCR)",
                color=[ENGINE_COLORS[e] for e in ENGINES], alpha=0.45,
                edgecolor="none", zorder=3, hatch="//")
    b2 = ax.bar(x + w/2, best_wers, w*1.8, label="Best corrected strategy",
                color=[ENGINE_COLORS[e] for e in ENGINES], alpha=0.9,
                edgecolor="none", zorder=3)

    # Improvement arrows + labels
    for i, (bv, bst) in enumerate(zip(b_wers, best_wers)):
        impr = (bv - bst) / bv * 100
        ax.annotate("", xy=(i + w/2, bst), xytext=(i + w/2, bv),
                    arrowprops=dict(arrowstyle="->", color="black",
                                    lw=1.5), zorder=5)
        ax.text(i + w/2 + 0.08, (bv + bst)/2,
                f"−{impr:.1f}%", va="center", fontsize=8.5,
                color="black", fontweight="bold")

    # Value labels
    for bar, val in [(b, v) for b, v in zip(b1, b_wers)] + \
                    [(b, v) for b, v in zip(b2, best_wers)]:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Average WER", fontsize=11)
    ax.set_ylim(0, max(b_wers) * 1.30)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.grid(True, color=C_GRID, linestyle="--", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)
    ax.legend(fontsize=9, framealpha=0.95, edgecolor=C_GRID)

    # Annotation: best strategy per engine
    for i, eng in enumerate(ENGINES):
        strat_label = best[eng]["strategy"].split("/")[0].strip()
        ax.text(i, -0.04, strat_label, ha="center", va="top",
                fontsize=6.5, color="grey", transform=ax.get_xaxis_transform(),
                style="italic")

    plt.tight_layout()
    out = RESULTS / "figures/engine_comparison/engine_improvement_chart.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    shutil.copy(out, FIGURES / out.name)
    plt.close()
    print(f"  ✓  {out.name}")


# ── Figure 4: Grouped bar — best Full strategy WER per engine ───────────────
def fig_full_strategy_bars():
    """
    One grouped bar per strategy (Full_* only), 3 bars per group (one per engine).
    """
    full_entries = sorted(
        (e for e in data if e["strategy"].startswith("Full_")),
        key=lambda e: e.get("overall_average_wer", 1.0)
    )

    # Deduplicate by strategy (keep best LLM)
    seen, deduped = set(), []
    for e in full_entries:
        if e["strategy"] not in seen:
            seen.add(e["strategy"])
            deduped.append(e)

    strategies = [e["strategy"].replace("Full_", "").replace("_", " ") for e in deduped]
    n = len(strategies)
    x = np.arange(n)
    w = 0.22

    fig, ax = plt.subplots(figsize=(max(12, n * 1.5), 5.5), facecolor="white")
    ax.set_facecolor(C_PANEL)

    for offset, eng in zip([-1, 0, 1], ENGINES):
        wers = [e.get("by_ocr_engine", {}).get(eng, {}).get("wer", np.nan)
                for e in deduped]
        bars = ax.bar(x + offset * w, wers, w * 1.7,
                      label=ENGINE_LABELS[eng],
                      color=ENGINE_COLORS[eng], alpha=0.88,
                      edgecolor="none", zorder=3)
        for bar, val in zip(bars, wers):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=6.5, fontweight="bold", color="black")

    # Baseline horizontal lines
    for eng in ENGINES:
        bv = baseline[eng].get("wer", np.nan)
        if not np.isnan(bv):
            ax.axhline(bv, color=ENGINE_COLORS[eng], lw=1.2,
                       linestyle=":", alpha=0.6,
                       label=f"{ENGINE_LABELS[eng]} baseline ({bv:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Average WER", fontsize=11)
    ax.set_ylim(0, max(
        baseline[eng].get("wer", 0) for eng in ENGINES
    ) * 1.30)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.grid(True, color=C_GRID, linestyle="--", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)
    ax.legend(fontsize=8, ncol=3, framealpha=0.95, edgecolor=C_GRID,
              loc="upper right")

    plt.tight_layout()
    out = RESULTS / "figures/engine_comparison/engine_full_strategy_bars.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    shutil.copy(out, FIGURES / out.name)
    plt.close()
    print(f"  ✓  {out.name}")


if __name__ == "__main__":
    print("Generating engine comparison figures …\n")
    fig_baseline_comparison()
    fig_strategy_heatmap()
    fig_improvement_chart()
    fig_full_strategy_bars()
    print(f"\nAll figures saved to results/ and paper/figures/ ✓")
