"""
plot_routing_density.py
=======================
Produces a single-panel figure showing the fraction of documents saved from
LLM correction as a function of the routing threshold for:
  - True WER (oracle upper bound)
  - Ridge pred-WER
  - MLP pred-WER
  - Confidence (avg_conf < T triggers routing; shown as continuous sweep)

Data source: results/routing_threshold_sweep.json
             results/routing_table.json
             results/low_confidence_words_90_tesseract.json
Output:      results/routing_density.png
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

RESULTS_DIR  = Path(__file__).resolve().parent.parent.parent / "results"
SWEEP_FILE   = RESULTS_DIR / "routing/routing_threshold_sweep.json"
CONF_FILE    = RESULTS_DIR / "confidence_data/low_confidence_words_90_tesseract.json"
ROUTING_FILE = RESULTS_DIR / "routing/routing_table.json"
OUT_FILE     = RESULTS_DIR / "figures/routing_density.png"

# ── Palette (matches threshold_comparison.png style) ──────────────────────────
COLORS = {
    "True WER":              "#2ecc71",   # green   – oracle
    "Ridge WER Prediction":  "#e74c3c",   # red     – Ridge
    "NN WER Prediction":     "#3498db",   # blue    – MLP
}
LABEL_DISPLAY = {
    "True WER":              "True WER (oracle)",
    "Ridge WER Prediction":  "Ridge pred-WER",
    "NN WER Prediction":     "MLP pred-WER",
}
CONF_COLOR = "#95a5a6"             # grey    – confidence (both thresholds)

def load_sweep():
    with open(SWEEP_FILE, encoding="utf-8") as f:
        return json.load(f)          # list of {label, rows:[{threshold, frac_routed, ...}]}

def extract(series, label):
    """Return (thresholds, frac_routed) for a named series."""
    for s in series:
        if s["label"] == label:
            rows = s["rows"]
            T    = [r["threshold"]   for r in rows]
            frac = [r["frac_routed"] for r in rows]
            return np.array(T), np.array(frac)
    raise KeyError(f"Label '{label}' not found in sweep file. "
                   f"Available: {[s['label'] for s in series]}")

def load_confidence_sweep():
    """Return (thresholds, savings_pct) for confidence-based routing.

    Signal = (1 − avg_confidence): route a document if (1 − avg_conf) >= T.
    This is the same direction as WER-based routing (higher signal → correct).
    savings = fraction of documents NOT routed.
    """
    with open(CONF_FILE, encoding="utf-8") as f:
        conf_raw = json.load(f)   # dict: filename -> {avg_confidence: float, ...}

    with open(ROUTING_FILE, encoding="utf-8") as f:
        routing_table = json.load(f)

    tess_test = [
        r for r in routing_table
        if r.get("split") == "test"
        and r.get("engine") == "tesseract"
        and r.get("has_correction", False)
    ]
    n = len(tess_test)

    # Attach avg_confidence to each test row
    avg_confs = []
    for row in tess_test:
        entry = conf_raw.get(row["filename"])
        avg_conf = entry["avg_confidence"] if isinstance(entry, dict) else None
        avg_confs.append(avg_conf)

    # Build sweep: signal = 1 - avg_conf (comparable to pred-WER scale)
    thresholds = np.arange(0.0, 1.005, 0.01)
    savings_pct = []
    for T in thresholds:
        routed = sum(
            1 for c in avg_confs
            if c is not None and (1.0 - c) >= T
        )
        savings_pct.append(100.0 * (1.0 - routed / n))

    return thresholds, np.array(savings_pct)


def main():
    series = load_sweep()
    print(f"Available labels: {[s['label'] for s in series]}")

    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    # ── continuous routing curves ──────────────────────────────────────────────
    for label, color in COLORS.items():
        try:
            T, frac = extract(series, label)
        except KeyError as e:
            print(f"  WARNING: {e} — skipping.")
            continue
        pct = (1 - frac) * 100   # savings = fraction NOT routed
        display = LABEL_DISPLAY.get(label, label)
        ax.plot(T, pct, color=color, lw=2.2, label=display, zorder=3)
        ax.fill_between(T, pct, alpha=0.08, color=color, zorder=2)

    # ── confidence sweep line ──────────────────────────────────────────────────
    T_conf, savings_conf = load_confidence_sweep()
    ax.plot(T_conf, savings_conf, color=CONF_COLOR, lw=2.0, ls="--",
            label="Confidence routing", zorder=3)
    ax.fill_between(T_conf, savings_conf, alpha=0.06, color=CONF_COLOR, zorder=2)

    # ── reference lines ────────────────────────────────────────────────────────
    ax.axhline(100, color="#555", lw=0.8, ls=":", zorder=1, alpha=0.7)
    ax.text(0.96, 101.5, "100% (correct all)", ha="right", va="bottom",
            fontsize=6.5, color="#555")
    ax.axhline(0, color="#555", lw=0.8, ls=":", zorder=1, alpha=0.7)

    # ── formatting ─────────────────────────────────────────────────────────────
    ax.set_xlabel("Routing threshold $T$\n"
                  r"(route document if signal $\geq T$)",
                  fontsize=9)
    ax.set_ylabel("Documents saved from LLM correction (%)", fontsize=9)
    ax.set_title("Article density per routing model and threshold\n"
                 "(Tesseract test split, $n=238$)", fontsize=10, fontweight="bold")

    ax.set_xlim(0.04, 0.96)
    ax.set_ylim(-3, 108)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))

    ax.grid(axis="y", color="#ddd", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    legend = ax.legend(frameon=True, fontsize=7.5, loc="upper right",
                       framealpha=0.9, edgecolor="#ccc")
    legend.get_frame().set_facecolor("white")

    plt.savefig(OUT_FILE, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
